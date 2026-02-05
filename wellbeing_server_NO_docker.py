"""wellbeing_server_NO_docker.py

FastMCP server that provides deterministic tools for a personal diary:
- SQLite storage (entries + sessions)
- FTS5 word search
- FAISS semantic search (E5 embeddings)
- Optional CrossEncoder reranking

Compared to the original version, this rewrite focuses on production hygiene:
- Safer config/path handling (Windows-friendly, Unicode-safe atomic writes)
- Better concurrency (embed outside index lock)
- Better FTS query sanitization (no user-controlled MATCH syntax)
- Graceful shutdown (flush FAISS if dirty)
- Structured logging (no noisy prints)
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import json
import logging
import os
import re
import shutil
import signal
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ================== OFFLINE / ENV FIRST ==================
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

WELLBEING_OFFLINE = os.getenv("WELLBEING_OFFLINE", "true").lower() == "true"
if WELLBEING_OFFLINE:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

WELLBEING_PLOT = os.getenv("WELLBEING_PLOT", "matplotlib").lower()

# ================== NORMAL IMPORTS ==================
import aiosqlite
import faiss
import numpy as np
from fastmcp import FastMCP
from sentence_transformers import CrossEncoder, SentenceTransformer
from starlette.requests import Request
from starlette.responses import PlainTextResponse

HAVE_MPL = False
if WELLBEING_PLOT == "matplotlib":
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        HAVE_MPL = True
    except Exception:
        HAVE_MPL = False

# ================== LOGGING ==================
LOG_LEVEL = os.getenv("WELLBEING_LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("wellbeing")


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int, *, min_v: int | None = None, max_v: int | None = None) -> int:
    try:
        v = int(os.getenv(name, str(default)).strip())
    except Exception:
        v = default
    if min_v is not None:
        v = max(v, min_v)
    if max_v is not None:
        v = min(v, max_v)
    return v


def _env_float(name: str, default: float, *, min_v: float | None = None, max_v: float | None = None) -> float:
    try:
        v = float(os.getenv(name, str(default)).strip())
    except Exception:
        v = default
    if min_v is not None:
        v = max(v, min_v)
    if max_v is not None:
        v = min(v, max_v)
    return v


def _norm_path(p: str) -> str:
    """Normalize user-provided paths across platforms."""
    p = (p or "").strip().strip('"')
    if not p:
        return p
    return os.path.abspath(os.path.expanduser(p))


# ================== PATHS / CONFIG ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = _norm_path(os.getenv("WELLBEING_DB_PATH", os.path.join(BASE_DIR, "wellbeing.db")))
FAISS_INDEX_PATH = _norm_path(os.getenv("WELLBEING_FAISS_INDEX", os.path.join(BASE_DIR, "faiss.index")))
FAISS_MAP_PATH = _norm_path(os.getenv("WELLBEING_FAISS_MAP", os.path.join(BASE_DIR, "faiss_map.json")))

EMBED_MODEL_NAME = os.getenv("WELLBEING_EMBED_MODEL", "intfloat/multilingual-e5-small").strip()
RERANK_MODEL_NAME = os.getenv("WELLBEING_RERANK_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1").strip()
ENABLE_RERANK = _env_bool("WELLBEING_ENABLE_RERANK", True)

HOST = os.getenv("WELLBEING_HOST", "0.0.0.0").strip()
PORT = _env_int("WELLBEING_PORT", 8100, min_v=1, max_v=65535)
PATH = os.getenv("WELLBEING_PATH", "/mcp").strip() or "/mcp"
if not PATH.startswith("/"):
    PATH = "/" + PATH

FAISS_MIN_SCORE = _env_float("WELLBEING_FAISS_MIN_SCORE", 0.25, min_v=0.0, max_v=1.0)
FAISS_PERSIST_EVERY = _env_int("WELLBEING_FAISS_PERSIST_EVERY", 50, min_v=1, max_v=10_000)
AUTO_REINDEX_ON_START = _env_bool("WELLBEING_AUTO_REINDEX_ON_START", False)
RERANK_DEBUG = _env_bool("WELLBEING_RERANK_DEBUG", False)

# Text augmentation for FAISS only
AUGMENT_FOR_SEARCH = _env_bool("WELLBEING_AUGMENT_TEXT", True)
AUG_PREFIX_MIN = _env_int("WELLBEING_AUG_PREFIX_MIN", 3, min_v=2, max_v=8)
AUG_PREFIX_MAX = _env_int("WELLBEING_AUG_PREFIX_MAX", 5, min_v=AUG_PREFIX_MIN, max_v=12)
AUG_CHAR_N = _env_int("WELLBEING_AUG_CHAR_N", 3, min_v=2, max_v=6)
AUG_CHAR_MAX = _env_int("WELLBEING_AUG_CHAR_MAX", 120, min_v=10, max_v=1000)

log.info("WELLBEING server config: DB=%s FAISS=%s MAP=%s", DB_PATH, FAISS_INDEX_PATH, FAISS_MAP_PATH)
log.info("Models: embed=%s rerank=%s enable_rerank=%s offline=%s", EMBED_MODEL_NAME, RERANK_MODEL_NAME, ENABLE_RERANK, WELLBEING_OFFLINE)
log.info("Search: faiss_min_score=%.3f persist_every=%d augment=%s", FAISS_MIN_SCORE, FAISS_PERSIST_EVERY, AUGMENT_FOR_SEARCH)


# ================== DB HELPERS ==================
def now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


async def get_conn() -> aiosqlite.Connection:
    conn = await aiosqlite.connect(DB_PATH)
    conn.row_factory = lambda c, r: dict(zip([col[0] for col in c.description], r))

    def _regexp(pattern: str, text: Optional[str]) -> int:
        if text is None:
            return 0
        try:
            return 1 if re.search(pattern, text) else 0
        except Exception:
            return 0

    await conn.create_function("REGEXP", 2, _regexp)
    return conn


async def init_db() -> None:
    conn = await get_conn()
    try:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                raw_text TEXT NOT NULL,
                mood_score INTEGER NOT NULL,
                tags TEXT,
                interpretation TEXT
            )
            """
        )
        await conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts
            USING fts5(raw_text, tags, content='entries', content_rowid='id')
            """
        )
        await conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS entries_ai AFTER INSERT ON entries BEGIN
              INSERT INTO entries_fts(rowid, raw_text, tags)
              VALUES (new.id, new.raw_text, COALESCE(new.tags,''));
            END
            """
        )
        await conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS entries_ad AFTER DELETE ON entries BEGIN
              INSERT INTO entries_fts(entries_fts, rowid, raw_text, tags)
              VALUES('delete', old.id, old.raw_text, COALESCE(old.tags,''));
            END
            """
        )
        await conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS entries_au AFTER UPDATE ON entries BEGIN
              INSERT INTO entries_fts(entries_fts, rowid, raw_text, tags)
              VALUES('delete', old.id, old.raw_text, COALESCE(old.tags,''));
              INSERT INTO entries_fts(rowid, raw_text, tags)
              VALUES (new.id, new.raw_text, COALESCE(new.tags,''));
            END
            """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                chat_id INTEGER PRIMARY KEY,
                pending_text TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )
        await conn.commit()
        log.info("DB initialized")
    finally:
        await conn.close()


# ================== TEXT AUGMENT (FAISS ONLY) ==================
def _basic_tokens(text: str) -> List[str]:
    t = (text or "").lower()
    out: List[str] = []
    cur: List[str] = []
    for ch in t:
        ok = ("a" <= ch <= "z") or ("0" <= ch <= "9") or ("а" <= ch <= "я") or (ch == "ё")
        if ok:
            cur.append(ch)
        else:
            if cur:
                out.append("".join(cur))
                cur = []
    if cur:
        out.append("".join(cur))
    return out


def _augment_text(text: str) -> str:
    if not AUGMENT_FOR_SEARCH:
        return (text or "").strip()

    base = (text or "").strip()
    if not base:
        return base

    toks = _basic_tokens(base)

    pref: List[str] = []
    for w in toks:
        if len(w) < AUG_PREFIX_MIN:
            continue
        for L in range(AUG_PREFIX_MIN, min(AUG_PREFIX_MAX, len(w)) + 1):
            pref.append(w[:L])

    glued = "".join(toks)
    grams: List[str] = []
    if len(glued) >= AUG_CHAR_N:
        for i in range(0, len(glued) - AUG_CHAR_N + 1):
            grams.append(glued[i : i + AUG_CHAR_N])
            if len(grams) >= AUG_CHAR_MAX:
                break

    extra: List[str] = []
    if pref:
        extra.append("pref: " + " ".join(pref[:200]))
    if grams:
        extra.append("grams: " + " ".join(grams[:AUG_CHAR_MAX]))

    return base + ("\n" + "\n".join(extra) if extra else "")


# ================== FAISS STORE ==================
class FaissStore:
    """IndexFlatIP + normalize_embeddings=True => cosine similarity."""

    def __init__(self, index_path: str, map_path: str, embed_model_name: str):
        self.index_path = index_path
        self.map_path = map_path
        self.embed_model_name = embed_model_name

        self.embedder: Optional[SentenceTransformer] = None
        self.dim: int = 0

        self.index: Optional[faiss.Index] = None
        self.id_map: List[int] = []

        # Lock only protects index + id_map updates and persistence.
        self._lock = asyncio.Lock()
        self._dirty = 0

    def _ensure_embedder(self) -> None:
        if self.embedder is not None:
            return
        try:
            self.embedder = SentenceTransformer(self.embed_model_name)
            self.dim = int(self.embedder.get_sentence_embedding_dimension())
            log.info("Embedder loaded: %s dim=%s", self.embed_model_name, self.dim)
        except Exception as e:
            self.embedder = None
            self.dim = 0
            log.warning("Embedder init failed: %r", e)

    async def _embed_async(self, texts: Sequence[str], *, is_query: bool) -> np.ndarray:
        self._ensure_embedder()
        if self.embedder is None:
            raise RuntimeError("Embedder not available")

        prefix = "query: " if is_query else "passage: "
        prepared = [_augment_text((t or "").strip()) for t in texts]
        prepared = [prefix + t for t in prepared]

        def _encode() -> np.ndarray:
            vecs = self.embedder.encode(prepared, normalize_embeddings=True)
            return np.asarray(vecs, dtype="float32")

        return await asyncio.to_thread(_encode)

    def _create_new_index(self) -> None:
        self._ensure_embedder()
        if self.dim <= 0:
            self.index = None
            self.id_map = []
            return
        self.index = faiss.IndexFlatIP(self.dim)
        self.id_map = []

    def load_or_create(self) -> None:
        self._ensure_embedder()
        if self.dim <= 0:
            log.warning("FAISS disabled (no embedder)")
            self.index = None
            self.id_map = []
            return

        if os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
            except Exception as e:
                ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                broken_path = self.index_path + f".broken.{ts}"
                try:
                    shutil.move(self.index_path, broken_path)
                except Exception:
                    pass
                log.warning("Failed to read FAISS index; recreated. error=%r moved=%s", e, broken_path)
                self._create_new_index()
        else:
            self._create_new_index()

        if self.index is not None and getattr(self.index, "d", None) != self.dim:
            log.warning("FAISS dim mismatch (index=%s model=%s) -> recreate", getattr(self.index, "d", None), self.dim)
            self._create_new_index()

        if os.path.exists(self.map_path):
            try:
                with open(self.map_path, "r", encoding="utf-8") as f:
                    self.id_map = json.load(f)
            except Exception as e:
                ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                broken_map = self.map_path + f".broken.{ts}"
                try:
                    shutil.move(self.map_path, broken_map)
                except Exception:
                    pass
                log.warning("Failed to read FAISS map; reset. error=%r moved=%s", e, broken_map)
                self.id_map = []
        else:
            self.id_map = []

        ntotal = int(self.index.ntotal) if self.index else 0
        if self.index is not None and ntotal != len(self.id_map):
            log.warning("FAISS index/map mismatch ntotal=%s map_len=%s", ntotal, len(self.id_map))
        log.info("FAISS ready ntotal=%s", ntotal)

    def _atomic_write_json(self, path: str, data: Any) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        shutil.move(tmp, path)

    def _persist_sync(self) -> None:
        if self.index is None:
            return

        os.makedirs(os.path.dirname(os.path.abspath(self.index_path)) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(self.map_path)) or ".", exist_ok=True)

        # Unicode-safe: write to temp dir then move.
        tmp_dir = tempfile.gettempdir()
        tmp_index = os.path.join(tmp_dir, "wellbeing_faiss.index.tmp")
        faiss.write_index(self.index, tmp_index)
        shutil.move(tmp_index, self.index_path)

        self._atomic_write_json(self.map_path, self.id_map)

    async def persist_async(self) -> None:
        if self.index is None:
            return
        await asyncio.to_thread(self._persist_sync)

    async def add_entry(self, entry_id: int, text: str) -> None:
        if self.index is None:
            return

        # Compute embedding outside the lock (heavy work).
        vec = await self._embed_async([text], is_query=False)

        async with self._lock:
            if self.index is None:
                return
            self.index.add(vec)
            self.id_map.append(int(entry_id))
            self._dirty += 1
            if self._dirty >= FAISS_PERSIST_EVERY:
                await self.persist_async()
                self._dirty = 0

    async def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        if self.index is None or self.index.ntotal == 0:
            return []
        query = (query or "").strip()
        if not query:
            return []

        # Embed outside the lock.
        qv = await self._embed_async([query], is_query=True)

        async with self._lock:
            if self.index is None or self.index.ntotal == 0:
                return []
            k = min(max(int(top_k), 1), int(self.index.ntotal))
            scores, idxs = self.index.search(qv, k)

            scores_l = scores[0].tolist()
            idxs_l = idxs[0].tolist()

            out: List[Dict[str, Any]] = []
            for faiss_id, s in zip(idxs_l, scores_l):
                if faiss_id < 0 or faiss_id >= len(self.id_map):
                    continue
                out.append({"entry_id": int(self.id_map[faiss_id]), "score": float(s)})
            return out


# ================== RERANKER ==================
class Reranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model: Optional[CrossEncoder] = None
        self._lock = asyncio.Lock()

    async def _ensure_async(self) -> None:
        if self.model is not None:
            return
        async with self._lock:
            if self.model is not None:
                return
            try:
                self.model = await asyncio.to_thread(lambda: CrossEncoder(self.model_name))
                log.info("Reranker loaded: %s", self.model_name)
            except Exception as e:
                self.model = None
                log.warning("Reranker init failed: %r (disabled)", e)

    async def rerank_async(self, query: str, docs: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
        if not docs:
            return []
        query = (query or "").strip()
        if not query:
            return []
        top_n = max(int(top_n), 1)

        await self._ensure_async()
        if self.model is None:
            return docs[:top_n]

        pairs = [(query, (d.get("raw_text") or "")) for d in docs]
        scores = await asyncio.to_thread(lambda: self.model.predict(pairs))
        scores_f = [float(x) for x in scores]

        if RERANK_DEBUG and scores_f:
            log.info("RERANK q=%r n=%d best=%.4f worst=%.4f", query, len(docs), max(scores_f), min(scores_f))

        enriched: List[Dict[str, Any]] = []
        for d, s in zip(docs, scores_f):
            dd = dict(d)
            dd["rerank_score"] = s
            enriched.append(dd)
        enriched.sort(key=lambda x: x.get("rerank_score", -1e9), reverse=True)
        return enriched[:top_n]


# ================== SERVER ==================
mcp = FastMCP(
    name="WellbeingDiaryServer",
    instructions="MCP server: SQLite storage + FTS5 word search + FAISS semantic search + optional reranker.",
)

store = FaissStore(FAISS_INDEX_PATH, FAISS_MAP_PATH, EMBED_MODEL_NAME)
reranker: Optional[Reranker] = Reranker(RERANK_MODEL_NAME) if ENABLE_RERANK else None


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> PlainTextResponse:
    return PlainTextResponse("OK")


# ================== INTERNAL HELPERS ==================
async def _db_get_entries_by_ids(ids: Sequence[int]) -> List[Dict[str, Any]]:
    ids = [int(x) for x in ids if int(x) > 0]
    if not ids:
        return []

    conn = await get_conn()
    try:
        placeholders = ",".join(["?"] * len(ids))
        q = f"SELECT * FROM entries WHERE id IN ({placeholders})"
        cursor = await conn.execute(q, tuple(ids))
        rows = await cursor.fetchall()
        by_id = {int(r["id"]): r for r in rows}
        return [by_id[i] for i in ids if i in by_id]
    finally:
        await conn.close()


def _fts_safe_query(user_query: str) -> List[str]:
    """Build safe MATCH queries.

We **do not** allow raw MATCH syntax from the user because it can error easily
and can be abused to create pathological queries.

Strategy:
- tokenize into words
- build: word1 OR word2 ...
- build prefix version for longer words: word* OR ...
- build quoted phrase (escaped)
"""

    q = (user_query or "").strip()
    if not q:
        return []

    toks = _basic_tokens(q)
    toks = [t for t in toks if t]
    out: List[str] = []

    # Phrase query (still safe, we escape quotes)
    qq = q.replace('"', '""')
    out.append(f'"{qq}"')

    if toks:
        out.append(" OR ".join(toks[:12]))

        stems = []
        for t in toks:
            if len(t) >= 4:
                stems.append(t[: min(len(t), 8)] + "*")
        if stems:
            out.append(" OR ".join(stems[:12]))

    # Dedup
    seen: set[str] = set()
    res: List[str] = []
    for v in out:
        if v and v not in seen:
            seen.add(v)
            res.append(v)
    return res


async def _fts_candidates_ids(query: str, limit: int = 50) -> List[int]:
    q = (query or "").strip()
    if not q:
        return []
    limit = max(1, int(limit))

    conn = await get_conn()
    try:
        sql = (
            "SELECT e.id "
            "FROM entries_fts AS f "
            "JOIN entries AS e ON e.id = f.rowid "
            "WHERE entries_fts MATCH ? "
            "ORDER BY e.created_at DESC "
            "LIMIT ?"
        )

        for v in _fts_safe_query(q):
            try:
                cur = await conn.execute(sql, (v, limit))
                rows = await cur.fetchall()
                if rows:
                    return [int(r["id"]) for r in rows]
            except Exception:
                continue
        return []
    finally:
        await conn.close()


async def _impl_search_semantic_only(query: str, top_k: int = 20) -> List[Dict[str, Any]]:
    cands = await store.search(query, top_k=int(top_k))
    if not cands:
        return []

    ids = [int(c["entry_id"]) for c in cands]
    docs = await _db_get_entries_by_ids(ids)

    score_map = {int(c["entry_id"]): float(c["score"]) for c in cands}
    for d in docs:
        d["faiss_score"] = score_map.get(int(d["id"]), None)
    docs.sort(key=lambda x: x.get("faiss_score", -1e9), reverse=True)

    best = docs[0].get("faiss_score") if docs else None
    if best is None or float(best) < FAISS_MIN_SCORE:
        return []
    return docs


async def _impl_search_with_rerank(query: str, top_k: int = 50, top_n: int = 8) -> List[Dict[str, Any]]:
    query = (query or "").strip()
    if not query:
        return []

    top_k = max(1, min(int(top_k), 80))
    top_n = max(1, min(int(top_n), top_k))

    faiss_docs = await _impl_search_semantic_only(query=query, top_k=min(top_k, 50))
    fts_ids = await _fts_candidates_ids(query, limit=min(top_k, 50))
    fts_docs = await _db_get_entries_by_ids(fts_ids) if fts_ids else []

    by_id: Dict[int, Dict[str, Any]] = {}
    for d in faiss_docs:
        by_id[int(d["id"])] = d
    for d in fts_docs:
        i = int(d["id"])
        if i not in by_id:
            by_id[i] = d

    docs = list(by_id.values())
    if not docs:
        return []

    if reranker is None:
        # Deterministic order: prefer faiss_score then recency
        docs.sort(key=lambda x: (x.get("faiss_score") is not None, x.get("faiss_score", -1e9), x.get("created_at", "")), reverse=True)
        return docs[:top_n]

    return await reranker.rerank_async(query, docs, top_n=top_n)


# ================== TOOLS ==================
@mcp.tool
async def debug_paths() -> Dict[str, Any]:
    return {
        "base_dir": BASE_DIR,
        "db_path": DB_PATH,
        "faiss_index_path": FAISS_INDEX_PATH,
        "faiss_map_path": FAISS_MAP_PATH,
        "faiss_ntotal": int(store.index.ntotal) if store.index else None,
        "map_len": len(store.id_map),
        "embed_dim": store.dim,
        "embedder_loaded": store.embedder is not None,
        "rerank_enabled": ENABLE_RERANK,
        "rerank_model_name": RERANK_MODEL_NAME if ENABLE_RERANK else None,
        "rerank_model_loaded": (reranker.model is not None) if reranker else False,
        "offline": WELLBEING_OFFLINE,
        "have_mpl": HAVE_MPL,
        "augment_text": AUGMENT_FOR_SEARCH,
        "persist_every": FAISS_PERSIST_EVERY,
        "auto_reindex_on_start": AUTO_REINDEX_ON_START,
        "faiss_min_score": FAISS_MIN_SCORE,
    }


@mcp.tool
async def rebuild_faiss_from_db(batch_size: int = 256) -> Dict[str, Any]:
    batch_size = max(int(batch_size), 16)

    conn = await get_conn()
    try:
        cursor = await conn.execute("SELECT id, raw_text FROM entries ORDER BY id")
        rows = await cursor.fetchall()
    finally:
        await conn.close()

    store._ensure_embedder()
    if store.dim <= 0:
        return {"status": "skip", "reason": "embedder not available", "total_db_rows": len(rows)}

    # Build embeddings outside the lock in batches, then write under lock.
    ids_all: List[int] = []
    vecs_all: List[np.ndarray] = []
    added = 0

    texts: List[str] = []
    ids: List[int] = []
    for r in rows:
        ids.append(int(r["id"]))
        texts.append((r["raw_text"] or "").strip())
        if len(texts) >= batch_size:
            vecs_all.append(await store._embed_async(texts, is_query=False))
            ids_all.extend(ids)
            added += len(ids)
            texts, ids = [], []
    if texts:
        vecs_all.append(await store._embed_async(texts, is_query=False))
        ids_all.extend(ids)
        added += len(ids)

    async with store._lock:
        store.index = faiss.IndexFlatIP(store.dim)
        store.id_map = []
        for block in vecs_all:
            store.index.add(block)
        store.id_map.extend(ids_all)
        await store.persist_async()
        store._dirty = 0

    return {
        "status": "ok",
        "total_db_rows": len(rows),
        "added_to_faiss": added,
        "faiss_ntotal": int(store.index.ntotal) if store.index else 0,
        "map_len": len(store.id_map),
    }


@mcp.tool
async def rebuild_fts_from_db() -> Dict[str, Any]:
    conn = await get_conn()
    try:
        await conn.execute("INSERT INTO entries_fts(entries_fts) VALUES('rebuild')")
        await conn.commit()
        cur = await conn.execute("SELECT count(*) as c FROM entries_fts")
        c = int((await cur.fetchone())["c"])
        return {"status": "ok", "fts_rows": c}
    finally:
        await conn.close()


@mcp.tool
async def log_entry(raw_text: str, mood_score: int, tags: str = "", interpretation: str = "") -> Dict[str, Any]:
    if not 1 <= int(mood_score) <= 5:
        raise ValueError("mood_score must be 1..5")
    raw_text = (raw_text or "").strip()
    if not raw_text:
        raise ValueError("raw_text is empty")

    ts = now_iso()
    conn = await get_conn()
    try:
        await conn.execute(
            "INSERT INTO entries (created_at, raw_text, mood_score, tags, interpretation) VALUES (?, ?, ?, ?, ?)",
            (ts, raw_text, int(mood_score), tags or "", interpretation or ""),
        )
        await conn.commit()
        cursor = await conn.execute("SELECT last_insert_rowid() as id")
        entry_id = int((await cursor.fetchone())["id"])
    finally:
        await conn.close()

    # Background FAISS update (doesn't block response)
    async def _bg_add() -> None:
        try:
            await store.add_entry(entry_id, raw_text)
        except Exception as e:
            log.warning("FAISS add_entry failed: %r", e)

    asyncio.create_task(_bg_add())

    return {"status": "ok", "entry_id": entry_id, "created_at": ts}


@mcp.tool
async def get_daily_summary(date: str) -> Dict[str, Any]:
    date = (date or "").strip()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        raise ValueError("date must be YYYY-MM-DD")

    conn = await get_conn()
    try:
        cursor = await conn.execute(
            "SELECT * FROM entries WHERE substr(created_at, 1, 10) = ? ORDER BY created_at",
            (date,),
        )
        entries = await cursor.fetchall()
        if not entries:
            return {"date": date, "entries": [], "total_entries": 0, "avg_mood": None}

        cursor = await conn.execute(
            "SELECT COUNT(*) as c, AVG(mood_score) as avg_mood FROM entries WHERE substr(created_at, 1, 10) = ?",
            (date,),
        )
        stats = await cursor.fetchone()
        return {
            "date": date,
            "entries": entries,
            "total_entries": int(stats["c"]),
            "avg_mood": float(stats["avg_mood"]) if stats["avg_mood"] is not None else None,
        }
    finally:
        await conn.close()


@mcp.tool
async def search_word(query: str, limit: int = 20) -> List[Dict[str, Any]]:
    q = (query or "").strip()
    if not q:
        return []

    limit = max(1, min(int(limit), 200))

    # Small ё/е normalization variants
    def variants(s: str) -> List[str]:
        out = [s]
        if "е" in s:
            out.append(s.replace("е", "ё"))
        if "ё" in s:
            out.append(s.replace("ё", "е"))
        seen: set[str] = set()
        res: List[str] = []
        for x in out:
            if x not in seen:
                seen.add(x)
                res.append(x)
        return res

    conn = await get_conn()
    try:
        sql = (
            "SELECT e.* "
            "FROM entries_fts AS f "
            "JOIN entries AS e ON e.id = f.rowid "
            "WHERE entries_fts MATCH ? "
            "ORDER BY e.created_at DESC "
            "LIMIT ?"
        )

        # Try safe MATCH forms for each variant.
        for qq in variants(q):
            for v in _fts_safe_query(qq):
                try:
                    cur = await conn.execute(sql, (v, limit))
                    rows = await cur.fetchall()
                    if rows:
                        return rows
                except Exception:
                    continue

        # Fallback: LIKE
        like_sql = "SELECT * FROM entries WHERE raw_text LIKE ? ORDER BY created_at DESC LIMIT ?"
        for qq in variants(q):
            cur = await conn.execute(like_sql, (f"%{qq}%", limit))
            rows = await cur.fetchall()
            if rows:
                return rows
        return []
    finally:
        await conn.close()


@mcp.tool
async def search_semantic_only(query: str, top_k: int = 20) -> List[Dict[str, Any]]:
    return await _impl_search_semantic_only(query=query, top_k=top_k)


@mcp.tool
async def search_with_rerank(query: str, top_k: int = 50, top_n: int = 8) -> List[Dict[str, Any]]:
    return await _impl_search_with_rerank(query=query, top_k=top_k, top_n=top_n)


@mcp.tool
async def export_last_weeks_report(weeks: int = 4, out_dir: str = "reports") -> Dict[str, Any]:
    if not HAVE_MPL:
        return {"status": "skip", "reason": "matplotlib not available"}

    weeks = max(1, min(int(weeks), 52))
    today = _dt.date.today()
    date_from = today - _dt.timedelta(days=weeks * 7)

    conn = await get_conn()
    try:
        cur = await conn.execute(
            "SELECT created_at, mood_score FROM entries WHERE substr(created_at, 1, 10) >= ? ORDER BY created_at",
            (date_from.isoformat(),),
        )
        rows = await cur.fetchall()
    finally:
        await conn.close()

    by_day: Dict[str, List[int]] = {}
    for r in rows:
        d = (r.get("created_at") or "")[:10]
        m = r.get("mood_score")
        if not d or m is None:
            continue
        by_day.setdefault(d, []).append(int(m))

    os.makedirs(out_dir, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(out_dir, f"mood_{weeks}w_{ts}.png")

    fig = plt.figure()
    ax = fig.add_subplot(111)

    if not by_day:
        ax.set_title(f"Mood за последние {weeks} недель")
        ax.set_xlabel("Дата")
        ax.set_ylabel("Средний mood")
        ax.text(0.5, 0.5, "Нет записей за период", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        return {
            "status": "ok",
            "date_from": date_from.isoformat(),
            "date_to": today.isoformat(),
            "total_entries": 0,
            "plot_path": plot_path,
        }

    days = sorted(by_day.keys())
    avg_moods = [sum(by_day[d]) / len(by_day[d]) for d in days]
    counts = [len(by_day[d]) for d in days]
    total_entries = sum(counts)

    ax.plot(days, avg_moods, marker="o")
    ax.set_title(f"Mood за последние {weeks} недель ({date_from.isoformat()} → {today.isoformat()})")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Средний mood (1–5)")
    ax.set_ylim(1, 5)
    ax.tick_params(axis="x", rotation=45)

    for x, y, c in zip(days, avg_moods, counts):
        ax.annotate(str(c), (x, y), textcoords="offset points", xytext=(0, 6), ha="center")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    return {
        "status": "ok",
        "date_from": date_from.isoformat(),
        "date_to": today.isoformat(),
        "total_entries": total_entries,
        "plot_path": plot_path,
    }


# ================== SESSIONS TOOLS ==================
@mcp.tool
async def get_session(chat_id: int) -> Dict[str, Any]:
    conn = await get_conn()
    try:
        cur = await conn.execute("SELECT * FROM sessions WHERE chat_id = ?", (int(chat_id),))
        row = await cur.fetchone()
        if not row:
            return {"chat_id": int(chat_id), "pending_text": None}
        return {"chat_id": int(chat_id), "pending_text": row.get("pending_text")}
    finally:
        await conn.close()


@mcp.tool
async def set_session(chat_id: int, pending_text: str) -> Dict[str, Any]:
    conn = await get_conn()
    try:
        await conn.execute(
            """
            INSERT INTO sessions(chat_id, pending_text, updated_at)
            VALUES(?, ?, ?)
            ON CONFLICT(chat_id) DO UPDATE SET
              pending_text=excluded.pending_text,
              updated_at=excluded.updated_at
            """,
            (int(chat_id), pending_text or "", now_iso()),
        )
        await conn.commit()
        return {"status": "ok", "chat_id": int(chat_id), "pending_text": pending_text or ""}
    finally:
        await conn.close()


@mcp.tool
async def clear_session(chat_id: int) -> Dict[str, Any]:
    conn = await get_conn()
    try:
        await conn.execute(
            """
            INSERT INTO sessions(chat_id, pending_text, updated_at)
            VALUES(?, NULL, ?)
            ON CONFLICT(chat_id) DO UPDATE SET
              pending_text=NULL,
              updated_at=excluded.updated_at
            """,
            (int(chat_id), now_iso()),
        )
        await conn.commit()
        return {"status": "ok", "chat_id": int(chat_id), "pending_text": None}
    finally:
        await conn.close()


@mcp.resource("wellbeing://status")
async def wellbeing_status() -> str:
    conn = await get_conn()
    try:
        cursor = await conn.execute("SELECT COUNT(*) as c FROM entries")
        count = int((await cursor.fetchone())["c"])
        faiss_total = int(store.index.ntotal) if store.index else 0
        return json.dumps(
            {
                "status": "online",
                "entries_count": count,
                "faiss_total": faiss_total,
                "timestamp": now_iso(),
                "rerank_enabled": ENABLE_RERANK,
                "rerank_model_loaded": (reranker.model is not None) if reranker else False,
                "offline": WELLBEING_OFFLINE,
                "have_mpl": HAVE_MPL,
                "augment_text": AUGMENT_FOR_SEARCH,
                "faiss_min_score": FAISS_MIN_SCORE,
            },
            ensure_ascii=False,
        )
    finally:
        await conn.close()


# ================== STARTUP / SHUTDOWN ==================
async def _startup() -> None:
    await init_db()
    store.load_or_create()

    if AUTO_REINDEX_ON_START:
        conn = await get_conn()
        try:
            cur = await conn.execute("SELECT COUNT(*) as c FROM entries")
            total = int((await cur.fetchone())["c"])
        finally:
            await conn.close()

        faiss_total = int(store.index.ntotal) if store.index else 0
        if total > 0 and faiss_total == 0:
            log.info("AUTO_REINDEX: DB has entries but FAISS empty -> rebuilding")
            try:
                await rebuild_faiss_from_db(batch_size=256)
            except Exception as e:
                log.warning("AUTO_REINDEX failed: %r", e)


async def _flush_on_shutdown() -> None:
    try:
        async with store._lock:
            if store.index is not None and store._dirty > 0:
                log.info("Flushing FAISS (dirty=%d)", store._dirty)
                await store.persist_async()
                store._dirty = 0
    except Exception as e:
        log.warning("Flush on shutdown failed: %r", e)


def _install_signal_handlers() -> None:
    loop = asyncio.get_event_loop()

    async def _handle(sig: str) -> None:
        log.info("Signal %s received, flushing...", sig)
        await _flush_on_shutdown()
        raise SystemExit(0)

    for s in ("SIGINT", "SIGTERM"):
        if hasattr(signal, s):
            sig = getattr(signal, s)
            try:
                loop.add_signal_handler(sig, lambda s=s: asyncio.create_task(_handle(s)))
            except NotImplementedError:
                # Windows event loop may not support it.
                pass


if __name__ == "__main__":
    asyncio.run(_startup())
    try:
        _install_signal_handlers()
    except Exception:
        pass

    log.info("Server listening: http://127.0.0.1:%s%s", PORT, PATH)
    mcp.run(transport="streamable-http", host=HOST, port=PORT, path=PATH)
