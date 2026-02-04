# server.py
# ================== OFFLINE / ENV FIRST ==================
import os
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

WELLBEING_OFFLINE = os.getenv("WELLBEING_OFFLINE", "true").lower() == "true"
if WELLBEING_OFFLINE:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

WELLBEING_PLOT = os.getenv("WELLBEING_PLOT", "matplotlib").lower()

# ================== NORMAL IMPORTS ==================
from fastmcp import FastMCP
from typing import List, Dict, Any, Optional

import aiosqlite
import datetime
import asyncio
import json
import shutil
import re
import tempfile

from starlette.requests import Request
from starlette.responses import PlainTextResponse

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

HAVE_MPL = False
if WELLBEING_PLOT == "matplotlib":
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        HAVE_MPL = True
    except Exception:
        HAVE_MPL = False

# ================== PATHS / CONFIG ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.getenv("WELLBEING_DB_PATH", os.path.join(BASE_DIR, "wellbeing.db"))
FAISS_INDEX_PATH = os.getenv("WELLBEING_FAISS_INDEX", os.path.join(BASE_DIR, "faiss.index"))
FAISS_MAP_PATH = os.getenv("WELLBEING_FAISS_MAP", os.path.join(BASE_DIR, "faiss_map.json"))

EMBED_MODEL_NAME = os.getenv("WELLBEING_EMBED_MODEL", "intfloat/multilingual-e5-small")

RERANK_MODEL_NAME = os.getenv("WELLBEING_RERANK_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
ENABLE_RERANK = os.getenv("WELLBEING_ENABLE_RERANK", "true").lower() == "true"

HOST = os.getenv("WELLBEING_HOST", "0.0.0.0")
PORT = int(os.getenv("WELLBEING_PORT", "8100"))
PATH = os.getenv("WELLBEING_PATH", "/mcp")

# Усиление поиска (для FAISS): префиксы + 3-граммы
AUGMENT_FOR_SEARCH = os.getenv("WELLBEING_AUGMENT_TEXT", "true").lower() == "true"
AUG_PREFIX_MIN = int(os.getenv("WELLBEING_AUG_PREFIX_MIN", "3"))
AUG_PREFIX_MAX = int(os.getenv("WELLBEING_AUG_PREFIX_MAX", "5"))
AUG_CHAR_N = int(os.getenv("WELLBEING_AUG_CHAR_N", "3"))
AUG_CHAR_MAX = int(os.getenv("WELLBEING_AUG_CHAR_MAX", "120"))

# FAISS persist debounce
FAISS_PERSIST_EVERY = int(os.getenv("WELLBEING_FAISS_PERSIST_EVERY", "20"))  # каждые N добавлений

# Важно: автопересборка FAISS при старте (если индекс пустой, а записи есть)
AUTO_REINDEX_ON_START = os.getenv("WELLBEING_AUTO_REINDEX_ON_START", "false").lower() == "true"

print("=== WELLBEING SERVER START ===")
print("BASE_DIR:", BASE_DIR)
print("DB_PATH:", os.path.abspath(DB_PATH))
print("FAISS_INDEX_PATH:", os.path.abspath(FAISS_INDEX_PATH))
print("FAISS_MAP_PATH:", os.path.abspath(FAISS_MAP_PATH))
print("EMBED_MODEL_NAME:", EMBED_MODEL_NAME)
print("RERANK_MODEL_NAME:", RERANK_MODEL_NAME)
print("ENABLE_RERANK:", ENABLE_RERANK)
print("WELLBEING_OFFLINE:", WELLBEING_OFFLINE, "(HF_HUB_OFFLINE=", os.getenv("HF_HUB_OFFLINE"), ")")
print("PLOT:", WELLBEING_PLOT, "HAVE_MPL=", HAVE_MPL)
print("AUGMENT_FOR_SEARCH:", AUGMENT_FOR_SEARCH)
print("AUTO_REINDEX_ON_START:", AUTO_REINDEX_ON_START)

# ================== DB HELPERS ==================
async def get_conn():
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

def now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")

async def init_db():
    conn = await get_conn()
    try:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                raw_text TEXT NOT NULL,
                mood_score INTEGER NOT NULL,
                tags TEXT,
                interpretation TEXT
            )
        """)
        await conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts
            USING fts5(raw_text, tags, content='entries', content_rowid='id')
        """)
        await conn.execute("""
            CREATE TRIGGER IF NOT EXISTS entries_ai AFTER INSERT ON entries BEGIN
              INSERT INTO entries_fts(rowid, raw_text, tags) VALUES (new.id, new.raw_text, COALESCE(new.tags,''));
            END
        """)
        await conn.execute("""
            CREATE TRIGGER IF NOT EXISTS entries_ad AFTER DELETE ON entries BEGIN
              INSERT INTO entries_fts(entries_fts, rowid, raw_text, tags)
              VALUES('delete', old.id, old.raw_text, COALESCE(old.tags,''));
            END
        """)
        await conn.execute("""
            CREATE TRIGGER IF NOT EXISTS entries_au AFTER UPDATE ON entries BEGIN
              INSERT INTO entries_fts(entries_fts, rowid, raw_text, tags)
              VALUES('delete', old.id, old.raw_text, COALESCE(old.tags,''));
              INSERT INTO entries_fts(rowid, raw_text, tags)
              VALUES (new.id, new.raw_text, COALESCE(new.tags,''));
            END
        """)
        await conn.commit()
        print("✅ DB initialized (entries + FTS5)")
    finally:
        await conn.close()

# ================== TEXT AUGMENT (FOR FAISS ONLY) ==================
def _basic_tokens(text: str) -> List[str]:
    t = (text or "").lower()
    out, cur = [], []
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

    pref = []
    for w in toks:
        if len(w) < AUG_PREFIX_MIN:
            continue
        for L in range(AUG_PREFIX_MIN, min(AUG_PREFIX_MAX, len(w)) + 1):
            pref.append(w[:L])

    glued = "".join(toks)
    grams = []
    n = max(2, int(AUG_CHAR_N))
    if len(glued) >= n:
        for i in range(0, len(glued) - n + 1):
            grams.append(glued[i:i+n])
            if len(grams) >= AUG_CHAR_MAX:
                break

    extra = []
    if pref:
        extra.append("pref: " + " ".join(pref[:200]))
    if grams:
        extra.append("grams: " + " ".join(grams[:AUG_CHAR_MAX]))

    return base + "\n" + "\n".join(extra)

# ================== FAISS STORE ==================
class FaissStore:
    """
    IndexFlatIP + normalize_embeddings=True => cosine.
    E5: query:/passage: prefixes.
    """
    def __init__(self, index_path: str, map_path: str, embed_model_name: str):
        self.index_path = index_path
        self.map_path = map_path
        self.embed_model_name = embed_model_name

        self.embedder: Optional[SentenceTransformer] = None
        self.dim: int = 0

        self.index: Optional[faiss.Index] = None
        self.id_map: List[int] = []
        self._lock = asyncio.Lock()

        self._dirty = 0

    def _ensure_embedder(self) -> None:
        if self.embedder is not None:
            return
        try:
            self.embedder = SentenceTransformer(self.embed_model_name)
            self.dim = int(self.embedder.get_sentence_embedding_dimension())
        except Exception as e:
            self.embedder = None
            self.dim = 0
            print("❌ Embedder init failed:", repr(e))
            print("   Semantic search disabled until model is available.")

    async def _embed_async(self, texts: List[str], *, is_query: bool) -> np.ndarray:
        self._ensure_embedder()
        if self.embedder is None:
            raise RuntimeError("Embedder not available (offline without cache?)")

        prefix = "query: " if is_query else "passage: "
        prepared = [_augment_text((t or "").strip()) for t in texts]
        prepared = [prefix + t for t in prepared]

        def _encode():
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
            print("⚠️ FAISS disabled (no embedder).")
            self.index = None
            self.id_map = []
            return

        if os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
            except Exception as e:
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                broken_path = self.index_path + f".broken.{ts}"
                try:
                    shutil.move(self.index_path, broken_path)
                except Exception:
                    pass
                print("⚠️ Failed to read faiss.index, created new. Error:", repr(e))
                print("⚠️ Broken index moved to:", broken_path)
                self._create_new_index()
        else:
            self._create_new_index()

        # dim mismatch -> сброс
        if self.index is not None and getattr(self.index, "d", None) != self.dim:
            print("⚠️ FAISS dim mismatch. index.d=", getattr(self.index, "d", None), "model.dim=", self.dim)
            self._create_new_index()

        if os.path.exists(self.map_path):
            try:
                with open(self.map_path, "r", encoding="utf-8") as f:
                    self.id_map = json.load(f)
            except Exception as e:
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                broken_map = self.map_path + f".broken.{ts}"
                try:
                    shutil.move(self.map_path, broken_map)
                except Exception:
                    pass
                print("⚠️ Failed to read faiss_map.json, reset map. Error:", repr(e))
                print("⚠️ Broken map moved to:", broken_map)
                self.id_map = []
        else:
            self.id_map = []

        ntotal = int(self.index.ntotal) if self.index else 0
        if self.index is not None and ntotal != len(self.id_map):
            print("⚠️ FAISS index/map mismatch:", ntotal, len(self.id_map))
        print("✅ FAISS loaded. ntotal =", int(self.index.ntotal) if self.index else None)

    def _persist_sync(self) -> None:
        """
        Unicode-safe persist:
        - FAISS на Windows может не уметь писать в путь с кириллицей.
        - Поэтому пишем в temp (ASCII), затем shutil.move в нужный путь.
        """
        if self.index is None:
            return

        os.makedirs(os.path.dirname(os.path.abspath(self.index_path)) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(self.map_path)) or ".", exist_ok=True)

        tmp_dir = tempfile.gettempdir()
        tmp_index = os.path.join(tmp_dir, "wellbeing_faiss.index.tmp")

        # пишем туда, где точно нет кириллицы
        faiss.write_index(self.index, tmp_index)
        # переносим Python'ом (Unicode OK)
        shutil.move(tmp_index, self.index_path)

        with open(self.map_path, "w", encoding="utf-8") as f:
            json.dump(self.id_map, f, ensure_ascii=False)

    async def persist_async(self) -> None:
        if self.index is None:
            return
        await asyncio.to_thread(self._persist_sync)

    async def add_entry(self, entry_id: int, text: str) -> None:
        async with self._lock:
            if self.index is None:
                return
            vec = await self._embed_async([text], is_query=False)
            self.index.add(vec)
            self.id_map.append(int(entry_id))

            self._dirty += 1
            if self._dirty >= max(1, FAISS_PERSIST_EVERY):
                await self.persist_async()
                self._dirty = 0

    async def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        async with self._lock:
            if self.index is None or self.index.ntotal == 0:
                return []
            query = (query or "").strip()
            if not query:
                return []

            qv = await self._embed_async([query], is_query=True)
            k = min(max(int(top_k), 1), int(self.index.ntotal))
            scores, idxs = self.index.search(qv, k)

            scores = scores[0].tolist()
            idxs = idxs[0].tolist()

            out: List[Dict[str, Any]] = []
            for faiss_id, s in zip(idxs, scores):
                if faiss_id < 0 or faiss_id >= len(self.id_map):
                    continue
                out.append({"entry_id": int(self.id_map[faiss_id]), "score": float(s)})
            return out

# ================== RERANKER ==================
class Reranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model: Optional[CrossEncoder] = None

    def _ensure(self) -> None:
        if self.model is not None:
            return
        try:
            self.model = CrossEncoder(self.model_name)
        except Exception as e:
            self.model = None
            print("⚠️ Reranker init failed:", repr(e))
            print("   Rerank disabled.")

    async def rerank_async(self, query: str, docs: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
        if not docs:
            return []
        query = (query or "").strip()
        if not query:
            return []
        top_n = max(int(top_n), 1)

        self._ensure()
        if self.model is None:
            return docs[:top_n]

        pairs = [(query, d.get("raw_text", "")) for d in docs]

        def _predict():
            return self.model.predict(pairs)

        scores = await asyncio.to_thread(_predict)
        scores = [float(x) for x in scores]

        enriched = []
        for d, s in zip(docs, scores):
            dd = dict(d)
            dd["rerank_score"] = s
            enriched.append(dd)

        enriched.sort(key=lambda x: x["rerank_score"], reverse=True)
        return enriched[:top_n]

# ================== SERVER ==================
mcp = FastMCP(
    name="WellbeingDiaryServer",
    instructions="MCP server: SQLite storage + FTS5 word search + FAISS semantic search + optional reranker."
)

store = FaissStore(FAISS_INDEX_PATH, FAISS_MAP_PATH, EMBED_MODEL_NAME)
reranker: Optional[Reranker] = Reranker(RERANK_MODEL_NAME) if ENABLE_RERANK else None

@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> PlainTextResponse:
    return PlainTextResponse("OK")

# ================== INTERNAL HELPERS ==================
async def _db_get_entries_by_ids(ids: List[int]) -> List[Dict[str, Any]]:
    if not ids:
        return []
    ids = [int(x) for x in ids]

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
    return docs

async def _impl_search_with_rerank(query: str, top_k: int = 50, top_n: int = 8) -> List[Dict[str, Any]]:
    docs = await _impl_search_semantic_only(query=query, top_k=int(top_k))
    if not docs:
        return []
    if reranker is None:
        return docs[: max(1, int(top_n))]
    return await reranker.rerank_async(query, docs, top_n=int(top_n))

# ================== TOOLS ==================
@mcp.tool
async def debug_paths() -> Dict[str, Any]:
    return {
        "base_dir": BASE_DIR,
        "db_path": os.path.abspath(DB_PATH),
        "faiss_index_path": os.path.abspath(FAISS_INDEX_PATH),
        "faiss_map_path": os.path.abspath(FAISS_MAP_PATH),
        "faiss_ntotal": int(store.index.ntotal) if store.index else None,
        "map_len": len(store.id_map),
        "embed_dim": store.dim,
        "rerank_enabled": ENABLE_RERANK,
        "offline": WELLBEING_OFFLINE,
        "have_mpl": HAVE_MPL,
        "augment_text": AUGMENT_FOR_SEARCH,
        "persist_every": FAISS_PERSIST_EVERY,
        "auto_reindex_on_start": AUTO_REINDEX_ON_START,
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

    async with store._lock:
        store._ensure_embedder()
        if store.dim <= 0:
            return {"status": "skip", "reason": "embedder not available", "total_db_rows": len(rows)}

        store.index = faiss.IndexFlatIP(store.dim)
        store.id_map = []

        texts: List[str] = []
        ids: List[int] = []
        added = 0

        for r in rows:
            ids.append(int(r["id"]))
            texts.append((r["raw_text"] or "").strip())

            if len(texts) >= batch_size:
                vecs = await store._embed_async(texts, is_query=False)
                store.index.add(vecs)
                store.id_map.extend(ids)
                added += len(ids)
                texts, ids = [], []

        if texts:
            vecs = await store._embed_async(texts, is_query=False)
            store.index.add(vecs)
            store.id_map.extend(ids)
            added += len(ids)

        # persist (unicode-safe)
        await store.persist_async()
        store._dirty = 0

    return {
        "status": "ok",
        "total_db_rows": len(rows),
        "added_to_faiss": added,
        "faiss_ntotal": int(store.index.ntotal) if store.index else 0,
        "map_len": len(store.id_map)
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
            (ts, raw_text, int(mood_score), tags or "", interpretation or "")
        )
        await conn.commit()
        cursor = await conn.execute("SELECT last_insert_rowid() as id")
        entry_id = int((await cursor.fetchone())["id"])
    finally:
        await conn.close()

    try:
        await store.add_entry(entry_id, raw_text)
    except Exception as e:
        print("❌ FAISS add_entry FAILED:", repr(e))

    return {
        "status": "ok",
        "entry": {
            "id": entry_id,
            "created_at": ts,
            "raw_text": raw_text,
            "mood_score": int(mood_score),
            "tags": tags or "",
            "interpretation": interpretation or ""
        }
    }

@mcp.tool
async def get_daily_summary(date: str) -> Dict[str, Any]:
    date = (date or "").strip()
    if len(date) != 10:
        raise ValueError("date must be YYYY-MM-DD")

    conn = await get_conn()
    try:
        cursor = await conn.execute(
            "SELECT * FROM entries WHERE substr(created_at, 1, 10) = ? ORDER BY created_at",
            (date,)
        )
        entries = await cursor.fetchall()

        if not entries:
            return {"date": date, "entries": [], "total_entries": 0, "avg_mood": None}

        cursor = await conn.execute(
            "SELECT COUNT(*) as c, AVG(mood_score) as avg_mood FROM entries WHERE substr(created_at, 1, 10) = ?",
            (date,)
        )
        stats = await cursor.fetchone()

        return {
            "date": date,
            "entries": entries,
            "total_entries": int(stats["c"]),
            "avg_mood": float(stats["avg_mood"]) if stats["avg_mood"] is not None else None
        }
    finally:
        await conn.close()

# -------- FIXED WORD SEARCH (FTS + prefix + LIKE + ё/е) --------
@mcp.tool
async def search_word(query: str, limit: int = 20) -> List[Dict[str, Any]]:
    q = (query or "").strip()
    if not q:
        return []

    def variants(s: str) -> List[str]:
        out = [s]
        if "е" in s:
            out.append(s.replace("е", "ё"))
        if "ё" in s:
            out.append(s.replace("ё", "е"))
        seen = set()
        res = []
        for x in out:
            if x not in seen:
                seen.add(x)
                res.append(x)
        return res

    toks = _basic_tokens(q)
    stems = []
    for t in toks:
        if len(t) >= 4:
            stem = t[: min(len(t), 8)]
            stems.append(stem + "*")
    prefix_or = " OR ".join(stems[:10]) if stems else ""

    conn = await get_conn()
    try:
        sql = """
            SELECT e.*
            FROM entries_fts AS f
            JOIN entries AS e ON e.id = f.rowid
            WHERE entries_fts MATCH ?
            ORDER BY e.created_at DESC
            LIMIT ?
        """

        qvars: List[str] = []
        for qq in variants(q):
            qvars.append(qq)
            qvars.append('"' + qq.replace('"', '""') + '"')
        if prefix_or:
            qvars.append(prefix_or)

        for v in qvars:
            try:
                cur = await conn.execute(sql, (v, int(limit)))
                rows = await cur.fetchall()
                if rows:
                    return rows
            except Exception:
                continue

        like_sql = """
            SELECT *
            FROM entries
            WHERE raw_text LIKE ?
            ORDER BY created_at DESC
            LIMIT ?
        """
        for qq in variants(q):
            cur = await conn.execute(like_sql, (f"%{qq}%", int(limit)))
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

# -------- PNG REPORT TOOL --------
@mcp.tool
async def export_last_weeks_report(weeks: int = 4, out_dir: str = "reports") -> Dict[str, Any]:
    if not HAVE_MPL:
        return {"status": "skip", "reason": "matplotlib not available"}

    weeks = max(int(weeks), 1)
    today = datetime.date.today()
    date_from = today - datetime.timedelta(days=weeks * 7)

    conn = await get_conn()
    try:
        cur = await conn.execute(
            """
            SELECT created_at, mood_score
            FROM entries
            WHERE substr(created_at, 1, 10) >= ?
            ORDER BY created_at
            """,
            (date_from.isoformat(),)
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
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
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
            "plot_path": plot_path
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
        "plot_path": plot_path
    }

@mcp.resource("wellbeing://status")
async def wellbeing_status() -> str:
    conn = await get_conn()
    try:
        cursor = await conn.execute("SELECT COUNT(*) as c FROM entries")
        count = int((await cursor.fetchone())["c"])
        faiss_total = int(store.index.ntotal) if store.index else 0
        return json.dumps({
            "status": "online",
            "entries_count": count,
            "faiss_total": faiss_total,
            "timestamp": now_iso(),
            "rerank_enabled": ENABLE_RERANK,
            "offline": WELLBEING_OFFLINE,
            "have_mpl": HAVE_MPL,
            "augment_text": AUGMENT_FOR_SEARCH
        }, ensure_ascii=False)
    finally:
        await conn.close()

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
            print("⚙️ AUTO_REINDEX: DB has entries but FAISS is empty -> rebuilding...")
            try:
                await rebuild_faiss_from_db(batch_size=256)
            except Exception as e:
                print("❌ AUTO_REINDEX failed:", repr(e))

if __name__ == "__main__":
    asyncio.run(_startup())
    print(f"🚀 http://127.0.0.1:{PORT}{PATH}")
    mcp.run(transport="streamable-http", host=HOST, port=PORT, path=PATH)
