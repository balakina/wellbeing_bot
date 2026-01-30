from fastmcp import FastMCP
from typing import List, Dict, Any, Optional
import aiosqlite
import os
import datetime
import asyncio
import json
import shutil

from starlette.requests import Request
from starlette.responses import PlainTextResponse

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder


# ================== PATHS / CONFIG ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.getenv("WELLBEING_DB_PATH", os.path.join(BASE_DIR, "wellbeing.db"))
FAISS_INDEX_PATH = os.getenv("WELLBEING_FAISS_INDEX", os.path.join(BASE_DIR, "faiss.index"))
FAISS_MAP_PATH = os.getenv("WELLBEING_FAISS_MAP", os.path.join(BASE_DIR, "faiss_map.json"))

# E5: используем query:/passage:
EMBED_MODEL_NAME = os.getenv("WELLBEING_EMBED_MODEL", "intfloat/multilingual-e5-small")

RERANK_MODEL_NAME = os.getenv("WELLBEING_RERANK_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
ENABLE_RERANK = os.getenv("WELLBEING_ENABLE_RERANK", "true").lower() == "true"

HOST = os.getenv("WELLBEING_HOST", "0.0.0.0")
PORT = int(os.getenv("WELLBEING_PORT", "8100"))
PATH = os.getenv("WELLBEING_PATH", "/mcp")

print("=== WELLBEING SERVER START ===")
print("BASE_DIR:", BASE_DIR)
print("DB_PATH:", os.path.abspath(DB_PATH))
print("FAISS_INDEX_PATH:", os.path.abspath(FAISS_INDEX_PATH))
print("FAISS_MAP_PATH:", os.path.abspath(FAISS_MAP_PATH))
print("EMBED_MODEL_NAME:", EMBED_MODEL_NAME)
print("RERANK_MODEL_NAME:", RERANK_MODEL_NAME)
print("ENABLE_RERANK:", ENABLE_RERANK)


# ================== DB HELPERS ==================
async def get_conn():
    conn = await aiosqlite.connect(DB_PATH)
    conn.row_factory = lambda c, r: dict(zip([col[0] for col in c.description], r))
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
        await conn.commit()
        print("✅ DB initialized")
    finally:
        await conn.close()


# ================== FAISS STORE ==================
class FaissStore:
    """
    FAISS IndexFlatIP + normalize_embeddings=True => cosine.
    E5: query:/passage: prefixes.
    """
    def __init__(self, index_path: str, map_path: str, embed_model_name: str):
        self.index_path = index_path
        self.map_path = map_path

        self.embedder = SentenceTransformer(embed_model_name)
        self.dim = int(self.embedder.get_sentence_embedding_dimension())

        self.index: Optional[faiss.Index] = None
        self.id_map: List[int] = []
        self._lock = asyncio.Lock()

    def _embed(self, texts: List[str], *, is_query: bool) -> np.ndarray:
        prefix = "query: " if is_query else "passage: "
        texts = [prefix + (t or "").strip() for t in texts]
        vecs = self.embedder.encode(texts, normalize_embeddings=True)
        return np.asarray(vecs, dtype="float32")

    def _create_new_index(self) -> None:
        self.index = faiss.IndexFlatIP(self.dim)
        self.id_map = []

    def load_or_create(self) -> None:
        # 1) load/create faiss index
        if os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
            except Exception as e:
                # битый индекс — переименуем и создадим новый
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

        # 2) load map
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

        # 3) sanity
        ntotal = int(self.index.ntotal) if self.index else 0
        if ntotal != len(self.id_map):
            print("⚠️ FAISS index/map mismatch:", ntotal, len(self.id_map))
            # не падаем — просто сообщаем

        print("✅ FAISS loaded. ntotal =", int(self.index.ntotal) if self.index else None)

    def persist(self) -> None:
        try:
            assert self.index is not None
            faiss.write_index(self.index, self.index_path)
            with open(self.map_path, "w", encoding="utf-8") as f:
                json.dump(self.id_map, f, ensure_ascii=False)
        except Exception as e:
            print("❌ FAISS persist FAILED:", repr(e))
            raise

    async def add_entry(self, entry_id: int, text: str) -> None:
        async with self._lock:
            assert self.index is not None
            vec = self._embed([text], is_query=False)  # passage
            self.index.add(vec)
            self.id_map.append(int(entry_id))
            self.persist()

    async def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        async with self._lock:
            assert self.index is not None
            if self.index.ntotal == 0:
                return []
            query = (query or "").strip()
            if not query:
                return []

            qv = self._embed([query], is_query=True)
            k = min(max(int(top_k), 1), int(self.index.ntotal))
            scores, idxs = self.index.search(qv, k)

            scores = scores[0].tolist()
            idxs = idxs[0].tolist()

            out: List[Dict[str, Any]] = []
            for faiss_id, s in zip(idxs, scores):
                if faiss_id < 0:
                    continue
                if faiss_id >= len(self.id_map):
                    continue
                out.append({"entry_id": int(self.id_map[faiss_id]), "score": float(s)})
            return out


# ================== RERANKER ==================
class Reranker:
    def __init__(self, model_name: str):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, docs: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
        if not docs:
            return []
        query = (query or "").strip()
        if not query:
            return []

        top_n = max(int(top_n), 1)
        pairs = [(query, d.get("raw_text", "")) for d in docs]
        scores = self.model.predict(pairs)
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
    instructions="MCP server: SQLite storage + FAISS semantic search + optional reranker."
)

store = FaissStore(FAISS_INDEX_PATH, FAISS_MAP_PATH, EMBED_MODEL_NAME)
reranker: Optional[Reranker] = Reranker(RERANK_MODEL_NAME) if ENABLE_RERANK else None


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> PlainTextResponse:
    return PlainTextResponse("OK")


# ================== INTERNAL HELPERS (NOT tools) ==================
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
    }


@mcp.tool
async def faiss_stats() -> Dict[str, Any]:
    return {
        "faiss_ntotal": int(store.index.ntotal) if store.index else None,
        "map_len": len(store.id_map),
        "index_path_exists": os.path.exists(FAISS_INDEX_PATH),
        "map_path_exists": os.path.exists(FAISS_MAP_PATH),
    }


@mcp.tool
async def rebuild_faiss_from_db(batch_size: int = 256) -> Dict[str, Any]:
    """
    Полностью пересобирает FAISS индекс из SQLite (E5 passage:).
    """
    batch_size = max(int(batch_size), 16)

    conn = await get_conn()
    try:
        cursor = await conn.execute("SELECT id, raw_text FROM entries ORDER BY id")
        rows = await cursor.fetchall()
    finally:
        await conn.close()

    async with store._lock:
        store.index = faiss.IndexFlatIP(store.dim)
        store.id_map = []

        texts: List[str] = []
        ids: List[int] = []
        added = 0

        for r in rows:
            ids.append(int(r["id"]))
            texts.append((r["raw_text"] or "").strip())

            if len(texts) >= batch_size:
                vecs = store._embed(texts, is_query=False)
                store.index.add(vecs)
                store.id_map.extend(ids)
                added += len(ids)
                texts, ids = [], []

        if texts:
            vecs = store._embed(texts, is_query=False)
            store.index.add(vecs)
            store.id_map.extend(ids)
            added += len(ids)

        store.persist()

    return {"status": "ok", "total_db_rows": len(rows), "added_to_faiss": added,
            "faiss_ntotal": int(store.index.ntotal), "map_len": len(store.id_map)}


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

    # FAISS add (не даём silently fail)
    try:
        before = int(store.index.ntotal) if store.index else -1
        await store.add_entry(entry_id, raw_text)
        after = int(store.index.ntotal) if store.index else -1
        print(f"✅ FAISS add_entry ok: before={before} after={after} entry_id={entry_id}")
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
async def get_last_entries(limit: int = 10) -> List[Dict[str, Any]]:
    limit = min(max(int(limit), 1), 200)
    conn = await get_conn()
    try:
        cursor = await conn.execute("SELECT * FROM entries ORDER BY id DESC LIMIT ?", (limit,))
        return await cursor.fetchall()
    finally:
        await conn.close()


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


@mcp.tool
async def semantic_search(query: str, top_k: int = 20) -> List[Dict[str, Any]]:
    """FAISS candidates: [{entry_id, score}, ...]"""
    return await store.search(query, top_k=int(top_k))


@mcp.tool
async def get_entries_by_ids(ids: List[int]) -> List[Dict[str, Any]]:
    return await _db_get_entries_by_ids(ids)


@mcp.tool
async def search_semantic_only(query: str, top_k: int = 20) -> List[Dict[str, Any]]:
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


@mcp.tool
async def search_with_rerank(query: str, top_k: int = 30, top_n: int = 5) -> List[Dict[str, Any]]:
    if reranker is None:
        return await search_semantic_only(query=query, top_k=top_k)

    cands = await store.search(query, top_k=int(top_k))
    if not cands:
        return []

    ids = [int(c["entry_id"]) for c in cands]
    docs = await _db_get_entries_by_ids(ids)

    score_map = {int(c["entry_id"]): float(c["score"]) for c in cands}
    for d in docs:
        d["faiss_score"] = score_map.get(int(d["id"]), None)

    return reranker.rerank(query, docs, top_n=int(top_n))


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
            "rerank_enabled": ENABLE_RERANK
        }, ensure_ascii=False)
    finally:
        await conn.close()


async def _startup() -> None:
    await init_db()
    store.load_or_create()


if __name__ == "__main__":
    asyncio.run(_startup())
    print(f"🚀 http://localhost:{PORT}{PATH}")
    mcp.run(transport="streamable-http", host=HOST, port=PORT, path=PATH)
