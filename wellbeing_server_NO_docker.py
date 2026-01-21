from fastmcp import FastMCP
from typing import List, Dict, Any
import aiosqlite
import os
import datetime
import asyncio
import json
from starlette.requests import Request
from starlette.responses import PlainTextResponse

print("DEBUG WELLBEING_SERVER STARTED")
DB_PATH = "wellbeing.db"
print("DB_PATH:", os.path.abspath(DB_PATH))


async def get_conn():
    conn = await aiosqlite.connect(DB_PATH)
    conn.row_factory = lambda c, r: dict(zip([col[0] for col in c.description], r))
    return conn


def now_iso() -> str:
    return datetime.datetime.now().isoformat()


async def init_db():
    """Инициализация БД"""
    conn = await get_conn()  # ← ПРОСТО!
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


mcp = FastMCP(
    name="WellbeingDiaryServer",
    instructions="MCP-сервер дневника: текст + mood 1-5 + теги + интерпретация."
)


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> PlainTextResponse:
    return PlainTextResponse("OK")


@mcp.tool
async def log_entry(raw_text: str, mood_score: int, tags: str = "", interpretation: str = "") -> Dict[str, Any]:
    if not 1 <= mood_score <= 5:
        raise ValueError("mood_score 1-5")

    ts = now_iso()
    conn = await get_conn()
    try:
        await conn.execute(
            "INSERT INTO entries (created_at, raw_text, mood_score, tags, interpretation) VALUES (?, ?, ?, ?, ?)",
            (ts, raw_text, mood_score, tags, interpretation)
        )
        await conn.commit()
        cursor = await conn.execute("SELECT last_insert_rowid() as id")
        entry_id = (await cursor.fetchone())["id"]
    finally:
        await conn.close()

    return {
        "status": "ok",
        "entry": {"id": entry_id, "created_at": ts, "raw_text": raw_text,
                  "mood_score": mood_score, "tags": tags, "interpretation": interpretation}
    }


@mcp.tool
async def get_last_entries(limit: int = 10) -> List[Dict[str, Any]]:
    conn = await get_conn()
    try:
        cursor = await conn.execute(
            "SELECT * FROM entries ORDER BY id DESC LIMIT ?",
            (limit,)
        )
        return await cursor.fetchall()
    finally:
        await conn.close()


@mcp.tool
async def get_entries_by_day(date: str) -> List[Dict[str, Any]]:
    conn = await get_conn()
    try:
        cursor = await conn.execute(
            "SELECT * FROM entries WHERE substr(created_at, 1, 10) = ? ORDER BY created_at",
            (date,)
        )
        return await cursor.fetchall()
    finally:
        await conn.close()


@mcp.tool
async def get_daily_summary(date: str) -> Dict[str, Any]:
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
            "date": date, "entries": entries,
            "total_entries": stats["c"],
            "avg_mood": float(stats["avg_mood"]) if stats["avg_mood"] else None
        }
    finally:
        await conn.close()


@mcp.tool
async def get_mood_stats(days: int = 30) -> Dict[str, Any]:
    cutoff = (datetime.datetime.now() - datetime.timedelta(days=days)).isoformat()
    conn = await get_conn()
    try:
        cursor = await conn.execute(
            "SELECT COUNT(*) as c, AVG(mood_score) as avg_mood FROM entries WHERE created_at >= ?",
            (cutoff,)
        )
        row = await cursor.fetchone()
        return {
            "days": days,
            "total_entries": row["c"],
            "avg_mood": float(row["avg_mood"]) if row["avg_mood"] else None
        }
    finally:
        await conn.close()


@mcp.resource("wellbeing://status")
async def wellbeing_status() -> str:
    conn = await get_conn()
    try:
        cursor = await conn.execute("SELECT COUNT(*) as c FROM entries")
        count = (await cursor.fetchone())["c"]
        return json.dumps({
            "status": "online", "entries_count": count, "timestamp": now_iso()
        }, ensure_ascii=False)
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(init_db())
    print("🚀 http://localhost:8100/mcp")
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8100, path="/mcp")
