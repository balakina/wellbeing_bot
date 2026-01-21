import asyncio
import datetime
import json
import re
import os
from typing import Optional, TypedDict, Dict, Any

from dotenv import load_dotenv

from langgraph.graph import StateGraph, END

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_gigachat.chat_models import GigaChat

load_dotenv()


# ============ MCP КЛИЕНТ ============
async def get_mcp_client():
    return MultiServerMCPClient({
        "wellbeing": {
            "transport": "streamable_http",
            "url": os.getenv("WELLBEING_MCP_URL", "http://127.0.0.1:8100/mcp/")
        }
    })


# ============ LLM ============
def build_llm():
    creds = os.getenv("GIGACHAT_CREDENTIALS", "").strip()
    if not creds:
        raise RuntimeError("Нет GIGACHAT_CREDENTIALS в .env")

    return GigaChat(
        credentials=creds,
        verify_ssl_certs=os.getenv("GIGACHAT_VERIFY_SSL", "false").lower() == "true",
        scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
    )


SYSTEM_FOR_TAGS_INTERP_REPLY = """
Ты — ассистент личного дневника.
Тебе дают текст записи и оценку настроения 1–5.
Верни ТОЛЬКО JSON:
{
  "tags": "слово1, слово2",
  "interpretation": "1-2 предложения",
  "reply": "2-4 предложения пользователю"
}

Правила:
- tags: 2-5 русских слов через запятую
- interpretation: мягкое описание смысла сказанного пользователем
- reply: поддержка + 1 маленький шаг
""".strip()


SYSTEM_FOR_SUMMARY = """
Анализируй JSON дня: записи, mood, теги.
Сводка в 2-4 предложениях:
- кол-во записей
- средний mood
- темы дня
- 1 наблюдение/лайфхак
- дай совет, что можно обсудить с псхихологом
Без диагнозов.
""".strip()


# ============ УТИЛИТЫ ============
def is_rating(text: str) -> bool:
    return text.strip() in {"1", "2", "3", "4", "5"}


def is_summary_request(text: str) -> bool:
    t = text.lower()
    return any(x in t for x in ["сводк", "итог", "резюме", "что было"])


def parse_date(text: str) -> Optional[str]:
    m = re.search(r"\d{4}-\d{2}-\d{2}", text)
    if m:
        return m.group(0)
    if "сегодня" in text.lower():
        return datetime.date.today().isoformat()
    return None


async def llm_tags_interp_reply(llm, raw_text: str, mood_score: int) -> tuple[str, str, str]:
    user_prompt = f"""Текст: "{raw_text}"
Mood: {mood_score}
Верни JSON."""

    resp = await llm.ainvoke([
        {"role": "system", "content": SYSTEM_FOR_TAGS_INTERP_REPLY},
        {"role": "user", "content": user_prompt}
    ])

    content = resp.content or ""
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return "", "", "Не удалось разобрать ответ."

    try:
        data = json.loads(match.group(0))
        return (
            data.get("tags", "") or "",
            data.get("interpretation", "") or "",
            data.get("reply", "") or ""
        )
    except Exception:
        return "", "", content.strip()


async def llm_daily_summary(llm, summary_json: dict) -> str:
    prompt = f"Сводка дня:\n{json.dumps(summary_json, ensure_ascii=False, indent=2)}"
    resp = await llm.ainvoke([
        {"role": "system", "content": SYSTEM_FOR_SUMMARY},
        {"role": "user", "content": prompt}
    ])
    return (resp.content or "").strip()


# ============ СОСТОЯНИЕ ============
class DiaryState(TypedDict, total=False):
    user_input: str
    pending_text: Optional[str]
    route: str
    date: Optional[str]
    out_text: str
    out_tags: str


# ============ НОДЫ (ctx берём из config) ============
def _ctx(config) -> Dict[str, Any]:
    # ctx кладем в config["configurable"]["ctx"]
    try:
        return config["configurable"]["ctx"]
    except Exception:
        raise RuntimeError("Не передан ctx в config: config={'configurable': {'ctx': ...}}")

async def node_route(state: DiaryState, config) -> DiaryState:
    user = (state.get("user_input") or "").strip()
    if not user:
        return {"route": "empty", "out_text": ""}

    if user.lower() in ("выход", "exit", "quit"):
        return {"route": "exit", "out_text": "👋 До свидания!"}

    date = parse_date(user)
    if is_summary_request(user) or date:
        return {"route": "summary", "date": date}

    pending = state.get("pending_text")
    if pending:
        if is_rating(user):
            return {"route": "save"}
        return {"route": "need_rating", "out_text": "😊 Оцени настроение цифрой 1–5:"}

    if is_rating(user):
        return {"route": "rating_without_text", "out_text": "Сначала напиши текст записи, потом поставь настроение 1–5 🙂"}

    return {"route": "new_text"}


async def node_summary(state: DiaryState, config) -> DiaryState:
    ctx = _ctx(config)
    llm = ctx["llm"]
    summary_tool = ctx.get("summary_tool")

    date = state.get("date") or datetime.date.today().isoformat()
    if not summary_tool:
        return {"out_text": "❌ tool get_daily_summary не найден на MCP сервере."}

    try:
        summary = await summary_tool.ainvoke({"date": date})
        text = await llm_daily_summary(llm, summary)
        return {"out_text": f"📊 Сводка за {date}:\n\n{text}"}
    except Exception as e:
        return {"out_text": f"❌ {e}"}


async def node_new_text(state: DiaryState, config) -> DiaryState:
    user = (state.get("user_input") or "").strip()
    return {"pending_text": user, "out_text": "😊 Оцени настроение 1–5:"}


async def node_save(state: DiaryState, config) -> DiaryState:
    ctx = _ctx(config)
    llm = ctx["llm"]
    log_entry_tool = ctx.get("log_entry_tool")

    pending_text = (state.get("pending_text") or "").strip()
    mood = int((state.get("user_input") or "0").strip())

    if not log_entry_tool:
        return {"out_text": "❌ tool log_entry не найден на MCP сервере.", "pending_text": None}

    try:
        tags, interp, reply = await llm_tags_interp_reply(llm, pending_text, mood)

        await log_entry_tool.ainvoke({
            "raw_text": pending_text,
            "mood_score": mood,
            "tags": tags,
            "interpretation": interp
        })

        out = f"🤖 {reply}".strip()
        if tags:
            out += f"\n🏷️ {tags}"

        return {"out_text": out, "out_tags": tags, "pending_text": None}

    except Exception as e:
        return {"out_text": f"❌ {e}", "pending_text": None}


async def node_smalltalk(state: DiaryState, config) -> DiaryState:
    ctx = _ctx(config)
    llm = ctx["llm"]

    user = (state.get("user_input") or "").strip()
    try:
        resp = await llm.ainvoke([{"role": "user", "content": user}])
        return {"out_text": f"🤖 {resp.content}".strip()}
    except Exception as e:
        return {"out_text": f"❌ LLM: {e}"}


def route_to_next(state: DiaryState) -> str:
    return state.get("route", "smalltalk")


def build_graph():
    g = StateGraph(DiaryState)

    g.add_node("route", node_route)
    g.add_node("summary", node_summary)
    g.add_node("new_text", node_new_text)
    g.add_node("save", node_save)
    g.add_node("smalltalk", node_smalltalk)

    g.set_entry_point("route")

    g.add_conditional_edges(
        "route",
        route_to_next,
        {
            "empty": END,
            "exit": END,
            "summary": "summary",
            "new_text": "new_text",
            "save": "save",
            "need_rating": END,
            "rating_without_text": END,
            "smalltalk": "smalltalk",
        }
    )

    g.add_edge("summary", END)
    g.add_edge("new_text", END)
    g.add_edge("save", END)
    g.add_edge("smalltalk", END)

    return g.compile()


async def main():
    print("🔗 Подключение к MCP серверу...")
    mcp_client = await get_mcp_client()
    llm = build_llm()

    tools = await mcp_client.get_tools()
    log_entry_tool = next((t for t in tools if t.name == "log_entry"), None)
    summary_tool = next((t for t in tools if t.name == "get_daily_summary"), None)

    if not log_entry_tool:
        print("❌ log_entry tool не найден! Проверь сервер и путь /mcp")
        return

    ctx = {"llm": llm, "log_entry_tool": log_entry_tool, "summary_tool": summary_tool}
    graph = build_graph()

    print("✅ Wellbeing-дневник (LangGraph) готов!")
    print("📝 Текст → 1-5 | 'сводка' | 'выход'")

    state: DiaryState = {"pending_text": None}

    while True:
        user = input("\nТы: ").strip()
        state["user_input"] = user

        new_state = await graph.ainvoke(state, config={"configurable": {"ctx": ctx}})
        out = (new_state.get("out_text") or "").strip()
        if out:
            print("\n" + out)

        state.update(new_state)

        if new_state.get("route") == "exit":
            break


if __name__ == "__main__":
    asyncio.run(main())
