# agent.py
import asyncio
import datetime
import json
import os
import re
from typing import Optional, TypedDict, Dict, Any, List, Tuple

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_gigachat.chat_models import GigaChat
from langchain_core.tools.base import ToolException

print("RUNNING FILE:", __file__)
load_dotenv()

CONVO_LOG_FILE = os.getenv("WELLBEING_CONVO_LOG", "wellbeing_conversation.jsonl")

def _append_jsonl(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def _norm_url(u: str) -> str:
    u = (u or "").strip()
    if not u:
        return "http://127.0.0.1:8100/mcp"
    u = u.rstrip("/")
    return u

async def get_mcp_client() -> MultiServerMCPClient:
    base = _norm_url(os.getenv("WELLBEING_MCP_URL", "http://127.0.0.1:8100/mcp"))
    return MultiServerMCPClient({
        "wellbeing": {"transport": "streamable_http", "url": base + "/"}
    })

def build_llm() -> GigaChat:
    creds = os.getenv("GIGACHAT_CREDENTIALS", "").strip()
    if not creds:
        raise RuntimeError("Нет GIGACHAT_CREDENTIALS в .env")
    return GigaChat(
        credentials=creds,
        verify_ssl_certs=os.getenv("GIGACHAT_VERIFY_SSL", "false").lower() == "true",
        scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
    )

SYSTEM_TAGS_INTERP_REPLY = """
Ты — ассистент личного дневника.
Тебе дают текст записи и оценку настроения 1–5, а также (опционально) похожие прошлые записи.

Верни ТОЛЬКО JSON:
{
  "tags": "слово1, слово2",
  "interpretation": "1-2 предложения",
  "reply": "2-4 предложения пользователю",
  "question": "1 короткий релевантный вопрос"
}

Правила:
- tags: 2-5 русских слов через запятую
- interpretation: мягко объясни смысл
- reply: поддержка + 1 маленький шаг
- question: один уточняющий вопрос
Без диагнозов.
""".strip()

SYSTEM_SUMMARY = """
Анализируй JSON дня: записи, mood, теги.
Сводка в 2-4 предложениях:
- кол-во записей
- средний mood
- темы дня
- 1 наблюдение/лайфхак
- 1 идея, что можно обсудить с психологом
Без диагнозов.
""".strip()

SYSTEM_SEARCH_ANSWER = """
Ты — ассистент дневника с поиском по прошлым записям.

Тебе дают:
- запрос пользователя
- найденные записи (если пусто — значит пусто)

ЖЁСТКИЕ ПРАВИЛА:
- Если записей нет — так и скажи. НЕ добавляй смежные темы. НЕ придумывай даты/факты.
- Никогда не придумывай записи.
- Используй только найденные записи.
- Дату выводи СТРОГО как YYYY-MM-DD (из created_at). НЕ меняй формат.

Формат:
1) 1 предложение: что нашлось/не нашлось
2) 1–5 строк: YYYY-MM-DD — короткий фрагмент (из raw_text)
3) 1 короткий вопрос ТОЛЬКО для уточнения запроса (другое слово/форма/контекст/период)
""".strip()

SYSTEM_SMALLTALK = """
Ты — очень краткий ассистент дневника.
Отвечай ОДНИМ предложением, дружелюбно.
Если это приветствие/болтовня — предложи записать события и поставить настроение 1–5.
""".strip()

def _today_iso() -> str:
    return datetime.date.today().isoformat()

def _is_rating(text: str) -> bool:
    return text.strip() in {"1", "2", "3", "4", "5"}

def _is_exit(text: str) -> bool:
    return text.strip().lower() in {"выход", "exit", "quit"}

def _parse_date_command(text: str) -> Optional[str]:
    t = text.lower().strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", t):
        return t
    if t in {"сегодня", "сводка сегодня", "итог сегодня"}:
        return _today_iso()
    return None

def _is_summary(text: str) -> bool:
    t = text.lower().strip()
    return t.startswith(("сводка", "итог", "резюме")) or (_parse_date_command(text) is not None)

def _is_paths(text: str) -> bool:
    return text.strip().lower() in {"paths", "debug", "статус", "status"}

def _is_reindex(text: str) -> bool:
    return text.strip().lower() in {"reindex", "реиндекс", "переиндекс", "пересобери", "пересобрать"}

def _is_find_cmd(text: str) -> bool:
    t = text.lower().strip()
    return t.startswith(("найди", "поищи", "поиск")) or ("что я писал" in t) or ("что я писала" in t)

def _extract_find_query_and_mode(text: str) -> Tuple[str, str]:
    raw = text.strip()
    low = raw.lower()

    if ("что я писал" in low) or ("что я писала" in low):
        return raw, "rerank"

    mode = "word"
    if low.startswith(("найди!", "поищи!", "поиск!")):
        mode = "rerank"

    for prefix in ("найди!", "поищи!", "поиск!", "найди", "поищи", "поиск"):
        if low.startswith(prefix):
            q = raw[len(prefix):].strip(" :—-")
            return q, mode

    return raw, mode

GREETINGS = {"привет", "приветик", "приветики", "хай", "hello", "hi", "йо", "здарова", "добрый день", "доброе утро", "добрый вечер"}
FEELINGS_MARKERS = ("груст", "тревож", "страш", "бою", "волную", "пережива", "плохо", "одиноко", "злю", "обид", "устал", "выгор", "панику", "рад", "счастлив", "раздраж")
SMALLTALK_SHORT = {"как дела", "как ты", "че как", "что нового", "ок", "ладно", "понятно", "ясно"}

def _is_smalltalk_not_diary(text: str) -> bool:
    t = text.lower().strip()
    if any(m in t for m in FEELINGS_MARKERS):
        return False
    if t in GREETINGS:
        return True
    if len(t) <= 12 and (t in SMALLTALK_SHORT or t.endswith("дела") or t.endswith("ты")):
        return True
    return False

def _extract_json_object(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"(\{.*?\})", text, flags=re.DOTALL)
    return m.group(1) if m else None

def _safe_json_loads(js: str) -> Optional[dict]:
    if not js:
        return None
    try:
        return json.loads(js)
    except Exception:
        pass
    fixed = re.sub(r",\s*([}\]])", r"\1", js)
    if "'" in fixed and '"' not in fixed:
        fixed = fixed.replace("'", '"')
    try:
        return json.loads(fixed)
    except Exception:
        return None

def _format_hits_for_prompt(hits: List[Dict[str, Any]], max_items: int = 5) -> str:
    lines = []
    for h in hits[:max_items]:
        created_at = (h.get("created_at") or "")[:10]
        mood = h.get("mood_score")
        tags = (h.get("tags") or "").strip()
        text = (h.get("raw_text") or "").strip().replace("\n", " ")
        if len(text) > 220:
            text = text[:220] + "…"
        meta = []
        if created_at:
            meta.append(created_at)
        if mood is not None:
            meta.append(f"mood={mood}")
        if tags:
            meta.append(f"tags={tags}")
        meta_s = " | ".join(meta)
        lines.append(f"- [{meta_s}] {text}")
    return "\n".join(lines) if lines else ""

def _unwrap_tool_text(res: Any) -> Any:
    if isinstance(res, list) and res and isinstance(res[0], dict) and "text" in res[0]:
        txt = res[0]["text"]
        try:
            return json.loads(txt)
        except Exception:
            return txt
    return res

async def llm_tags_interp_reply(llm: GigaChat, raw_text: str, mood_score: int, similar_hits: Optional[List[Dict[str, Any]]] = None) -> Dict[str, str]:
    similar_block = ""
    if similar_hits:
        similar_block = "\n\nПохожие прошлые записи:\n" + _format_hits_for_prompt(similar_hits, max_items=3)

    user_prompt = f"""Текст записи: "{raw_text}"
Mood: {mood_score}{similar_block}

Верни JSON строго по схеме. Без markdown и без текста до/после JSON."""
    resp = await llm.ainvoke([
        {"role": "system", "content": SYSTEM_TAGS_INTERP_REPLY},
        {"role": "user", "content": user_prompt}
    ])

    content = (resp.content or "").strip()
    js = _extract_json_object(content)
    data = _safe_json_loads(js) if js else None

    if not data:
        return {"tags": "", "interpretation": "", "reply": "Я записала это. Хочешь добавить подробность или оставить как есть?", "question": ""}

    return {
        "tags": (data.get("tags") or "").strip(),
        "interpretation": (data.get("interpretation") or "").strip(),
        "reply": (data.get("reply") or "").strip(),
        "question": (data.get("question") or "").strip(),
    }

async def llm_daily_summary(llm: GigaChat, summary_json: dict) -> str:
    prompt = f"JSON дня:\n{json.dumps(summary_json, ensure_ascii=False, indent=2)}"
    resp = await llm.ainvoke([
        {"role": "system", "content": SYSTEM_SUMMARY},
        {"role": "user", "content": prompt}
    ])
    return (resp.content or "").strip()

async def llm_answer_from_search(llm: GigaChat, query: str, hits: List[Dict[str, Any]]) -> str:
    if not hits:
        return "🔎 Ничего не найдено по этому запросу. Попробуй другую форму слова (например: программировала / программир)."

    hits_block = _format_hits_for_prompt(hits, max_items=5)
    prompt = f"""Запрос пользователя: {query}

Найденные записи:
{hits_block}
"""
    resp = await llm.ainvoke([
        {"role": "system", "content": SYSTEM_SEARCH_ANSWER},
        {"role": "user", "content": prompt}
    ])
    return (resp.content or "").strip()

async def llm_smalltalk(llm: GigaChat, user: str) -> str:
    resp = await llm.ainvoke([
        {"role": "system", "content": SYSTEM_SMALLTALK},
        {"role": "user", "content": user}
    ])
    return (resp.content or "").strip()

class DiaryState(TypedDict, total=False):
    user_input: str
    pending_text: Optional[str]
    route: str
    date: Optional[str]
    out_text: str
    search_mode: str
    find_query: str

def _ctx(config) -> Dict[str, Any]:
    return config["configurable"]["ctx"]

async def node_route(state: DiaryState, config) -> DiaryState:
    user = (state.get("user_input") or "").strip()
    if not user:
        return {"route": "empty", "out_text": ""}

    pending = (state.get("pending_text") or "").strip()

    if _is_exit(user):
        return {"route": "exit", "out_text": "👋 До свидания!"}

    if _is_paths(user):
        return {"route": "paths"}

    if _is_reindex(user):
        return {"route": "reindex"}

    if _is_rating(user):
        if pending:
            return {"route": "save"}
        return {"route": "rating_without_text", "out_text": "Сначала напиши текст записи, потом поставь настроение 1–5 🙂"}

    if _is_summary(user):
        date = _parse_date_command(user) or _today_iso()
        return {"route": "summary", "date": date}

    if _is_find_cmd(user):
        q, mode = _extract_find_query_and_mode(user)
        return {"route": "find", "find_query": q, "search_mode": mode}

    if pending:
        return {"route": "need_rating", "out_text": "😊 Оцени настроение цифрой 1–5:"}

    if _is_smalltalk_not_diary(user):
        return {"route": "smalltalk"}

    return {"route": "new_text"}

async def node_new_text(state: DiaryState, config) -> DiaryState:
    user = (state.get("user_input") or "").strip()
    return {"pending_text": user, "out_text": "😊 Оцени настроение 1–5:"}

async def node_save(state: DiaryState, config) -> DiaryState:
    ctx = _ctx(config)
    llm = ctx["llm"]
    log_entry_tool = ctx.get("log_entry_tool")
    semantic_tool = ctx.get("semantic_tool")

    pending_text = (state.get("pending_text") or "").strip()
    mood = int((state.get("user_input") or "0").strip())

    if not log_entry_tool:
        return {"out_text": "❌ tool log_entry не найден на MCP сервере.", "pending_text": None}

    similar_hits: List[Dict[str, Any]] = []
    if semantic_tool and pending_text:
        try:
            similar_hits = await semantic_tool.ainvoke({"query": pending_text, "top_k": 15})
            similar_hits = _unwrap_tool_text(similar_hits)
            similar_hits = similar_hits[:3] if isinstance(similar_hits, list) else []
        except Exception:
            similar_hits = []

    gen = await llm_tags_interp_reply(llm, pending_text, mood, similar_hits=similar_hits)
    tags, interp, reply, question = gen["tags"], gen["interpretation"], gen["reply"], gen["question"]

    await log_entry_tool.ainvoke({
        "raw_text": pending_text,
        "mood_score": mood,
        "tags": tags,
        "interpretation": interp
    })

    out_parts = []
    if reply:
        out_parts.append(f"🤖 {reply}".strip())
    if question:
        out_parts.append(f"❓ {question}".strip())
    if tags:
        out_parts.append(f"🏷️ {tags}".strip())

    return {"out_text": "\n\n".join(out_parts).strip(), "pending_text": None}

async def node_summary(state: DiaryState, config) -> DiaryState:
    ctx = _ctx(config)
    llm = ctx["llm"]
    summary_tool = ctx.get("summary_tool")

    date = (state.get("date") or "").strip() or _today_iso()
    if not summary_tool:
        return {"out_text": "❌ tool get_daily_summary не найден на MCP сервере."}

    summary = await summary_tool.ainvoke({"date": date})
    summary = _unwrap_tool_text(summary)
    text = await llm_daily_summary(llm, summary)
    return {"out_text": f"📊 Сводка за {date}:\n\n{text}"}

async def node_find(state: DiaryState, config) -> DiaryState:
    ctx = _ctx(config)
    llm = ctx["llm"]

    search_word_tool = ctx.get("search_word_tool")
    rerank_tool = ctx.get("rerank_tool")
    semantic_tool = ctx.get("semantic_tool")

    q = (state.get("find_query") or "").strip()
    mode = (state.get("search_mode") or "word").strip()

    hits: Any = []
    try:
        if mode == "rerank":
            if rerank_tool:
                hits = await rerank_tool.ainvoke({"query": q, "top_k": 30, "top_n": 5})
            elif semantic_tool:
                hits = await semantic_tool.ainvoke({"query": q, "top_k": 20})
            else:
                hits = []
        else:
            if not search_word_tool:
                return {"out_text": "❌ tool search_word не найден на сервере."}
            hits = await search_word_tool.ainvoke({"query": q, "limit": 20})
    except ToolException as e:
        return {"out_text": f"❌ Ошибка поиска: {e}"}
    except Exception as e:
        return {"out_text": f"❌ Ошибка поиска: {repr(e)}"}

    hits = _unwrap_tool_text(hits)
    answer = await llm_answer_from_search(llm, q, hits if isinstance(hits, list) else [])
    return {"out_text": answer}

async def node_smalltalk(state: DiaryState, config) -> DiaryState:
    ctx = _ctx(config)
    llm = ctx["llm"]
    user = (state.get("user_input") or "").strip()
    text = await llm_smalltalk(llm, user)
    return {"out_text": f"🤖 {text}\n\nЕсли хочешь — напиши, что произошло/что чувствуешь, и оцени настроение 1–5."}

async def node_paths(state: DiaryState, config) -> DiaryState:
    ctx = _ctx(config)
    debug_tool = ctx.get("debug_tool")
    if not debug_tool:
        return {"out_text": "❌ debug_paths tool не найден."}
    res = await debug_tool.ainvoke({})
    res = _unwrap_tool_text(res)
    return {"out_text": "DEBUG PATHS:\n" + json.dumps(res, ensure_ascii=False, indent=2)}

async def node_reindex(state: DiaryState, config) -> DiaryState:
    ctx = _ctx(config)
    rebuild_tool = ctx.get("rebuild_tool")
    rebuild_fts_tool = ctx.get("rebuild_fts_tool")

    if not rebuild_tool:
        return {"out_text": "❌ rebuild_faiss_from_db tool не найден на сервере."}

    try:
        res1 = await rebuild_tool.ainvoke({"batch_size": 256})
        res1 = _unwrap_tool_text(res1)
    except ToolException as e:
        return {"out_text": f"❌ Reindex упал: {e}"}
    except Exception as e:
        return {"out_text": f"❌ Reindex упал: {repr(e)}"}

    res2 = None
    if rebuild_fts_tool:
        try:
            res2 = await rebuild_fts_tool.ainvoke({})
            res2 = _unwrap_tool_text(res2)
        except Exception:
            res2 = None

    out = "✅ Reindex done:\n" + json.dumps(res1, ensure_ascii=False, indent=2)
    if res2 is not None:
        out += "\n\n✅ FTS rebuild:\n" + json.dumps(res2, ensure_ascii=False, indent=2)
    return {"out_text": out}

def route_to_next(state: DiaryState) -> str:
    return state.get("route", "new_text")

def build_graph():
    g = StateGraph(DiaryState)
    g.add_node("route", node_route)
    g.add_node("new_text", node_new_text)
    g.add_node("save", node_save)
    g.add_node("summary", node_summary)
    g.add_node("find", node_find)
    g.add_node("smalltalk", node_smalltalk)
    g.add_node("paths", node_paths)
    g.add_node("reindex", node_reindex)

    g.set_entry_point("route")
    g.add_conditional_edges(
        "route",
        route_to_next,
        {
            "empty": END,
            "exit": END,
            "paths": "paths",
            "reindex": "reindex",
            "new_text": "new_text",
            "save": "save",
            "need_rating": END,
            "rating_without_text": END,
            "summary": "summary",
            "find": "find",
            "smalltalk": "smalltalk",
        }
    )
    g.add_edge("new_text", END)
    g.add_edge("save", END)
    g.add_edge("summary", END)
    g.add_edge("find", END)
    g.add_edge("smalltalk", END)
    g.add_edge("paths", END)
    g.add_edge("reindex", END)
    return g.compile()

async def main():
    print("🔗 Подключение к MCP серверу...")
    mcp_client = await get_mcp_client()
    llm = build_llm()

    tools = await mcp_client.get_tools()
    print("=== TOOLS ===")
    for t in tools:
        print(t.name, type(t), "ainvoke=", hasattr(t, "ainvoke"))
    print("=============")

    log_entry_tool = next((t for t in tools if t.name == "log_entry"), None)
    summary_tool = next((t for t in tools if t.name == "get_daily_summary"), None)
    search_word_tool = next((t for t in tools if t.name == "search_word"), None)
    semantic_tool = next((t for t in tools if t.name == "search_semantic_only"), None)
    rerank_tool = next((t for t in tools if t.name == "search_with_rerank"), None)
    debug_tool = next((t for t in tools if t.name == "debug_paths"), None)
    rebuild_tool = next((t for t in tools if t.name == "rebuild_faiss_from_db"), None)
    rebuild_fts_tool = next((t for t in tools if t.name == "rebuild_fts_from_db"), None)

    if not log_entry_tool:
        print("❌ log_entry tool не найден! Проверь сервер и путь /mcp")
        return

    ctx = {
        "llm": llm,
        "log_entry_tool": log_entry_tool,
        "summary_tool": summary_tool,
        "search_word_tool": search_word_tool,
        "semantic_tool": semantic_tool,
        "rerank_tool": rerank_tool,
        "debug_tool": debug_tool,
        "rebuild_tool": rebuild_tool,
        "rebuild_fts_tool": rebuild_fts_tool,
    }

    graph = build_graph()

    print("✅ Wellbeing-дневник (LangGraph) готов!")
    print("Команды:")
    print(" - Напиши текст записи → потом оцени 1–5")
    print(" - 'сводка' / 'итог' / дата YYYY-MM-DD / 'сегодня' → сводка дня")
    print(" - 'найди <слово/слова>' → поиск по словам (FTS5 + prefix + LIKE fallback)")
    print(" - 'найди! <запрос>' → смысловой поиск (FAISS + rerank)")
    print(" - 'paths' → показать пути + faiss_ntotal")
    print(" - 'reindex' → пересобрать FAISS и FTS")
    print(" - 'выход' → выйти")

    state: DiaryState = {"pending_text": None}

    while True:
        user = input("\nТы: ").strip()
        state["user_input"] = user

        _append_jsonl(CONVO_LOG_FILE, {"ts": datetime.datetime.now().isoformat(timespec="seconds"), "role": "user", "text": user})

        new_state = await graph.ainvoke(state, config={"configurable": {"ctx": ctx}})
        out = (new_state.get("out_text") or "").strip()
        if out:
            print("\n" + out)
            _append_jsonl(CONVO_LOG_FILE, {"ts": datetime.datetime.now().isoformat(timespec="seconds"), "role": "assistant", "text": out, "route": new_state.get("route")})

        state.update(new_state)

        if new_state.get("route") == "exit":
            break

if __name__ == "__main__":
    asyncio.run(main())
