# tg_bot.py
import os
import asyncio
from dotenv import load_dotenv

from telegram import Update
from telegram.ext import Application, MessageHandler, ContextTypes, filters

from agent_core import init_ctx, build_graph

load_dotenv()

INSTRUCTION_TEXT = (
    "📔 Инструкция по работе с дневником\n\n"
    "📝 Как добавить запись:\n"
    "1) Напиши текст (что произошло / что чувствуешь)\n"
    "2) Отправь цифру 1–5 — оценка настроения\n\n"
    "📊 Сводка:\n"
    "• сводка\n"
    "• сегодня\n"
    "• YYYY-MM-DD (например: 2026-02-04)\n\n"
    "🔍 Поиск:\n"
    "• найди слово\n"
    "• найди! запрос — смысловой поиск\n\n"
    "📈 Отчёт:\n"
    "• график\n"
    "• отчет\n\n"
    "🛠 Служебное:\n"
    "• status / paths — состояние сервера\n"
    "• reindex — пересобрать поиск\n\n"
    "ℹ️ Помощь:\n"
    "• инструкция\n"
    "• помощь\n"
    "• как пользоваться\n"
)

HELP_TRIGGERS = {
    "инструкция",
    "помощь",
    "как пользоваться",
    "как это работает",
    "что ты умеешь",
    "что умеешь",
}

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    if not text:
        return

    if text.lower() in HELP_TRIGGERS:
        await update.message.reply_text(INSTRUCTION_TEXT)
        return

    graph = context.application.bot_data["graph"]
    ctx = context.application.bot_data["ctx"]

    chat_id = int(update.effective_chat.id)

    state = {
        "chat_id": chat_id,
        "user_input": text,
    }

    new_state = await graph.ainvoke(state, config={"configurable": {"ctx": ctx}})

    out = (new_state.get("out_text") or "").strip()
    if out:
        await update.message.reply_text(out)

    plot_path = new_state.get("plot_path")
    if plot_path and os.path.exists(plot_path):
        try:
            with open(plot_path, "rb") as f:
                await update.message.reply_photo(photo=f)
        except Exception:
            pass

async def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Нет TELEGRAM_BOT_TOKEN в .env")

    ctx = await init_ctx()
    graph = build_graph()

    app = Application.builder().token(token).build()
    app.bot_data["ctx"] = ctx
    app.bot_data["graph"] = graph

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    await app.initialize()
    await app.start()
    await app.updater.start_polling()
    print("✅ Telegram bot started (text-only). Напиши в чат: инструкция")

    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())
