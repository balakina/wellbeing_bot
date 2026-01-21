# Wellbeing Diary — MCP Server + LangGraph Agent

Учебный проект личного дневника с чётким разделением ответственности:
- **MCP-сервер (FastMCP)** — хранение данных в SQLite и предоставление инструментов (tools) по протоколу MCP.
- **CLI-агент (LangGraph)** — диалоговый интерфейс, генерация тегов/интерпретации/ответа через GigaChat, сохранение данных через MCP, формирование сводки дня.

Проект демонстрирует архитектуру *Agent ↔ MCP tools ↔ Storage* и может использоваться как шаблон для более сложных агентных систем.

---

## Возможности

### MCP-сервер
- Хранение записей дневника в SQLite
- MCP-инструменты:
  - `log_entry` — сохранение записи
  - `get_last_entries` — последние записи
  - `get_entries_by_day` — записи за выбранную дату
  - `get_daily_summary` — агрегированная сводка дня
  - `get_mood_stats` — статистика настроения за период
- Health-check endpoint (`/health`)

### CLI-агент
- Диалоговый ввод записи
- Проверка корректности оценки настроения (1–5)
- Генерация:
  - тегов
  - интерпретации записи
  - поддерживающего ответа пользователю
- Команды:
  - `сводка` / `итог` / `резюме` — сводка за сегодня
  - `сводка YYYY-MM-DD` — сводка за конкретную дату
  - `выход` / `exit` — завершение работы

---

## Архитектура

```
User (CLI)
   ↓
LangGraph Agent
   ↓
MCP Client (tools)
   ↓
FastMCP Server
   ↓
SQLite (wellbeing.db)
```

LLM (GigaChat) используется только на стороне агента.

---

## Структура проекта

```
.
├── server.py          # MCP сервер (FastMCP + SQLite)
├── agent.py           # CLI агент (LangGraph + GigaChat)
├── wellbeing.db       # SQLite база данных (создаётся автоматически)
├── .env               # конфигурация окружения (локально)
└── README.md
```

---

## Требования

- Python 3.10+
- Доступ к GigaChat
- Локальный запуск (CLI)

---

## Установка

### 1. Виртуальное окружение

**Windows**
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Установка зависимостей

```bash
pip install fastmcp aiosqlite starlette python-dotenv langgraph langchain-mcp-adapters langchain-gigachat
```

---

## Конфигурация

Создай файл `.env` в корне проекта:

```env
WELLBEING_MCP_URL=http://127.0.0.1:8100/mcp/
GIGACHAT_CREDENTIALS=PASTE_YOUR_CREDENTIALS_HERE
GIGACHAT_VERIFY_SSL=false
GIGACHAT_SCOPE=GIGACHAT_API_PERS
```

⚠️ Файл `.env` не должен попадать в репозиторий.

---

## Запуск

### 1. Запуск сервера

```bash
python server.py
```

Ожидаемый вывод:
- `DB initialized`
- `http://localhost:8100/mcp`

Проверка:
```
http://localhost:8100/health
```

### 2. Запуск агента

```bash
python agent.py
```

Пример диалога:
```
Ты: Сегодня было напряжённо, но продуктивно
Ты: 4
```

Сводка:
```
Ты: сводка
```

---

## Хранение данных

База данных: `wellbeing.db`

Таблица `entries`:

| Поле | Тип | Описание |
|-----|----|---------|
| id | INTEGER | первичный ключ |
| created_at | TEXT | ISO timestamp |
| raw_text | TEXT | текст записи |
| mood_score | INTEGER | оценка 1–5 |
| tags | TEXT | теги |
| interpretation | TEXT | интерпретация |

---

## Назначение проекта

Проект предназначен для:
- изучения MCP и FastMCP
- демонстрации LangGraph как диалогового роутера
- примера разделения логики агента и серверных инструментов
- прототипирования wellbeing / diary-агентов

---

## Возможные улучшения

- Поиск по записям
- Экспорт данных
- Web-интерфейс
- Семантический поиск (embeddings + vector DB)

