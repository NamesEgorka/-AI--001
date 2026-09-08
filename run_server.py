"""
Точка входа для запуска сервера с ПЕРСИСТЕНТНЫМ хранилищем сессий
(SQLite вместо MemorySaver). Обычный `uvicorn api.main:app --reload`
по-прежнему работает и даёт in-memory поведение (удобно для быстрой
разработки и для тестов — ничего не меняется, create_app() без
аргумента `graph` строит MemorySaver-граф, как и раньше).

Используйте этот скрипт, когда нужно, чтобы диалоги переживали
перезапуск процесса:

    python3 run_server.py

По умолчанию база — sessions.db в текущей директории. Задать другой
путь:

    SESSIONS_DB_PATH=/tmp/my_sessions.db python3 run_server.py

ВАЖНО: --reload (автоперезагрузка при правке файлов) тут недоступна —
uvicorn.Server.serve() не поддерживает reload в этом режиме программного
запуска. Для разработки с hot-reload используйте обычный
`uvicorn api.main:app --reload` (in-memory), для проверки персистентности
— именно этот скрипт.
"""

from __future__ import annotations

import asyncio
import os

import uvicorn
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from api.main import create_app
from orchestrator.core import Orchestrator
from orchestrator.graph import build_graph

DB_PATH = os.environ.get("SESSIONS_DB_PATH", "sessions.db")


async def main() -> None:
    orchestrator = Orchestrator()
    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        graph = build_graph(orchestrator, checkpointer=checkpointer)
        app = create_app(orchestrator=orchestrator, graph=graph)

        config = uvicorn.Config(app, host="0.0.0.0", port=8000)
        server = uvicorn.Server(config)
        print(f"Персистентное хранилище сессий: {DB_PATH}")
        await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
