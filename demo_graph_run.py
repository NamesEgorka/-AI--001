"""
Демонстрация: запускаем граф, видим, как он останавливается на
interrupt() перед созданием заказа, затем "отвечаем" за пользователя
и возобновляем выполнение.

Запуск:
    PYTHONPATH=. python3 demo_graph_run.py
"""

from __future__ import annotations

import asyncio

# MemorySaver сериализует state между шагами через msgpack, чтобы работать
# одинаково что с in-memory checkpointer'ом, что с Postgres/Redis в проде.
# Наши DialogueState/SearchResultSnapshot — обычные @dataclass, не входящие
# в стандартный набор типов LangGraph "из коробки", поэтому их нужно явно
# зарегистрировать как разрешённые для (де)сериализации. Без этого шага
# всё РАБОТАЕТ (просто ругается warning'ом) — но в будущих версиях
# LangGraph это станет жёсткой ошибкой, поэтому регистрируем сразу.
from langgraph.types import Command
from orchestrator.core import Orchestrator
from orchestrator.graph import build_graph
from orchestrator.state import DialogueState, SearchResultSnapshot

# Переиспользуем те же фейковые клиенты, что и в golden dialogue тестах —
# граф работает поверх того же Orchestrator, поэтому подмена клиентов
# работает точно так же, без единого реального сетевого вызова.
from tests.test_golden_dialogues import FakeInternalApiClient, FakeKiwiClient


async def main() -> None:
    orchestrator = Orchestrator(
        internal_api=FakeInternalApiClient(policy_compliant=True, approval_required=False),
        kiwi_client=FakeKiwiClient(),
    )
    app = build_graph(orchestrator)

    thread_config = {"configurable": {"thread_id": "demo_session_1"}}

    initial_input = {
        "dialogue_state": DialogueState(session_id="demo_session_1"),
        "origin": "LAX",
        "destination": "JFK",
        "date_from": "2026-08-07",
        "passengers": 1,
        "selected_option_id": "opt_1",
        "user_confirmed": None,
        "final_result": None,
        "error": None,
    }

    print("--- Первый запуск: граф идёт до interrupt() и останавливается ---")
    result = await app.ainvoke(initial_input, config=thread_config)

    if "__interrupt__" in result:
        interrupt_payload = result["__interrupt__"][0].value
        print("Граф на паузе, ждёт подтверждения пользователя:")
        print(f"  Вопрос агента: {interrupt_payload['question']}")
        print(f"  Вариант: {interrupt_payload['selected_option']}")
        print(f"  Соответствие политике: {interrupt_payload['policy_compliant']}")
    else:
        print("Граф завершился без остановки (ошибка или пустой результат):", result.get("error"))
        return

    print("\n--- Пользователь отвечает 'да' ---")
    final_state = await app.ainvoke(Command(resume=True), config=thread_config)

    print("Итоговое состояние графа:")
    print(f"  current_state: {final_state['dialogue_state'].current_state}")
    print(f"  final_result: {final_state.get('final_result')}")
    print(f"  error: {final_state.get('error')}")


if __name__ == "__main__":
    asyncio.run(main())
