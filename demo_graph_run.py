"""
Демонстрация: запускаем граф, видим, как он останавливается на
interrupt() перед созданием заказа, затем "отвечаем" за пользователя
и возобновляем выполнение.

Запуск:
    PYTHONPATH=. python3 demo_graph_run.py
"""

from __future__ import annotations

import asyncio

# Примечание: DialogueState — обычный @dataclass, целиком кладётся в один
# канал GraphState.dialogue_state (см. orchestrator/graph.py). MemorySaver
# сериализует его через встроенный jsonplus-сериализатор без дополнительной
# регистрации — специального импорта приватных функций library не требуется.
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

    search_input = {
        "dialogue_state": DialogueState(session_id="demo_session_1"),
        # С шага 5 (router-узел на входе графа) точка входа больше не
        # жёстко зашита в build_graph() — её явно указывает вызывающий
        # код (в проде — api/main.py на основе orchestrator/router.py).
        "intent_entry_node": "search_flights",
        "origin": "LAX",
        "destination": "JFK",
        "date_from": "2026-08-07",
        "passengers": 1,
        "user_confirmed": None,
        "final_result": None,
        "error": None,
    }

    print("--- Ход 1: SearchFlight — граф показывает варианты и завершает ход ---")
    # ВАЖНО: после исправления route_after_search (см. graph.py) поиск и
    # выбор варианта — ДВА РАЗНЫХ хода диалога, ровно как по-настоящему
    # происходит по HTTP (см. api/main.py, tests/test_api.py). Раньше
    # здесь option_id передавался сразу вместе с поиском одним вызовом —
    # это маскировало то, что пользователь физически не может знать
    # option_id ДО того, как увидит результаты.
    search_result = await app.ainvoke(search_input, config=thread_config)
    if search_result.get("error"):
        print("Ошибка поиска:", search_result["error"])
        return
    options = search_result["dialogue_state"].last_search_result.options
    print(f"  Найдено вариантов: {len(options)}, первый: {options[0]}")

    print("\n--- Ход 2: SelectOption — граф идёт до interrupt() и останавливается ---")
    select_input = {
        "intent_entry_node": "select_option",
        "selected_option_id": options[0]["option_id"],
        "error": None,
        "final_result": None,
    }
    result = await app.ainvoke(select_input, config=thread_config)

    if "__interrupt__" in result:
        interrupt_payload = result["__interrupt__"][0].value
        print("Граф на паузе, ждёт подтверждения пользователя:")
        print(f"  Вопрос агента: {interrupt_payload['question']}")
        print(f"  Вариант: {interrupt_payload['selected_option']}")
        print(f"  Соответствие политике: {interrupt_payload['policy_compliant']}")
    else:
        print("Граф завершился без остановки (ошибка или пустой результат):", result.get("error"))
        return

    print("\n--- Ход 3: пользователь отвечает 'да' ---")
    final_state = await app.ainvoke(Command(resume=True), config=thread_config)

    print("Итоговое состояние графа:")
    print(f"  current_state: {final_state['dialogue_state'].current_state}")
    print(f"  final_result: {final_state.get('final_result')}")
    print(f"  error: {final_state.get('error')}")


if __name__ == "__main__":
    asyncio.run(main())
