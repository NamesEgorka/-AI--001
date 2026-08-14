"""
Первая версия настоящего LangGraph StateGraph поверх уже готового
Orchestrator'а (orchestrator/core.py). Ничего из бизнес-логики не
переписываем — граф только формирует ПОРЯДОК вызова уже существующих
и протестированных методов.

Путь, который мы реализуем: SearchFlight -> SelectOption -> CheckPolicy
-> (пауза на подтверждение пользователя) -> CreateOrder.

Ключевая идея для тех, кто первый раз видит LangGraph: граф — это просто
конечный автомат (мы его уже спроектировали в transitions.py!), где
каждая "нода" — обычная Python-функция, а "рёбра" говорят, какая нода
выполняется следующей.
"""

from __future__ import annotations

from typing import Any, Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from .core import Orchestrator
from .state import DialogueState

# --- 1. State графа -----------------------------------------------------
#
# LangGraph хочет знать, какие "каналы" данных существуют между нодами.
# Мы не расписываем каждое поле DialogueState отдельным каналом — это
# было бы избыточно, — а просто кладём весь наш уже готовый DialogueState
# ОДНИМ полем. Это нормальный подход, когда у вас уже есть богатый
# объект состояния и не хочется его "распрямлять" под LangGraph.


class GraphState(TypedDict):
    dialogue_state: DialogueState
    origin: str
    destination: str
    date_from: str
    passengers: int
    selected_option_id: Optional[str]
    order_id_to_check: Optional[str]
    order_id_to_cancel: Optional[str]
    user_confirmed: Optional[bool]
    final_result: Optional[dict[str, Any]]
    error: Optional[str]


# --- 2. Ноды -------------------------------------------------------------
#
# Каждая нода — async-функция, которая принимает GraphState и возвращает
# ЧАСТИЧНОЕ обновление (LangGraph сам смержит его в общий state).
# Обратите внимание: ноды НЕ содержат бизнес-логики — они просто вызывают
# методы Orchestrator'а, которые мы уже написали и покрыли тестами.


def make_nodes(orchestrator: Orchestrator):
    """
    Фабрика нод, замкнутая на конкретный Orchestrator (с его клиентами —
    настоящими или фейковыми для теста). Так граф можно тестировать
    на FakeKiwiClient/FakeInternalApiClient, не трогая реальную сеть.
    """

    async def node_search_flights(state: GraphState) -> dict[str, Any]:
        ds: DialogueState = state["dialogue_state"]
        try:
            ds = await orchestrator.search_flights(
                ds,
                origin=state["origin"],
                destination=state["destination"],
                date_from=state["date_from"],
                passengers=state.get("passengers", 1),
            )
        except Exception as exc:  # noqa: BLE001
            return {"error": f"Ошибка поиска рейсов: {exc}"}
        return {"dialogue_state": ds}

    async def node_select_option(state: GraphState) -> dict[str, Any]:
        ds: DialogueState = state["dialogue_state"]
        option_id = state.get("selected_option_id")
        if not option_id:
            return {"error": "option_id не указан — нужно уточнить у пользователя."}
        try:
            orchestrator.select_option(ds, option_id=option_id)
        except Exception as exc:  # noqa: BLE001 — guardrail-нарушения тоже сюда
            return {"error": f"Guardrail заблокировал выбор варианта: {exc}"}
        return {"dialogue_state": ds}

    async def node_check_policy(state: GraphState) -> dict[str, Any]:
        ds: DialogueState = state["dialogue_state"]
        try:
            await orchestrator.check_policy(ds, user_id="demo_user")
        except Exception as exc:  # noqa: BLE001
            return {"error": f"Ошибка проверки политики: {exc}"}
        return {"dialogue_state": ds}

    async def node_await_user_confirmation(state: GraphState) -> dict[str, Any]:
        """
        Здесь и происходит "магия" interrupt(): выполнение графа
        ОСТАНАВЛИВАЕТСЯ прямо тут, состояние сохраняется в checkpointer,
        и функция возобновится только когда вы явно передадите
        Command(resume=...) при следующем вызове графа.

        Это прямой аналог interrupt_config, который мы настраивали
        в Fleet для create_order/cancel_order — только теперь это
        не конфиг платформы, а наш собственный код.
        """
        ds: DialogueState = state["dialogue_state"]
        policy = ds.policy_verdict or {}
        user_answer = interrupt(
            {
                "question": "Подтвердите оформление заказа?",
                "policy_compliant": policy.get("compliant"),
                "selected_option": (ds.order_draft or {}).get("selected_option"),
            }
        )
        return {"user_confirmed": bool(user_answer)}

    async def node_create_order(state: GraphState) -> dict[str, Any]:
        ds: DialogueState = state["dialogue_state"]
        try:
            result = await orchestrator.create_order(
                ds, user_confirmed=state.get("user_confirmed", False)
            )
        except Exception as exc:  # noqa: BLE001
            return {"error": f"Не удалось создать заказ: {exc}"}
        return {"dialogue_state": ds, "final_result": result}

    async def node_check_order_status(state: GraphState) -> dict[str, Any]:
        ds: DialogueState = state["dialogue_state"]
        order_id = state.get("order_id_to_check")
        if not order_id:
            return {"error": "order_id не указан для проверки статуса."}
        try:
            status = await orchestrator.check_order_status(ds, order_id=order_id)
        except Exception as exc:  # noqa: BLE001
            return {"error": f"Не удалось получить статус заказа: {exc}"}
        return {"dialogue_state": ds, "final_result": status}

    async def node_cancel_order(state: GraphState) -> dict[str, Any]:
        ds: DialogueState = state["dialogue_state"]
        order_id = state.get("order_id_to_cancel")
        if not order_id:
            return {"error": "order_id не указан для отмены."}
        try:
            result = await orchestrator.cancel_order(
                ds, order_id=order_id, user_confirmed=state.get("user_confirmed", False)
            )
        except Exception as exc:  # noqa: BLE001 — guardrail-блок (нет подтверждения) тоже сюда
            return {"error": f"Не удалось отменить заказ: {exc}"}
        return {"dialogue_state": ds, "final_result": result}

    return {
        "search_flights": node_search_flights,
        "select_option": node_select_option,
        "check_policy": node_check_policy,
        "await_user_confirmation": node_await_user_confirmation,
        "create_order": node_create_order,
        "check_order_status": node_check_order_status,
        "cancel_order": node_cancel_order,
    }


# --- 3. Условные рёбра -----------------------------------------------------
#
# Это буквально код из transitions.py, только теперь он решает, В КАКУЮ
# НОДУ идти дальше, а не просто проверяет допустимость перехода.


def route_after_search(state: GraphState) -> str:
    if state.get("error"):
        return END
    ds: DialogueState = state["dialogue_state"]
    if not ds.last_search_result or not ds.last_search_result.options:
        return END  # в полной версии — нода "уточнить параметры", а не END
    return "select_option"


def route_after_select(state: GraphState) -> str:
    return END if state.get("error") else "check_policy"


def route_after_policy(state: GraphState) -> str:
    return END if state.get("error") else "await_user_confirmation"


def route_after_confirmation(state: GraphState) -> str:
    return "create_order" if state.get("user_confirmed") else END


# --- 4. Сборка графа ---------------------------------------------------------


def build_graph(orchestrator: Orchestrator):
    """
    ВАЖНО: этот граф описывает ОДИН линейный поток — SearchFlight -> ... ->
    CreateOrder. Ноды check_order_status и cancel_order уже реализованы
    (см. make_nodes) и покрыты тестами на уровне Orchestrator'а, но НЕ
    подключены рёбрами здесь: это независимые intent'ы, а не шаги внутри
    сценария бронирования. Чтобы агент реально мог выбирать между
    intent'ами (а не всегда идти search_flights -> ... -> create_order),
    нужен отдельный router-узел на входе, который решает, В КАКОЙ подграф
    идти, на основе результата NLU. Это следующий шаг — см. README.md.
    """
    nodes = make_nodes(orchestrator)

    graph = StateGraph(GraphState)
    for name, fn in nodes.items():
        graph.add_node(name, fn)

    graph.add_edge(START, "search_flights")
    graph.add_conditional_edges("search_flights", route_after_search)
    graph.add_conditional_edges("select_option", route_after_select)
    graph.add_conditional_edges("check_policy", route_after_policy)
    graph.add_conditional_edges("await_user_confirmation", route_after_confirmation)
    graph.add_edge("create_order", END)

    # MemorySaver — простейший checkpointer (состояние в памяти процесса).
    # Наши DialogueState/SearchResultSnapshot — обычные @dataclass, не
    # входящие в стандартный набор типов LangGraph "из коробки", поэтому
    # явно регистрируем их как разрешённые для (де)сериализации через
    # официальный публичный параметр (а не через приватные функции модуля,
    # имена которых меняются между версиями библиотеки).
    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("orchestrator.state", "DialogueState"),
            ("orchestrator.state", "SearchResultSnapshot"),
        ]
    )
    checkpointer = MemorySaver(serde=serde)
    # Он и даёт нам interrupt()/resume работать, а заодно — персистентную
    # память диалога между вызовами по thread_id (в проде — Postgres/Redis
    # checkpointer вместо MemorySaver, интерфейс тот же).
    return graph.compile(checkpointer=checkpointer)
