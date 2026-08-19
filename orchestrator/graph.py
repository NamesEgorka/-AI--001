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
from .router import route_from_graph_state
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
    # Куда войти в граф в этом ходу — считается router'ом (orchestrator/
    # router.py) ДО вызова graph.ainvoke() и передаётся сюда одним полем,
    # а не пересчитывается внутри графа (см. docstring route_from_graph_state).
    intent_entry_node: Optional[str]
    origin: str
    destination: str
    date_from: str
    passengers: int
    hotel_city: str
    check_in: str
    check_out: str
    guests: int
    train_origin: str
    train_destination: str
    train_date_from: str
    train_passengers: int
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
        # .get(...), а не state["origin"]: раньше эта нода вызывалась ТОЛЬКО
        # как единственная точка входа графа (см. HANDOFF.md), поэтому все
        # поля гарантированно приходили от demo_graph_run.py/тестов. Теперь,
        # когда роутер может привести сюда трафик с любым набором полей
        # (в норме — уже отфильтрованных router.route(), но defense in depth
        # не помешает), прямая индексация упала бы KeyError вместо понятной
        # ошибки пользователю.
        origin, destination, date_from = state.get("origin"), state.get("destination"), state.get("date_from")
        if not origin or not destination or not date_from:
            return {"error": "Не хватает origin/destination/date_from для поиска рейсов."}
        try:
            ds = await orchestrator.search_flights(
                ds, origin=origin, destination=destination, date_from=date_from,
                passengers=state.get("passengers", 1),
            )
        except Exception as exc:  # noqa: BLE001
            return {"error": f"Ошибка поиска рейсов: {exc}"}
        return {"dialogue_state": ds}

    async def node_search_hotels(state: GraphState) -> dict[str, Any]:
        """
        Зеркало node_search_flights — тот же паттерн (вызвать метод
        Orchestrator'а, обработать guardrail/сетевую ошибку), только
        источник данных другой. Обратите внимание: select_option,
        check_policy, await_user_confirmation и create_order ниже —
        ТЕ ЖЕ САМЫЕ ноды, что и для рейсов, без единого изменения. Это
        и есть выгода от единого SearchResultSnapshot формата: путь
        "нашли варианты -> выбрали -> проверили политику -> подтвердили
        -> создали заказ" не знает и не должен знать, рейс это или отель.
        """
        ds: DialogueState = state["dialogue_state"]
        hotel_city, check_in, check_out = state.get("hotel_city"), state.get("check_in"), state.get("check_out")
        if not hotel_city or not check_in or not check_out:
            return {"error": "Не хватает hotel_city/check_in/check_out для поиска отелей."}
        try:
            ds = await orchestrator.search_hotels(
                ds, destination=hotel_city, check_in=check_in, check_out=check_out,
                guests=state.get("guests", 1),
            )
        except Exception as exc:  # noqa: BLE001
            return {"error": f"Ошибка поиска отелей: {exc}"}
        return {"dialogue_state": ds}

    async def node_search_trains(state: GraphState) -> dict[str, Any]:
        """
        Третье зеркало node_search_flights/node_search_hotels — та же
        нода select_option/check_policy/create_order ниже подходит и сюда
        без изменений.
        """
        ds: DialogueState = state["dialogue_state"]
        train_origin = state.get("train_origin")
        train_destination = state.get("train_destination")
        train_date_from = state.get("train_date_from")
        if not train_origin or not train_destination or not train_date_from:
            return {"error": "Не хватает train_origin/train_destination/train_date_from для поиска жд-билетов."}
        try:
            ds = await orchestrator.search_trains(
                ds, origin=train_origin, destination=train_destination, date_from=train_date_from,
                passengers=state.get("train_passengers", 1),
            )
        except Exception as exc:  # noqa: BLE001
            return {"error": f"Ошибка поиска жд-билетов: {exc}"}
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

    async def node_unsupported_intent(state: GraphState) -> dict[str, Any]:
        """
        Точка приземления для intent'ов, у которых пока нет обработчика
        (ExplainPolicy/SmallTalk/OutOfScope — см. router.py) или которые
        router вообще не смог сопоставить ни одной ноде. НЕ должно
        случаться при нормальной работе api/main.py (там router.route()
        уже бросил бы RouterError раньше, до вызова графа) — это
        последний рубеж на случай, если кто-то вызовет граф напрямую,
        в обход роутера.
        """
        return {"error": f"Неподдержанный intent, нет обработчика в графе: {state.get('intent_entry_node')!r}."}

    return {
        "search_flights": node_search_flights,
        "search_hotels": node_search_hotels,
        "search_trains": node_search_trains,
        "select_option": node_select_option,
        "check_policy": node_check_policy,
        "await_user_confirmation": node_await_user_confirmation,
        "create_order": node_create_order,
        "check_order_status": node_check_order_status,
        "cancel_order": node_cancel_order,
        "unsupported_intent": node_unsupported_intent,
    }


# --- 3. Условные рёбра -----------------------------------------------------
#
# Это буквально код из transitions.py, только теперь он решает, В КАКУЮ
# НОДУ идти дальше, а не просто проверяет допустимость перехода.


def route_after_search(state: GraphState) -> str:
    """
    С шага 5 (router на входе графа) SearchX и SelectOption — ДВА РАЗНЫХ
    хода диалога (два отдельных HTTP-вызова /intent), а не одна цепочка
    внутри одного graph.ainvoke(): пользователь ещё не мог выбрать
    option_id в том же ходу, где эти варианты только что показаны.

    Поэтому после успешного поиска граф ВСЕГДА завершает ход здесь (END),
    вне зависимости от того, нашлись варианты или нет — отличие видно
    по dialogue_state.current_state ("results_shown" vs "collecting_params",
    см. Orchestrator.search_flights/search_hotels/search_trains). Следующий
    ход — отдельный intent SelectOption через router (entry_node
    "select_option", тот же session_id) — см. route().
    """
    return END


def route_after_select(state: GraphState) -> str:
    return END if state.get("error") else "check_policy"


def route_after_policy(state: GraphState) -> str:
    return END if state.get("error") else "await_user_confirmation"


def route_after_confirmation(state: GraphState) -> str:
    return "create_order" if state.get("user_confirmed") else END


# --- 4. Сборка графа ---------------------------------------------------------


def build_graph(orchestrator: Orchestrator):
    """
    Граф с router-узлом на входе (шаг 5 из HANDOFF.md): вместо одного
    жёсткого пути START -> search_flights, START ветвится по
    state["intent_entry_node"] (уже посчитанному router.route() ДО вызова
    графа — см. api/main.py) в одну из пяти точек входа:

        search_flights / search_hotels / search_trains
            -> select_option -> check_policy -> await_user_confirmation
            -> create_order (после resume)

        select_option   — ВТОРОЙ ход в уже существующем thread_id (после
                           того, как предыдущий SearchX-ход показал
                           варианты) — продолжает по тому же пути выше.

        check_order_status / cancel_order — самостоятельные короткие пути,
                           сразу в END, без цепочки policy/approval.

    unsupported_intent — честный тупик для intent'ов без обработчика
    (см. node_unsupported_intent) — не должен случаться в норме, т.к.
    router.route() отсекает такие intent'ы раньше, до вызова графа.
    """
    nodes = make_nodes(orchestrator)

    graph = StateGraph(GraphState)
    for name, fn in nodes.items():
        graph.add_node(name, fn)

    graph.add_conditional_edges(
        START,
        route_from_graph_state,
        {
            "search_flights": "search_flights",
            "search_hotels": "search_hotels",
            "search_trains": "search_trains",
            "select_option": "select_option",
            "check_order_status": "check_order_status",
            "cancel_order": "cancel_order",
            "unsupported_intent": "unsupported_intent",
        },
    )
    graph.add_conditional_edges("search_flights", route_after_search)
    graph.add_conditional_edges("search_hotels", route_after_search)
    graph.add_conditional_edges("search_trains", route_after_search)
    graph.add_conditional_edges("select_option", route_after_select)
    graph.add_conditional_edges("check_policy", route_after_policy)
    graph.add_conditional_edges("await_user_confirmation", route_after_confirmation)
    graph.add_edge("create_order", END)
    graph.add_edge("check_order_status", END)
    graph.add_edge("cancel_order", END)
    graph.add_edge("unsupported_intent", END)

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
