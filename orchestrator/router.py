"""
Router — единственное место, которое решает "в какую точку графа войти
в этом ходу диалога", на основе intent'а, который предложил NLU-слой.

Это и есть тот самый пробел из HANDOFF.md: раньше build_graph() умел
только один жёсткий путь (SearchFlight -> ... -> CreateOrder). Роутер
не содержит бизнес-логики сам по себе — он только:

  1. проверяет, что предложенный NLU intent вообще имеет точку входа
     в графе (SmallTalk/OutOfScope/ExplainPolicy — пока нет ноды, это
     ЧЕСТНО отражено как NotImplemented, а не тихо проглочено);
  2. проверяет, что обязательные слоты для этого intent'а присутствуют
     (defense in depth: NLU-слой уже обязан был выставить
     clarification_needed=True при нехватке слотов — см. nlu_output.py,
     но роутер не доверяет этому вслепую, а перепроверяет сам, чтобы
     нода не упала с KeyError на реальном трафике);
  3. переносит "сырые" ExtractedEntity (слоты с именами вроде "origin",
     "check_in", "guests") в типизированные поля GraphState, с явной
     таблицей маппинга на каждый intent — потому что одно и то же имя
     слота может значить разное для разных intent'ов (например,
     "destination" для SearchFlight — это аэропорт назначения, а для
     SearchHotel — город отеля; хранить их в одном поле GraphState
     было бы источником путаницы между доменами).

ВАЖНО: роутер выбирает узел ВХОДА в граф для НОВОГО отрезка диалога.
Продолжение уже начатого потока (после results_shown -> policy_check ->
await_user_confirmation) идёт НЕ через роутер, а через Command(resume=...)
у уже существующего thread_id — см. api/main.py, эндпоинт /confirm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from nlu_output import ExtractedEntity, IntentName

# --- Intent -> нода входа в граф ------------------------------------------
#
# Только те intent'ы, для которых уже есть ЗАКОНЧЕННЫЙ нода-обработчик в
# orchestrator/graph.py. Намеренно НЕ включены сюда:
#   - SelectOption как ПЕРВЫЙ ход диалога — бессмысленно (нечего выбирать
#     без предшествующего поиска в этом же thread_id), но как ПРОДОЛЖЕНИЕ
#     диалога (thread_id уже содержит last_search_result) — включён, это
#     нормальный второй ход после SearchFlight/Hotel/Train.
#   - RequestApproval/CreateOrder — не самостоятельная точка входа: они
#     всегда идут через interrupt()/resume в рамках уже начатого потока,
#     а не как реакция на новый intent с нуля.
#   - ExplainPolicy/SmallTalk/OutOfScope — для них пока нет ноды-обработчика
#     в графе (см. HANDOFF.md, "Оставшийся план") — router честно возвращает
#     UnsupportedIntentError, а не молча падает или делает вид, что обработал.

ENTRY_NODE_BY_INTENT: dict[str, str] = {
    "SearchFlight": "search_flights",
    "SearchHotel": "search_hotels",
    "SearchTrain": "search_trains",
    "SelectOption": "select_option",
    "CheckOrderStatus": "check_order_status",
    "CancelOrder": "cancel_order",
}

# --- Intent -> (обязательные слоты, маппинг slot_name -> поле GraphState) --
#
# required — то, без чего нода гарантированно упадёт или не сможет
# осмысленно отработать (проверяется роутером ДО входа в граф).
# slot_map — slot_name (как его называет NLU) -> имя канала в GraphState.

@dataclass(frozen=True)
class IntentSpec:
    required_slots: tuple[str, ...]
    slot_map: dict[str, str]
    defaults: dict[str, Any] = field(default_factory=dict)


INTENT_SPECS: dict[str, IntentSpec] = {
    "SearchFlight": IntentSpec(
        required_slots=("origin", "destination", "date_from"),
        slot_map={
            "origin": "origin",
            "destination": "destination",
            "date_from": "date_from",
            "passengers": "passengers",
        },
        defaults={"passengers": 1},
    ),
    "SearchHotel": IntentSpec(
        required_slots=("destination", "check_in", "check_out"),
        slot_map={
            # Важно: у SearchHotel слот называется так же, как у SearchFlight
            # ("destination"), но в GraphState это РАЗНОЕ поле (hotel_city) —
            # см. docstring модуля.
            "destination": "hotel_city",
            "check_in": "check_in",
            "check_out": "check_out",
            "guests": "guests",
        },
        defaults={"guests": 1},
    ),
    "SearchTrain": IntentSpec(
        required_slots=("origin", "destination", "date_from"),
        slot_map={
            "origin": "train_origin",
            "destination": "train_destination",
            "date_from": "train_date_from",
            "passengers": "train_passengers",
        },
        defaults={"train_passengers": 1},
    ),
    "SelectOption": IntentSpec(
        required_slots=("option_id",),
        slot_map={"option_id": "selected_option_id"},
    ),
    "CheckOrderStatus": IntentSpec(
        required_slots=("order_id",),
        slot_map={"order_id": "order_id_to_check"},
    ),
    "CancelOrder": IntentSpec(
        # user_confirmed сюда намеренно НЕ входит: guardrail в
        # Orchestrator.cancel_order сам отклонит вызов без подтверждения
        # (см. GuardrailViolation) — роутер не должен дублировать эту
        # проверку, только передать то, что реально пришло.
        required_slots=("order_id",),
        slot_map={"order_id": "order_id_to_cancel", "user_confirmed": "user_confirmed"},
    ),
}


class RouterError(Exception):
    """Базовый класс ошибок роутера — отличается от GuardrailViolation:
    это не нарушение бизнес-правила, а невозможность вообще войти в граф."""


class UnsupportedIntentError(RouterError):
    def __init__(self, intent_name: str):
        self.intent_name = intent_name
        super().__init__(
            f"Intent {intent_name!r} пока не имеет обработчика в графе "
            f"(см. HANDOFF.md, раздел 'Оставшийся план'). "
            f"Поддержаны: {sorted(ENTRY_NODE_BY_INTENT)}"
        )


class MissingRequiredSlotsError(RouterError):
    def __init__(self, intent_name: str, missing: list[str]):
        self.intent_name = intent_name
        self.missing = missing
        super().__init__(
            f"Intent {intent_name!r} не может войти в граф — не хватает "
            f"обязательных слотов: {missing}. Нужно вернуть пользователю "
            f"уточняющий вопрос, а не вызывать граф."
        )


@dataclass
class RoutingDecision:
    entry_node: str
    graph_params: dict[str, Any]


def route(intent_name: str, entities: list[ExtractedEntity]) -> RoutingDecision:
    """
    Главная функция роутера. Бросает RouterError, если войти в граф нельзя —
    вызывающий код (api/main.py) обязан поймать это и превратить в
    уточняющий вопрос пользователю, а не в 500-ю ошибку.
    """
    entry_node = ENTRY_NODE_BY_INTENT.get(intent_name)
    if entry_node is None:
        raise UnsupportedIntentError(intent_name)

    spec = INTENT_SPECS[intent_name]
    raw_slots: dict[str, str] = {e.slot_name: e.value for e in entities}

    missing = [slot for slot in spec.required_slots if slot not in raw_slots]
    if missing:
        raise MissingRequiredSlotsError(intent_name, missing)

    graph_params: dict[str, Any] = dict(spec.defaults)
    for slot_name, channel_name in spec.slot_map.items():
        if slot_name in raw_slots:
            graph_params[channel_name] = raw_slots[slot_name]

    return RoutingDecision(entry_node=entry_node, graph_params=graph_params)


def route_from_graph_state(state: dict[str, Any]) -> str:
    """
    Функция для graph.add_conditional_edges(START, ...): читает уже
    посчитанный entry_node из state["intent_entry_node"] (его туда кладёт
    api/main.py ДО вызова graph.ainvoke — см. route() выше) и просто
    возвращает имя ноды. Сам выбор ноды НЕ пересчитывается внутри графа,
    чтобы вся логика роутинга (включая ошибки) была в одном месте —
    в route() — и её можно было протестировать без поднятия LangGraph.
    """
    entry_node: Optional[str] = state.get("intent_entry_node")
    if not entry_node or entry_node not in ENTRY_NODE_BY_INTENT.values():
        return "unsupported_intent"
    return entry_node
