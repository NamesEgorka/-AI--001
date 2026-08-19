"""
Юнит-тесты роутера (orchestrator/router.py) — изолированно от LangGraph
и от Orchestrator'а: только маппинг intent+slots -> (entry_node, params)
и ошибки на неподдержанных/неполных intent'ах.
"""

from __future__ import annotations

import pytest

from nlu_output import ExtractedEntity
from orchestrator.router import (
    MissingRequiredSlotsError,
    UnsupportedIntentError,
    route,
    route_from_graph_state,
)


def _entity(slot_name: str, value: str) -> ExtractedEntity:
    return ExtractedEntity(
        slot_name=slot_name, value=value, raw_span=value,
        confidence=1.0, source="current_utterance",
    )


def test_route_search_flight_maps_slots_to_typed_fields():
    decision = route(
        "SearchFlight",
        [_entity("origin", "LAX"), _entity("destination", "JFK"), _entity("date_from", "2026-08-07")],
    )
    assert decision.entry_node == "search_flights"
    assert decision.graph_params == {
        "origin": "LAX", "destination": "JFK", "date_from": "2026-08-07", "passengers": 1,
    }


def test_route_search_hotel_uses_hotel_city_not_destination():
    """
    Ключевой тест на путаницу доменов: SearchHotel и SearchFlight оба
    используют слот с именем "destination", но должны попасть в РАЗНЫЕ
    поля GraphState (hotel_city vs destination) — см. docstring router.py.
    """
    decision = route(
        "SearchHotel",
        [_entity("destination", "Amsterdam"), _entity("check_in", "2026-03-20"), _entity("check_out", "2026-03-23")],
    )
    assert decision.entry_node == "search_hotels"
    assert decision.graph_params["hotel_city"] == "Amsterdam"
    assert "destination" not in decision.graph_params


def test_route_search_train_maps_to_train_prefixed_fields():
    decision = route(
        "SearchTrain",
        [_entity("origin", "Berlin"), _entity("destination", "Munich"), _entity("date_from", "2026-03-20")],
    )
    assert decision.entry_node == "search_trains"
    assert decision.graph_params["train_origin"] == "Berlin"
    assert decision.graph_params["train_destination"] == "Munich"
    assert decision.graph_params["train_passengers"] == 1


def test_route_select_option():
    decision = route("SelectOption", [_entity("option_id", "opt_1")])
    assert decision.entry_node == "select_option"
    assert decision.graph_params == {"selected_option_id": "opt_1"}


def test_route_check_order_status():
    decision = route("CheckOrderStatus", [_entity("order_id", "ord_42")])
    assert decision.entry_node == "check_order_status"
    assert decision.graph_params == {"order_id_to_check": "ord_42"}


def test_route_cancel_order_passes_through_user_confirmed_if_present():
    decision = route(
        "CancelOrder", [_entity("order_id", "ord_42"), _entity("user_confirmed", "true")],
    )
    assert decision.entry_node == "cancel_order"
    assert decision.graph_params["order_id_to_cancel"] == "ord_42"
    assert decision.graph_params["user_confirmed"] == "true"


def test_route_cancel_order_without_user_confirmed_slot_still_routes():
    """
    user_confirmed НЕ входит в required_slots намеренно (см. router.py) —
    guardrail внутри Orchestrator.cancel_order сам отклонит вызов без
    подтверждения, роутер не должен дублировать эту проверку.
    """
    decision = route("CancelOrder", [_entity("order_id", "ord_42")])
    assert decision.entry_node == "cancel_order"
    assert "user_confirmed" not in decision.graph_params


def test_route_missing_required_slot_raises():
    with pytest.raises(MissingRequiredSlotsError) as exc_info:
        route("SearchFlight", [_entity("origin", "LAX")])
    assert exc_info.value.missing == ["destination", "date_from"]


def test_route_unsupported_intent_raises():
    for intent in ("SmallTalk", "OutOfScope", "ExplainPolicy", "RequestApproval", "CreateOrder"):
        with pytest.raises(UnsupportedIntentError):
            route(intent, [])


def test_route_from_graph_state_returns_entry_node_when_valid():
    assert route_from_graph_state({"intent_entry_node": "search_flights"}) == "search_flights"


def test_route_from_graph_state_falls_back_to_unsupported_when_missing_or_unknown():
    assert route_from_graph_state({}) == "unsupported_intent"
    assert route_from_graph_state({"intent_entry_node": "not_a_real_node"}) == "unsupported_intent"
