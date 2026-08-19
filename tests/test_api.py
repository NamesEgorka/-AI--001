"""
Интеграционные тесты api/main.py — гоняем реальные HTTP-запросы (через
httpx.ASGITransport, без поднятия сокета) поверх Orchestrator'а с
фейковыми клиентами. Цель — доказать, что весь путь "HTTP -> router ->
LangGraph -> Orchestrator -> guardrails" работает end-to-end, а не только
что отдельные слои работают по отдельности (это уже покрыто
test_router.py/test_golden_dialogues.py/test_guardrails.py).
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from api.main import create_app
from orchestrator.core import Orchestrator
from tests.test_golden_dialogues import (
    FakeHotelClient,
    FakeInternalApiClient,
    FakeKiwiClient,
    FakeTrainClientForTest,
)


def _make_client(**internal_api_kwargs) -> AsyncClient:
    orch = Orchestrator(
        internal_api=FakeInternalApiClient(**internal_api_kwargs),
        kiwi_client=FakeKiwiClient(),
        hotel_client=FakeHotelClient(),
        train_client=FakeTrainClientForTest(),
    )
    app = create_app(orchestrator=orch)
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.mark.asyncio
async def test_full_flight_flow_over_http():
    async with _make_client(policy_compliant=True, approval_required=False) as client:
        r1 = await client.post(
            "/sessions/http_flight_1/intent",
            json={"intent": "SearchFlight", "slots": {
                "origin": "LAX", "destination": "JFK", "date_from": "2026-08-07",
            }},
        )
        assert r1.status_code == 200, r1.text
        body1 = r1.json()
        assert body1["current_state"] == "results_shown"
        assert body1["awaiting_confirmation"] is False
        # Исправлено на шаге 5 (см. graph.py route_after_search): успешный
        # поиск — это ЧИСТЫЙ результат без error, ждём следующий ход
        # (SelectOption) отдельным вызовом /intent, а не то же graph.ainvoke.
        assert body1["error"] is None

        r2 = await client.post(
            "/sessions/http_flight_1/intent",
            json={"intent": "SelectOption", "slots": {"option_id": "opt_1"}},
        )
        assert r2.status_code == 200, r2.text
        body2 = r2.json()
        assert body2["awaiting_confirmation"] is True
        assert body2["confirmation_question"]["policy_compliant"] is True

        r3 = await client.post("/sessions/http_flight_1/confirm", json={"confirmed": True})
        assert r3.status_code == 200, r3.text
        body3 = r3.json()
        assert body3["current_state"] == "order_confirmed"
        assert body3["final_result"]["order_id"] == "ord_123"

        r4 = await client.get("/sessions/http_flight_1/state")
        assert r4.status_code == 200, r4.text
        assert r4.json()["current_state"] == "order_confirmed"


@pytest.mark.asyncio
async def test_full_hotel_flow_over_http_reuses_same_endpoints():
    """Тот же путь, что и flight, но SearchHotel — доказывает, что HTTP-слой
    тоже не знает разницы между доменами (никакого /hotels/select vs
    /flights/select — один и тот же /intent для всех)."""
    async with _make_client(policy_compliant=True, approval_required=False) as client:
        r1 = await client.post(
            "/sessions/http_hotel_1/intent",
            json={"intent": "SearchHotel", "slots": {
                "destination": "Amsterdam", "check_in": "2026-03-20", "check_out": "2026-03-23",
            }},
        )
        assert r1.status_code == 200, r1.text

        r2 = await client.post(
            "/sessions/http_hotel_1/intent",
            json={"intent": "SelectOption", "slots": {"option_id": "hotel_opt_1"}},
        )
        assert r2.json()["awaiting_confirmation"] is True

        r3 = await client.post("/sessions/http_hotel_1/confirm", json={"confirmed": True})
        assert r3.json()["current_state"] == "order_confirmed"


@pytest.mark.asyncio
async def test_check_order_status_standalone_intent_over_http():
    async with _make_client() as client:
        r = await client.post(
            "/sessions/http_status_1/intent",
            json={"intent": "CheckOrderStatus", "slots": {"order_id": "ord_99"}},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["final_result"]["status"] == "confirmed"
        assert body["current_state"] == "idle"


@pytest.mark.asyncio
async def test_cancel_order_without_confirmation_slot_is_blocked_by_guardrail():
    async with _make_client() as client:
        r = await client.post(
            "/sessions/http_cancel_1/intent",
            json={"intent": "CancelOrder", "slots": {"order_id": "ord_99"}},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert "подтверждения" in body["error"]


@pytest.mark.asyncio
async def test_unsupported_intent_returns_501():
    async with _make_client() as client:
        r = await client.post(
            "/sessions/http_smalltalk_1/intent",
            json={"intent": "SmallTalk", "slots": {}},
        )
        assert r.status_code == 501
        assert "SmallTalk" in r.json()["detail"]


@pytest.mark.asyncio
async def test_missing_required_slots_returns_422():
    async with _make_client() as client:
        r = await client.post(
            "/sessions/http_incomplete_1/intent",
            json={"intent": "SearchFlight", "slots": {"origin": "LAX"}},
        )
        assert r.status_code == 422
        assert "destination" in r.json()["detail"]


@pytest.mark.asyncio
async def test_confirm_without_pending_interrupt_returns_409():
    async with _make_client() as client:
        r = await client.post("/sessions/http_no_interrupt_1/confirm", json={"confirmed": True})
        assert r.status_code == 409


@pytest.mark.asyncio
async def test_new_intent_while_awaiting_confirmation_returns_409():
    async with _make_client(policy_compliant=True, approval_required=False) as client:
        await client.post(
            "/sessions/http_conflict_1/intent",
            json={"intent": "SearchFlight", "slots": {
                "origin": "LAX", "destination": "JFK", "date_from": "2026-08-07",
            }},
        )
        r_select = await client.post(
            "/sessions/http_conflict_1/intent",
            json={"intent": "SelectOption", "slots": {"option_id": "opt_1"}},
        )
        assert r_select.json()["awaiting_confirmation"] is True

        # граф стоит на interrupt() — новый SearchHotel в этом же
        # session_id должен быть отклонён, а не молча смешан с ожидающим
        # подтверждением потоком.
        r_conflict = await client.post(
            "/sessions/http_conflict_1/intent",
            json={"intent": "SearchHotel", "slots": {
                "destination": "Paris", "check_in": "2026-04-01", "check_out": "2026-04-03",
            }},
        )
        assert r_conflict.status_code == 409


@pytest.mark.asyncio
async def test_state_endpoint_404_for_unknown_session():
    async with _make_client() as client:
        r = await client.get("/sessions/never_existed/state")
        assert r.status_code == 404
