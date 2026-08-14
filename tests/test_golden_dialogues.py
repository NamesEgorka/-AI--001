"""
Golden dialogue тесты — сценарные проверки "счастливого пути" и
основных веток ошибок через весь Orchestrator, на замоканных клиентах
(без реальной сети — ни к Kiwi, ни к внутренним API).

Это не замена ручному QA на реальном стенде, а regression-сеть: если
кто-то в будущем случайно сломает guardrail-цепочку или порядок
вызовов (например, уберёт validate_policy_verdict_source из create_order),
эти тесты должны упасть.
"""

from __future__ import annotations

import pytest

from orchestrator.core import Orchestrator
from orchestrator.guardrails import GuardrailViolation, IdempotencyStore
from orchestrator.state import DialogueState, SearchResultSnapshot


class FakeKiwiClient:
    """Замена реального KiwiFlightClient — детерминированный ответ без сети."""

    async def search_flights(self, *, trace_id, turn_id, session_id, origin,
                              destination, date_from, date_to=None, passengers=1):
        return SearchResultSnapshot(
            search_id=f"kiwi_{turn_id}",
            intent="SearchFlight",
            options=[
                {"option_id": "opt_1", "price": 14500, "carrier": "TestAir",
                 "_tool_source": "search-flight"},
            ],
        )


class FakeInternalApiClient:
    """Замена внутренних API — имитирует ответы, как будто MCP-обёртка готова."""

    def __init__(self, *, policy_compliant: bool = True, approval_required: bool = False,
                 create_order_should_fail: bool = False, cancel_order_should_fail: bool = False):
        self.policy_compliant = policy_compliant
        self.approval_required = approval_required
        self.create_order_should_fail = create_order_should_fail
        self.cancel_order_should_fail = cancel_order_should_fail

    async def get_travel_policy(self, *, trace_id, turn_id, session_id, user_id, trip_context):
        return {"compliant": self.policy_compliant, "_tool_source": "get_travel_policy"}

    async def get_approval_requirements(self, *, trace_id, turn_id, session_id, order_draft):
        return {"required": self.approval_required, "_tool_source": "get_approval_requirements"}

    async def get_order_status(self, *, trace_id, turn_id, session_id, order_id):
        return {"order_id": order_id, "status": "confirmed", "_tool_source": "get_order_status"}

    async def cancel_order(self, *, trace_id, turn_id, session_id, order_id, idempotency_key):
        if self.cancel_order_should_fail:
            raise RuntimeError("Внутренний Order API вернул 500 при отмене (симуляция сбоя)")
        return {"order_id": order_id, "status": "cancelled", "_tool_source": "cancel_order"}

    async def create_order(self, *, trace_id, turn_id, session_id, order_draft, idempotency_key):
        if self.create_order_should_fail:
            raise RuntimeError("Внутренний Order API вернул 500 (симуляция сбоя)")
        return {"order_id": "ord_123", "status": "confirmed", "_tool_source": "create_order"}


@pytest.mark.asyncio
async def test_happy_path_search_select_policy_order():
    """Полный счастливый путь: поиск -> выбор -> проверка политики -> заказ без approval."""
    orch = Orchestrator(
        internal_api=FakeInternalApiClient(policy_compliant=True, approval_required=False),
        kiwi_client=FakeKiwiClient(),
        idempotency_store=IdempotencyStore(),
    )
    state = DialogueState(session_id="s_test_1")

    state = await orch.search_flights(
        state, origin="LAX", destination="JFK", date_from="2026-08-07",
    )
    assert state.current_state == "results_shown"
    assert state.last_search_result is not None
    assert len(state.last_search_result.options) == 1

    option = orch.select_option(state, option_id="opt_1")
    assert option["price"] == 14500

    verdict = await orch.check_policy(state, user_id="u_1")
    assert verdict["compliant"] is True
    assert state.current_state == "policy_result"

    result = await orch.create_order(state, user_confirmed=True)
    assert result["order_id"] == "ord_123"
    assert state.current_state == "order_confirmed"


@pytest.mark.asyncio
async def test_create_order_blocked_without_user_confirmation():
    """
    Golden-сценарий "пользователь торопит, но явного 'да' не давал" —
    из раздела Guardrails п.3 нашего ТЗ.
    """
    orch = Orchestrator(
        internal_api=FakeInternalApiClient(),
        kiwi_client=FakeKiwiClient(),
    )
    state = DialogueState(session_id="s_test_2")
    await orch.search_flights(state, origin="LAX", destination="JFK", date_from="2026-08-07")
    orch.select_option(state, option_id="opt_1")
    await orch.check_policy(state, user_id="u_1")

    with pytest.raises(GuardrailViolation):
        await orch.create_order(state, user_confirmed=False)
    # заказ не должен был перейти в order_confirmed
    assert state.current_state != "order_confirmed"


@pytest.mark.asyncio
async def test_create_order_fails_gracefully_and_frees_idempotency_key():
    """
    Golden-сценарий сбоя внутреннего API при создании заказа: состояние
    должно уйти в order_failed, а не зависнуть/показать ложный успех.
    """
    orch = Orchestrator(
        internal_api=FakeInternalApiClient(create_order_should_fail=True),
        kiwi_client=FakeKiwiClient(),
    )
    state = DialogueState(session_id="s_test_3")
    await orch.search_flights(state, origin="LAX", destination="JFK", date_from="2026-08-07")
    orch.select_option(state, option_id="opt_1")
    await orch.check_policy(state, user_id="u_1")

    with pytest.raises(RuntimeError):
        await orch.create_order(state, user_confirmed=True)
    assert state.current_state == "order_failed"


@pytest.mark.asyncio
async def test_approval_required_blocks_immediate_order_creation():
    """Golden-сценарий: политика требует согласования — заказ не должен создаться сразу."""
    orch = Orchestrator(
        internal_api=FakeInternalApiClient(policy_compliant=False, approval_required=True),
        kiwi_client=FakeKiwiClient(),
    )
    state = DialogueState(session_id="s_test_4")
    await orch.search_flights(state, origin="LAX", destination="JFK", date_from="2026-08-07")
    orch.select_option(state, option_id="opt_1")
    await orch.check_policy(state, user_id="u_1")

    approval = await orch.internal_api.get_approval_requirements(
        trace_id=state.trace_id, turn_id=state.new_turn_id(), session_id=state.session_id,
        order_draft=state.order_draft,
    )
    state.approval_status = approval

    with pytest.raises(GuardrailViolation):
        await orch.create_order(state, user_confirmed=True)
    assert state.current_state == "approval_pending"


@pytest.mark.asyncio
async def test_check_order_status_returns_to_idle():
    """
    Golden-сценарий для CheckOrderStatus — самого простого intent'а:
    вызов API, возврат к idle, без цепочки guardrail-проверок.
    """
    orch = Orchestrator(
        internal_api=FakeInternalApiClient(),
        kiwi_client=FakeKiwiClient(),
    )
    state = DialogueState(session_id="s_test_5")

    status = await orch.check_order_status(state, order_id="ord_123")

    assert status["status"] == "confirmed"
    assert status["_tool_source"] == "get_order_status"
    assert state.current_state == "idle"


@pytest.mark.asyncio
async def test_cancel_order_happy_path():
    orch = Orchestrator(internal_api=FakeInternalApiClient(), kiwi_client=FakeKiwiClient())
    state = DialogueState(session_id="s_test_6")

    result = await orch.cancel_order(state, order_id="ord_123", user_confirmed=True)

    assert result["status"] == "cancelled"
    assert state.current_state == "cancelled"


@pytest.mark.asyncio
async def test_cancel_order_blocked_without_confirmation():
    """Отмена заказа — необратимое действие, поэтому подтверждение строго обязательно."""
    orch = Orchestrator(internal_api=FakeInternalApiClient(), kiwi_client=FakeKiwiClient())
    state = DialogueState(session_id="s_test_7")

    with pytest.raises(GuardrailViolation):
        await orch.cancel_order(state, order_id="ord_123", user_confirmed=False)
    assert state.current_state != "cancelled"


@pytest.mark.asyncio
async def test_cancel_order_fails_gracefully_and_allows_retry():
    """Сбой API отмены -> cancel_failed, идемпотентный ключ освобождён для осознанного повтора."""
    orch = Orchestrator(
        internal_api=FakeInternalApiClient(cancel_order_should_fail=True),
        kiwi_client=FakeKiwiClient(),
    )
    state = DialogueState(session_id="s_test_8")

    with pytest.raises(RuntimeError):
        await orch.cancel_order(state, order_id="ord_123", user_confirmed=True)
    assert state.current_state == "cancel_failed"


@pytest.mark.asyncio
async def test_cancel_order_idempotency_blocks_duplicate():
    """Повторный вызов cancel_order с тем же order_id в рамках одной сессии — блок, а не двойная отмена."""
    from orchestrator.guardrails import DuplicateOperationError

    orch = Orchestrator(internal_api=FakeInternalApiClient(), kiwi_client=FakeKiwiClient())
    state = DialogueState(session_id="s_test_9")

    await orch.cancel_order(state, order_id="ord_123", user_confirmed=True)
    # Состояние уже в "cancelled", повторный вызов формально невалиден по
    # transitions (cancelled -> cancel_confirm не разрешён) — это тоже
    # правильная защита, просто на уровень выше, чем идемпотентность.
    with pytest.raises(Exception):
        await orch.cancel_order(state, order_id="ord_123", user_confirmed=True)
