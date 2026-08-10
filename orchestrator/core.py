"""
Orchestrator — детерминированный код, который решает, что делать с
результатом NLU-слоя: вызвать инструмент, задать уточняющий вопрос
или отклонить действие по guardrail-причине.

Это не полный production-граф (LangGraph-версию с полным набором нод
для каждого intent имеет смысл разворачивать поверх этого модуля),
а рабочее ядро с двумя самыми критичными путями, которые были прямыми
пробелами в предыдущей версии:

  1. select_option() — anti-hallucination выбор варианта поездки
  2. create_order()  — идемпотентное создание заказа с guardrail-цепочкой
     (policy -> approval -> order), где каждый шаг обязан быть
     подтверждён источником-истины, а не сгенерирован моделью
"""

from __future__ import annotations

from typing import Any

from tools.internal_api_client import InternalApiClient
from tools.kiwi_client import KiwiFlightClient

from .guardrails import (
    GuardrailViolation,
    IdempotencyStore,
    validate_approval_status_source,
    validate_policy_verdict_source,
    validate_selected_option,
)
from .state import DialogueState, SearchResultSnapshot
from .tracing import log_guardrail_block
from .transitions import assert_valid_transition


class Orchestrator:
    def __init__(
        self,
        *,
        internal_api: InternalApiClient | None = None,
        kiwi_client: KiwiFlightClient | None = None,
        idempotency_store: IdempotencyStore | None = None,
    ) -> None:
        self.internal_api = internal_api or InternalApiClient()
        self.kiwi_client = kiwi_client or KiwiFlightClient()
        self.idempotency_store = idempotency_store or IdempotencyStore()

    # --- SearchFlight -----------------------------------------------------

    async def search_flights(
        self, state: DialogueState, *, origin: str, destination: str, date_from: str,
        passengers: int = 1,
    ) -> DialogueState:
        assert_valid_transition(state.current_state, "searching")
        state.transition("searching")

        turn_id = state.new_turn_id()
        snapshot = await self.kiwi_client.search_flights(
            trace_id=state.trace_id, turn_id=turn_id, session_id=state.session_id,
            origin=origin, destination=destination, date_from=date_from,
            passengers=passengers,
        )
        state.last_search_result = snapshot

        target = "results_shown" if snapshot.options else "collecting_params"
        assert_valid_transition(state.current_state, target)
        state.transition(target)
        return state

    # --- SelectOption (anti-hallucination guardrail) -----------------------

    def select_option(self, state: DialogueState, *, option_id: str) -> dict[str, Any]:
        """
        Ключевой guardrail-путь: option_id обязан существовать в последнем
        SearchResultSnapshot. Если LLM "придумал" вариант — это долетает
        сюда как GuardrailViolation, а не как красивый, но ложный ответ
        пользователю.
        """
        try:
            option = validate_selected_option(option_id, state.last_search_result)
        except GuardrailViolation as exc:
            log_guardrail_block(
                trace_id=state.trace_id, turn_id=state.new_turn_id(),
                session_id=state.session_id, guardrail_name="validate_selected_option",
                reason=str(exc),
            )
            raise
        state.order_draft = {"selected_option": option}
        return option

    # --- CheckPolicyCompliance ---------------------------------------------

    async def check_policy(
        self, state: DialogueState, *, user_id: str
    ) -> dict[str, Any]:
        assert_valid_transition(state.current_state, "policy_check")
        state.transition("policy_check")

        turn_id = state.new_turn_id()
        verdict = await self.internal_api.get_travel_policy(
            trace_id=state.trace_id, turn_id=turn_id, session_id=state.session_id,
            user_id=user_id, trip_context=state.order_draft or {},
        )
        verdict = validate_policy_verdict_source(verdict)  # source-of-truth guardrail
        state.policy_verdict = verdict

        assert_valid_transition(state.current_state, "policy_result")
        state.transition("policy_result")
        return verdict

    # --- RequestApproval / CreateOrder (идемпотентность) --------------------

    async def create_order(self, state: DialogueState, *, user_confirmed: bool) -> dict[str, Any]:
        """
        Полная цепочка guardrail-проверок перед фактическим созданием заказа:
          1. policy_verdict обязан быть от get_travel_policy (не выдуман)
          2. approval_status обязан быть от get_approval_requirements
          3. явное подтверждение пользователя (не молчаливое допущение)
          4. идемпотентный ключ — защита от двойного заказа при retry
        """
        if not user_confirmed:
            raise GuardrailViolation(
                "CreateOrder вызван без явного подтверждения пользователя — "
                "заблокировано (см. Guardrails п.3 и правило "
                "'обязательно уточнять' из раздела 1.4)."
            )

        validate_policy_verdict_source(state.policy_verdict)
        if state.approval_status is not None:
            validate_approval_status_source(state.approval_status)
            approved = bool(state.approval_status.get("approved"))
            intermediate_target = "approval_pending"
        else:
            approved = True  # approval не требовался вовсе
            intermediate_target = "approval_not_required"

        # Всегда явно проходим через промежуточное состояние (approval_pending
        # или approval_not_required) — это отражает Intent Map буквально:
        # policy_result -> {approval_pending | approval_not_required} -> order_creating.
        # Прямой прыжок policy_result -> order_creating запрещён таблицей
        # переходов намеренно, чтобы нельзя было его случайно "обойти".
        assert_valid_transition(state.current_state, intermediate_target)
        state.transition(intermediate_target)

        if not approved:
            raise GuardrailViolation(
                "Approval ещё не получен — CreateOrder не может быть вызван."
            )

        assert_valid_transition(state.current_state, "order_creating")
        state.transition("order_creating")

        idempotency_key = self.idempotency_store.build_key(
            session_id=state.session_id, operation="create_order",
            payload=state.order_draft or {},
        )
        self.idempotency_store.check_and_reserve(idempotency_key)  # бросит DuplicateOperationError при повторе

        turn_id = state.new_turn_id()
        try:
            result = await self.internal_api.create_order(
                trace_id=state.trace_id, turn_id=turn_id, session_id=state.session_id,
                order_draft=state.order_draft or {}, idempotency_key=idempotency_key,
            )
        except Exception:
            self.idempotency_store.mark_failed(idempotency_key)
            assert_valid_transition(state.current_state, "order_failed")
            state.transition("order_failed")
            raise

        self.idempotency_store.mark_completed(idempotency_key, result)
        assert_valid_transition(state.current_state, "order_confirmed")
        state.transition("order_confirmed")
        return result
