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
from tools.train_client import FakeTrainClient
from tools.trivago_client import TrivagoHotelClient

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
        hotel_client: TrivagoHotelClient | None = None,
        train_client: FakeTrainClient | None = None,
        idempotency_store: IdempotencyStore | None = None,
    ) -> None:
        self.internal_api = internal_api or InternalApiClient()
        self.kiwi_client = kiwi_client or KiwiFlightClient()
        self.hotel_client = hotel_client or TrivagoHotelClient()
        self.train_client = train_client or FakeTrainClient()
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

    # --- SearchHotel ----------------------------------------------------

    async def search_hotels(
        self, state: DialogueState, *, destination: str, check_in: str, check_out: str,
        guests: int = 1,
    ) -> DialogueState:
        """
        Зеркало search_flights, только источник данных — trivago вместо
        Kiwi. Обратите внимание: результат кладётся в тот же
        last_search_result — SelectOption работает одинаково для рейсов
        и отелей, потому что оба клиента возвращают SearchResultSnapshot
        одного и того же формата.
        """
        assert_valid_transition(state.current_state, "searching")
        state.transition("searching")

        turn_id = state.new_turn_id()
        snapshot = await self.hotel_client.search_hotels(
            trace_id=state.trace_id, turn_id=turn_id, session_id=state.session_id,
            destination=destination, check_in=check_in, check_out=check_out,
            guests=guests,
        )
        state.last_search_result = snapshot

        target = "results_shown" if snapshot.options else "collecting_params"
        assert_valid_transition(state.current_state, target)
        state.transition(target)
        return state

    # --- SearchTrain --------------------------------------------------------

    async def search_trains(
        self, state: DialogueState, *, origin: str, destination: str, date_from: str,
        passengers: int = 1,
    ) -> DialogueState:
        """
        Третье зеркало search_flights/search_hotels. Источник данных —
        FakeTrainClient (честная заглушка, см. tools/train_client.py —
        публичного no-key API с ценой/наличием мест для ЖД не нашлось,
        в отличие от Kiwi/trivago). Путь ниже (select_option/check_policy/
        create_order) не меняется ни на строчку — тот же SearchResultSnapshot.
        """
        assert_valid_transition(state.current_state, "searching")
        state.transition("searching")

        turn_id = state.new_turn_id()
        snapshot = await self.train_client.search_trains(
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

    # --- CheckOrderStatus ---------------------------------------------------

    async def check_order_status(self, state: DialogueState, *, order_id: str) -> dict[str, Any]:
        """
        Самый простой intent из Intent Map: один синхронный вызов внутреннего
        API, без цепочки guardrail-проверок (тут нечего "придумать" — либо
        API вернул статус, либо нет).
        """
        assert_valid_transition(state.current_state, "status_check")
        state.transition("status_check")

        turn_id = state.new_turn_id()
        status = await self.internal_api.get_order_status(
            trace_id=state.trace_id, turn_id=turn_id, session_id=state.session_id,
            order_id=order_id,
        )

        assert_valid_transition(state.current_state, "idle")
        state.transition("idle")
        return status

    # --- CancelOrder (идемпотентность, как и CreateOrder) --------------------

    async def cancel_order(
        self, state: DialogueState, *, order_id: str, user_confirmed: bool
    ) -> dict[str, Any]:
        """
        Как и CreateOrder — критичная операция с идемпотентностью, чтобы
        повторный вызов (например, из-за сетевого retry) не попытался
        отменить один и тот же заказ дважды. В отличие от CreateOrder,
        здесь нет цепочки policy/approval — отмена не требует проверки
        тревел-политики, только явное подтверждение пользователя.
        """
        if not user_confirmed:
            raise GuardrailViolation(
                "CancelOrder вызван без явного подтверждения пользователя — "
                "заблокировано."
            )

        assert_valid_transition(state.current_state, "cancel_confirm")
        state.transition("cancel_confirm")

        idempotency_key = self.idempotency_store.build_key(
            session_id=state.session_id, operation="cancel_order",
            payload={"order_id": order_id},
        )
        self.idempotency_store.check_and_reserve(idempotency_key)

        turn_id = state.new_turn_id()
        try:
            result = await self.internal_api.cancel_order(
                trace_id=state.trace_id, turn_id=turn_id, session_id=state.session_id,
                order_id=order_id, idempotency_key=idempotency_key,
            )
        except Exception:
            self.idempotency_store.mark_failed(idempotency_key)
            assert_valid_transition(state.current_state, "cancel_failed")
            state.transition("cancel_failed")
            raise

        self.idempotency_store.mark_completed(idempotency_key, result)
        assert_valid_transition(state.current_state, "cancelled")
        state.transition("cancelled")
        return result
