"""
Состояние диалога (Dialogue State) — единственный источник истины о том,
на каком шаге сейчас находится разговор.

ВАЖНО: LLM НЕ управляет переходами между состояниями напрямую. NLU-слой
только предлагает intent/сущности; переходы state -> state валидирует
код Orchestrator'а (см. graph.py), что и есть наша защита от того, что
модель "решит" перейти туда, куда переходить в этот момент нельзя
(например, сразу в order_creating, минуя policy_check).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

# Состояния, зеркалящие колонку "Переходы состояний" из Intent Map
DialogueStateName = Literal[
    "idle",
    "collecting_params",
    "searching",
    "results_shown",
    "policy_check",
    "policy_result",
    "approval_pending",
    "approval_not_required",
    "order_creating",
    "order_confirmed",
    "order_failed",
    "status_check",
    "cancel_confirm",
    "cancelled",
    "cancel_failed",
    "intent_switch_confirm",
]


@dataclass
class SearchResultSnapshot:
    """
    Снимок результата поиска с TTL и версией.

    Это защита от ситуации "агент подтвердил рейс, который уже неактуален":
    SelectOption валидируется именно против этого объекта, а не против
    того, что LLM "помнит" из своего вывода.
    """

    search_id: str
    intent: str  # SearchFlight | SearchTrain | SearchHotel
    options: list[dict[str, Any]]  # сырые option'ы, как вернул tool (option_id внутри)
    fetched_at: float = field(default_factory=time.time)
    ttl_seconds: int = 600  # 10 минут по умолчанию — тарифы протухают быстро

    def is_expired(self) -> bool:
        return (time.time() - self.fetched_at) > self.ttl_seconds

    def find_option(self, option_id: str) -> Optional[dict[str, Any]]:
        return next((o for o in self.options if o.get("option_id") == option_id), None)


@dataclass
class DialogueState:
    session_id: str
    trace_id: str = field(default_factory=lambda: f"tr_{uuid.uuid4().hex[:12]}")

    current_state: DialogueStateName = "idle"
    active_intent: Optional[str] = None

    # Собранные (но ещё не обязательно провалидированные через справочники) слоты
    collected_slots: dict[str, Any] = field(default_factory=dict)
    missing_required_slots: list[str] = field(default_factory=list)

    # Последний результат поиска — с TTL/версией, для anti-hallucination проверки SelectOption
    last_search_result: Optional[SearchResultSnapshot] = None

    # Результаты вызовов внутренних API за текущий цикл сбора заказа —
    # хранятся отдельно, чтобы явно видеть источник каждого факта
    policy_verdict: Optional[dict[str, Any]] = None  # только из get_travel_policy tool
    approval_status: Optional[dict[str, Any]] = None  # только из approval tool
    order_draft: Optional[dict[str, Any]] = None

    # История для anaphora resolution в NLU-слое (последние N реплик)
    turn_history: list[dict[str, str]] = field(default_factory=list)

    def new_turn_id(self) -> str:
        return f"t_{uuid.uuid4().hex[:12]}"

    def transition(self, new_state: DialogueStateName) -> None:
        """
        Единственная разрешённая точка изменения current_state.
        Оставлено намеренно "глупой" здесь — валидация допустимости
        конкретного перехода (какая state -> state пара разрешена)
        живёт в graph.py, чтобы её было легко покрыть тестами отдельно
        от самого объекта состояния.
        """
        self.current_state = new_state
