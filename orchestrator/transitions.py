"""
Явная таблица разрешённых переходов между состояниями диалога.

Это прямой ответ на пробел "нет валидации переходов между состояниями
Intent Map": раньше единственным местом, где жила эта логика, был текст
промпта в AGENTS.md — то есть LLM мог в теории "решить" пропустить
policy_check и сразу перейти к order_creating, если его удастся уговорить.
Теперь это невозможно технически: переход, которого нет в этой таблице,
кодом отклоняется ДО вызова какого-либо инструмента.
"""

from __future__ import annotations

from .state import DialogueStateName

ALLOWED_TRANSITIONS: dict[DialogueStateName, set[DialogueStateName]] = {
    # idle -> searching разрешён напрямую: если пользователь одной фразой
    # сразу назвал все обязательные параметры, промежуточное состояние
    # collecting_params логически "пройдено" мгновенно, и незачем требовать
    # от Orchestrator'а искусственной остановки в нём.
    "idle": {"collecting_params", "searching", "intent_switch_confirm"},
    "collecting_params": {"collecting_params", "searching", "intent_switch_confirm"},
    "searching": {"results_shown", "collecting_params"},  # collecting_params — если поиск вернул пусто и нужно уточнить
    "results_shown": {"policy_check", "collecting_params", "searching", "intent_switch_confirm"},
    "policy_check": {"policy_result"},
    "policy_result": {"approval_pending", "approval_not_required", "results_shown"},
    "approval_pending": {"order_creating", "results_shown"},  # order_creating только если approval одобрен
    "approval_not_required": {"order_creating"},
    "order_creating": {"order_confirmed", "order_failed"},
    "order_confirmed": {"idle", "status_check", "cancel_confirm"},
    "order_failed": {"order_creating", "idle"},  # повторная попытка — ТОЛЬКО по явному новому запросу пользователя
    "status_check": {"idle"},
    "cancel_confirm": {"cancelled", "cancel_failed", "idle"},
    "cancelled": {"idle"},
    "cancel_failed": {"cancel_confirm", "idle"},
    "intent_switch_confirm": {"collecting_params", "idle"},
}


class InvalidTransitionError(Exception):
    def __init__(self, current: DialogueStateName, target: DialogueStateName):
        self.current = current
        self.target = target
        super().__init__(
            f"Переход {current!r} -> {target!r} запрещён. "
            f"Разрешённые переходы из {current!r}: "
            f"{sorted(ALLOWED_TRANSITIONS.get(current, set()))}"
        )


def assert_valid_transition(current: DialogueStateName, target: DialogueStateName) -> None:
    if target not in ALLOWED_TRANSITIONS.get(current, set()):
        raise InvalidTransitionError(current, target)
