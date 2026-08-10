"""
Guardrails, реализованные в коде, а не только текстом промпта.

Раньше (в Fleet AGENTS.md) все шесть правил из раздела "Guardrails"
архитектурного документа существовали только как инструкция для LLM —
то есть модель МОГЛА их нарушить при неудачной генерации или
провокационном вводе пользователя, и ничто в системе бы этого не
поймало. Здесь каждое правило — это функция, которая либо пропускает
данные дальше, либо бросает GuardrailViolation, и Orchestrator обязан
её вызвать до того, как что-либо попадёт в ответ пользователю.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .state import SearchResultSnapshot


class GuardrailViolation(Exception):
    """Базовое исключение для любого нарушения guardrail-правила."""


class HallucinatedOptionError(GuardrailViolation):
    """option_id, который выбрал агент, отсутствует в последнем результате поиска."""


class ExpiredSearchResultError(GuardrailViolation):
    """Результат поиска устарел (TTL истёк) — нужен повторный поиск перед выбором."""


class MissingSourceOfTruthError(GuardrailViolation):
    """Попытка утверждать факт (политика/цена/approval), для которого нет ответа tool'а."""


class DuplicateOperationError(GuardrailViolation):
    """Повторный вызов критичной операции (CreateOrder/CancelOrder) с тем же ключом."""


# --- Guardrail 1 и 6: anti-hallucination на выбор варианта ------------------

def validate_selected_option(
    option_id: str, last_search_result: Optional[SearchResultSnapshot]
) -> dict[str, Any]:
    """
    Разрешает SelectOption ТОЛЬКО если option_id реально присутствует
    в последнем результате поиска и этот результат ещё не протух.
    Возвращает сам option (сырые данные из tool'а), чтобы дальше по коду
    использовались именно они, а не то, что LLM "запомнил".
    """
    if last_search_result is None:
        raise HallucinatedOptionError(
            "Нет ни одного результата поиска в этой сессии — SelectOption невозможен."
        )
    if last_search_result.is_expired():
        raise ExpiredSearchResultError(
            f"Результат поиска {last_search_result.search_id} устарел "
            f"(TTL {last_search_result.ttl_seconds}s). Нужен повторный SearchX."
        )
    option = last_search_result.find_option(option_id)
    if option is None:
        raise HallucinatedOptionError(
            f"option_id={option_id!r} отсутствует в результате поиска "
            f"{last_search_result.search_id!r}. Похоже на попытку выбрать "
            f"вариант, которого не существует."
        )
    return option


# --- Guardrail 2: интерпретация тревел-политики запрещена --------------------

def validate_policy_verdict_source(policy_verdict: Optional[dict[str, Any]]) -> dict[str, Any]:
    """
    Разрешает переход в policy_result ТОЛЬКО если policy_verdict пришёл
    от get_travel_policy tool в этом же цикле (проверяется по наличию
    служебного поля _tool_source, которое подставляет только код клиента
    инструмента — см. tools/internal_api_client.py — LLM не может
    сформировать это поле сам через генерацию текста).
    """
    if policy_verdict is None:
        raise MissingSourceOfTruthError(
            "Нет ответа от get_travel_policy — вердикт по политике не может "
            "быть вынесен."
        )
    if policy_verdict.get("_tool_source") != "get_travel_policy":
        raise MissingSourceOfTruthError(
            "policy_verdict не помечен как результат вызова get_travel_policy — "
            "похоже на попытку подставить сгенерированный вердикт вместо "
            "реального ответа API."
        )
    return policy_verdict


def validate_approval_status_source(approval_status: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Аналогичная проверка для approval — см. validate_policy_verdict_source."""
    if approval_status is None:
        raise MissingSourceOfTruthError(
            "Нет ответа от get_approval_requirements — решение об approval "
            "не может быть вынесено."
        )
    if approval_status.get("_tool_source") != "get_approval_requirements":
        raise MissingSourceOfTruthError(
            "approval_status не помечен как результат вызова "
            "get_approval_requirements."
        )
    return approval_status


# --- Guardrail 3 + идемпотентность: подтверждение заказа --------------------

@dataclass
class IdempotencyStore:
    """
    Простое in-memory хранилище идемпотентных ключей для критичных операций
    (CreateOrder, CancelOrder). В продакшне нужно заменить на Redis/Postgres
    (см. TODO ниже) — интерфейс намеренно минимальный, чтобы замена была
    тривиальной.
    """

    _seen: dict[str, dict[str, Any]] = field(default_factory=dict)

    @staticmethod
    def build_key(session_id: str, operation: str, payload: dict[str, Any]) -> str:
        """
        Ключ строится из session_id + operation + хэша payload, чтобы:
        - один и тот же order_draft, отправленный дважды подряд (например,
          из-за retry на сетевом сбое), не создал два заказа;
        - при этом разные заказы в рамках одной сессии не блокировали друг друга.
        """
        payload_hash = hashlib.sha256(
            repr(sorted(payload.items())).encode("utf-8")
        ).hexdigest()[:16]
        return f"{session_id}:{operation}:{payload_hash}"

    def check_and_reserve(self, key: str) -> None:
        """
        Бросает DuplicateOperationError, если такой ключ уже зарезервирован
        (то есть операция уже выполняется или выполнена). Иначе резервирует
        ключ атомарно относительно текущего процесса.

        TODO(prod): заменить на Redis SETNX с TTL, чтобы резервирование было
        атомарным между несколькими инстансами Orchestrator'а, а не только
        внутри одного процесса.
        """
        if key in self._seen:
            raise DuplicateOperationError(
                f"Операция с ключом {key!r} уже была инициирована "
                f"в {self._seen[key]['reserved_at']}. Повторный вызов заблокирован "
                f"для защиты от двойного бронирования/отмены."
            )
        self._seen[key] = {"reserved_at": time.time(), "status": "reserved"}

    def mark_completed(self, key: str, result: dict[str, Any]) -> None:
        if key in self._seen:
            self._seen[key]["status"] = "completed"
            self._seen[key]["result"] = result

    def mark_failed(self, key: str) -> None:
        """
        При неуспехе освобождаем ключ, чтобы явный повторный запрос
        пользователя (не автоматический retry!) мог пройти снова.
        """
        self._seen.pop(key, None)


# --- Guardrail 4: цена/наличие только из tool result -------------------------

def assert_has_tool_source(value: Any, tool_name: str, field_label: str) -> None:
    """
    Универсальная проверка: любое значение, которое уйдёт в ответ пользователю
    как факт (цена, наличие, статус), обязано быть словарём с меткой
    _tool_source == tool_name. Используется в местах сборки финального
    ответа, чтобы не дать LLM подставить "правдоподобное" число вместо
    реального результата инструмента.
    """
    if not isinstance(value, dict) or value.get("_tool_source") != tool_name:
        raise MissingSourceOfTruthError(
            f"{field_label} должен быть получен из tool={tool_name!r}, "
            f"но не помечен соответствующим _tool_source."
        )
