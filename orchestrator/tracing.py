"""
Трассировка tool calling.

Закрывает пробел "нет трейсинга tool calling с trace_id/turn_id" —
раньше это было только полем в JSON Schema NLUOutput, но никто реально
не логировал вызовы. Здесь каждый вызов инструмента оборачивается
и пишет структурированную запись, по которой потом в реальном инциденте
("почему агент сказал X") можно восстановить всю цепочку:
реплика -> NLU -> tool call -> tool result -> guardrail check -> ответ.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Iterator, Optional

logger = logging.getLogger("travel_agent.trace")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))  # уже JSON, форматтер не нужен
    logger.addHandler(_handler)


@dataclass
class ToolCallRecord:
    trace_id: str
    turn_id: str
    session_id: str
    tool_name: str
    status: str  # "started" | "success" | "error" | "guardrail_blocked"
    latency_ms: Optional[float] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    # PII-guardrail: сюда НЕЛЬЗЯ класть сырые payload/response целиком, если
    # там могут быть паспортные/платёжные данные. Логируем только метаданные;
    # полный payload — в отдельный защищённый sink при необходимости (TODO prod).

    def emit(self) -> None:
        logger.info(json.dumps(asdict(self), ensure_ascii=False))


@contextmanager
def trace_tool_call(
    *, trace_id: str, turn_id: str, session_id: str, tool_name: str
) -> Iterator[None]:
    """
    Использование:

        with trace_tool_call(trace_id=..., turn_id=..., session_id=..., tool_name="search-flight"):
            result = call_kiwi_search(...)

    Пишет запись "started", затем "success" или "error" с latency.
    Guardrail-блокировки логируются отдельно через log_guardrail_block,
    так как они происходят ДО или ПОСЛЕ вызова tool'а, а не во время него.
    """
    started_at = time.monotonic()
    ToolCallRecord(
        trace_id=trace_id, turn_id=turn_id, session_id=session_id,
        tool_name=tool_name, status="started",
    ).emit()
    try:
        yield
    except Exception as exc:  # noqa: BLE001 — трейсинг обязан поймать всё и перебросить
        ToolCallRecord(
            trace_id=trace_id, turn_id=turn_id, session_id=session_id,
            tool_name=tool_name, status="error",
            latency_ms=(time.monotonic() - started_at) * 1000,
            error_type=type(exc).__name__, error_message=str(exc),
        ).emit()
        raise
    else:
        ToolCallRecord(
            trace_id=trace_id, turn_id=turn_id, session_id=session_id,
            tool_name=tool_name, status="success",
            latency_ms=(time.monotonic() - started_at) * 1000,
        ).emit()


def log_guardrail_block(
    *, trace_id: str, turn_id: str, session_id: str, guardrail_name: str, reason: str
) -> None:
    """Отдельная запись для случаев, когда guardrail заблокировал операцию."""
    ToolCallRecord(
        trace_id=trace_id, turn_id=turn_id, session_id=session_id,
        tool_name=guardrail_name, status="guardrail_blocked",
        error_message=reason,
    ).emit()
