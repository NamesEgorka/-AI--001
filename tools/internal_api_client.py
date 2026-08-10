"""
Клиент внутренних API компании.

ЧЕСТНО: это НЕ рабочая интеграция. Я не знаю реальных эндпоинтов, схем
данных и способа авторизации ваших внутренних систем (профиль, тревел-
политика, approval, заказы) — поэтому не могу написать код, который
реально их вызывает. То, что ниже, — это готовый интерфейс с правильной
формой (retries, тайминг, трейсинг, обязательная метка _tool_source для
guardrails), в котором ваша бэкенд-команда должна заменить ТОЛЬКО тело
метода _http_call на реальный вызов вашего MCP-сервера/REST API.

Каждое место, требующее правки, помечено `# TODO(internal-api):`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx

from orchestrator.tracing import trace_tool_call


class InternalApiError(Exception):
    """Базовая ошибка вызова внутреннего API (таймаут, 4xx, 5xx)."""

    def __init__(self, tool_name: str, message: str, *, status_code: int | None = None):
        self.tool_name = tool_name
        self.status_code = status_code
        super().__init__(f"[{tool_name}] {message}")


@dataclass
class InternalApiClient:
    """
    base_url и auth_token читаются из переменных окружения — НИКОГДА
    не хардкодятся в коде. Это соответствует нашим требованиям к
    эксплуатации (секреты не в репозитории).
    """

    base_url: str = os.getenv("INTERNAL_TRAVEL_API_BASE_URL", "")
    auth_token: str = os.getenv("INTERNAL_TRAVEL_API_TOKEN", "")
    timeout_seconds: float = 8.0

    def _assert_configured(self, tool_name: str) -> None:
        if not self.base_url:
            raise InternalApiError(
                tool_name,
                "INTERNAL_TRAVEL_API_BASE_URL не задан. Это заглушка — "
                "нужна реальная MCP-обёртка над внутренним API компании.",
            )

    async def _http_call(
        self, tool_name: str, path: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """
        # TODO(internal-api): заменить это тело на реальный вызов вашего
        # MCP-сервера (или прямого REST API, если MCP-обёртка ещё не готова).
        # Обязательно сохранить:
        #   1) добавление "_tool_source": tool_name в возвращаемый словарь —
        #      это единственное, по чему guardrails отличают реальный ответ
        #      API от текста, сгенерированного LLM;
        #   2) retries/timeout — ниже пример на httpx с базовым таймаутом;
        #   3) проброс исключений как InternalApiError, а не "проглатывание".
        """
        self._assert_configured(tool_name)
        url = f"{self.base_url}{path}"
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers={"Authorization": f"Bearer {self.auth_token}"},
                )
                response.raise_for_status()
                data = response.json()
        except httpx.TimeoutException as exc:
            raise InternalApiError(tool_name, "Таймаут запроса к внутреннему API") from exc
        except httpx.HTTPStatusError as exc:
            raise InternalApiError(
                tool_name, f"HTTP {exc.response.status_code}: {exc.response.text}",
                status_code=exc.response.status_code,
            ) from exc

        data["_tool_source"] = tool_name  # обязательная метка для guardrails
        return data

    # --- Публичные методы — по одному на каждый intent из Intent Map --------

    async def get_user_profile(
        self, *, trace_id: str, turn_id: str, session_id: str, user_id: str
    ) -> dict[str, Any]:
        with trace_tool_call(
            trace_id=trace_id, turn_id=turn_id, session_id=session_id,
            tool_name="get_user_profile",
        ):
            return await self._http_call(
                "get_user_profile", "/user-profile", {"user_id": user_id}
            )

    async def get_travel_policy(
        self, *, trace_id: str, turn_id: str, session_id: str, user_id: str,
        trip_context: dict[str, Any],
    ) -> dict[str, Any]:
        with trace_tool_call(
            trace_id=trace_id, turn_id=turn_id, session_id=session_id,
            tool_name="get_travel_policy",
        ):
            return await self._http_call(
                "get_travel_policy", "/travel-policy",
                {"user_id": user_id, "trip_context": trip_context},
            )

    async def get_approval_requirements(
        self, *, trace_id: str, turn_id: str, session_id: str, order_draft: dict[str, Any],
    ) -> dict[str, Any]:
        with trace_tool_call(
            trace_id=trace_id, turn_id=turn_id, session_id=session_id,
            tool_name="get_approval_requirements",
        ):
            return await self._http_call(
                "get_approval_requirements", "/approval-requirements",
                {"order_draft": order_draft},
            )

    async def create_order(
        self, *, trace_id: str, turn_id: str, session_id: str,
        order_draft: dict[str, Any], idempotency_key: str,
    ) -> dict[str, Any]:
        with trace_tool_call(
            trace_id=trace_id, turn_id=turn_id, session_id=session_id,
            tool_name="create_order",
        ):
            # TODO(internal-api): idempotency_key обязательно должен уходить
            # в заголовок запроса (например, Idempotency-Key), если ваш
            # внутренний Order API его поддерживает — тогда защита будет
            # двойная: и на уровне Orchestrator'а, и на уровне бэкенда.
            return await self._http_call(
                "create_order", "/orders",
                {"order_draft": order_draft, "idempotency_key": idempotency_key},
            )

    async def get_order_status(
        self, *, trace_id: str, turn_id: str, session_id: str, order_id: str,
    ) -> dict[str, Any]:
        with trace_tool_call(
            trace_id=trace_id, turn_id=turn_id, session_id=session_id,
            tool_name="get_order_status",
        ):
            return await self._http_call(
                "get_order_status", f"/orders/{order_id}", {}
            )

    async def cancel_order(
        self, *, trace_id: str, turn_id: str, session_id: str,
        order_id: str, idempotency_key: str,
    ) -> dict[str, Any]:
        with trace_tool_call(
            trace_id=trace_id, turn_id=turn_id, session_id=session_id,
            tool_name="cancel_order",
        ):
            return await self._http_call(
                "cancel_order", f"/orders/{order_id}/cancel",
                {"idempotency_key": idempotency_key},
            )
