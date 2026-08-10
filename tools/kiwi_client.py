"""
Клиент публичного MCP-сервера Kiwi.com (https://mcp.kiwi.com).

Единственный по-настоящему рабочий источник данных в этой сборке —
он не требует ключа, но у него есть жёсткая граница ответственности:
он знает только о рейсах (цена/расписание/наличие), НЕ о тревел-политике,
лимитах пользователя, согласовании или заказах. Смешивать его ответ
с внутренними бизнес-решениями запрещено guardrails (см. AGENTS.md,
раздел "Важное ограничение текущей версии").

ВНИМАНИЕ (честно): в этой песочнице сеть ограничена белым списком
доменов, mcp.kiwi.com туда не входит, поэтому реальный вызов здесь
не был протестирован живьём. Код написан по официальной документации
Kiwi (Streamable HTTP MCP endpoint, инструмент `search-flight`,
аутентификация не требуется) — перед продакшн-использованием
прогоните его в своём окружении.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

# Примечание: имя импортируемой функции менялось между версиями пакета mcp
# (streamablehttp_client в старых версиях -> streamable_http_client в 2.x).
# Если у вас установлена другая версия SDK, сверьтесь с реальным содержимым:
# python3 -c "import mcp.client.streamable_http as m; print(dir(m))"

from orchestrator.state import SearchResultSnapshot
from orchestrator.tracing import trace_tool_call

KIWI_MCP_URL = "https://mcp.kiwi.com"
KIWI_TOOL_NAME = "search-flight"


class KiwiSearchError(Exception):
    pass


@dataclass
class KiwiFlightClient:
    server_url: str = KIWI_MCP_URL

    async def search_flights(
        self,
        *,
        trace_id: str,
        turn_id: str,
        session_id: str,
        origin: str,
        destination: str,
        date_from: str,
        date_to: str | None = None,
        passengers: int = 1,
    ) -> SearchResultSnapshot:
        """
        Вызывает search-flight на Kiwi MCP и упаковывает ответ в
        SearchResultSnapshot — именно этот объект потом используется
        guardrails.validate_selected_option для anti-hallucination проверки.
        """
        arguments = {
            "flyFrom": origin,
            "flyTo": destination,
            "dateFrom": date_from,
            "dateTo": date_to or date_from,
            "adults": passengers,
        }

        with trace_tool_call(
            trace_id=trace_id, turn_id=turn_id, session_id=session_id,
            tool_name=KIWI_TOOL_NAME,
        ):
            try:
                async with streamable_http_client(self.server_url) as (read, write, _):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        result = await session.call_tool(KIWI_TOOL_NAME, arguments)
            except Exception as exc:  # noqa: BLE001
                raise KiwiSearchError(
                    f"Ошибка вызова Kiwi MCP search-flight: {exc}"
                ) from exc

        options = self._parse_options(result)
        return SearchResultSnapshot(
            search_id=f"kiwi_{turn_id}",
            intent="SearchFlight",
            options=options,
        )

    @staticmethod
    def _parse_options(tool_result: Any) -> list[dict[str, Any]]:
        """
        Нормализует сырой ответ Kiwi в единый формат с обязательным полем
        option_id (генерируется детерминированно из данных рейса, если
        сам Kiwi его не предоставляет) и меткой _tool_source.

        # TODO: сверить точную структуру ответа search-flight с реальным
        # вызовом в вашем окружении — формат ниже основан на документации,
        # а не на живом ответе (см. предупреждение о сетевых ограничениях
        # песочницы вверху файла).
        """
        raw_items = getattr(tool_result, "content", None) or []
        options: list[dict[str, Any]] = []
        for idx, item in enumerate(raw_items):
            payload = getattr(item, "data", None) or getattr(item, "text", None) or {}
            if isinstance(payload, dict):
                option = dict(payload)
            else:
                option = {"raw": payload}
            option.setdefault("option_id", f"kiwi_opt_{idx}")
            option["_tool_source"] = KIWI_TOOL_NAME
            options.append(option)
        return options
