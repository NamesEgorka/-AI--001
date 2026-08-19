"""
Клиент публичного MCP-сервера trivago (https://mcp.trivago.com).

Как и Kiwi.com для рейсов — публичный, без ключа, знает только про сами
отели (цена/рейтинг/удобства), НЕ про тревел-политику, лимиты пользователя
или согласование. Та же граница ответственности, что и у KiwiFlightClient
(см. AGENTS.md/README про разделение источников истины).

ВНИМАНИЕ (честно, как и с Kiwi): в песочнице, где я это писал, сеть
ограничена белым списком доменов, mcp.trivago.com туда не входит —
реальный вызов не был протестирован живьём. Код написан по официальной
документации (https://mcp.trivago.com/docs): Streamable HTTP MCP endpoint
по адресу https://mcp.trivago.com/mcp, инструмент search_hotels,
аутентификация не требуется. Прогоните в своём окружении (например,
в Codespace, где сеть не ограничена) перед боевым использованием.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from orchestrator.state import SearchResultSnapshot
from orchestrator.tracing import trace_tool_call

TRIVAGO_MCP_URL = "https://mcp.trivago.com/mcp"
TRIVAGO_TOOL_NAME = "search_hotels"


class TrivagoSearchError(Exception):
    pass


@dataclass
class TrivagoHotelClient:
    server_url: str = TRIVAGO_MCP_URL

    async def search_hotels(
        self,
        *,
        trace_id: str,
        turn_id: str,
        session_id: str,
        destination: str,
        check_in: str,
        check_out: str,
        guests: int = 1,
    ) -> SearchResultSnapshot:
        """
        Вызывает search_hotels на trivago MCP и упаковывает ответ в
        SearchResultSnapshot — тот же формат, что и у поиска рейсов,
        поэтому guardrails.validate_selected_option работает одинаково
        для обоих доменов без каких-либо изменений.
        """
        arguments = {
            "destination": destination,
            "check_in": check_in,
            "check_out": check_out,
            "guests": guests,
        }

        with trace_tool_call(
            trace_id=trace_id, turn_id=turn_id, session_id=session_id,
            tool_name=TRIVAGO_TOOL_NAME,
        ):
            try:
                async with streamable_http_client(self.server_url) as (read, write, _):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        result = await session.call_tool(TRIVAGO_TOOL_NAME, arguments)
            except Exception as exc:  # noqa: BLE001
                raise TrivagoSearchError(
                    f"Ошибка вызова trivago MCP search_hotels: {exc}"
                ) from exc

        options = self._parse_options(result)
        return SearchResultSnapshot(
            search_id=f"trivago_{turn_id}",
            intent="SearchHotel",
            options=options,
        )

    @staticmethod
    def _parse_options(tool_result: Any) -> list[dict[str, Any]]:
        """
        Нормализует ответ trivago в тот же формат, что и у Kiwi: список
        словарей с обязательным option_id и меткой _tool_source.

        # TODO: сверить точную структуру ответа search_hotels с реальным
        # вызовом в вашем окружении (см. предупреждение о сетевых
        # ограничениях песочницы вверху файла) — формат ниже основан на
        # примере из документации (hotel name, rating, price per night,
        # amenities), а не на живом ответе.
        """
        raw_items = getattr(tool_result, "content", None) or []
        options: list[dict[str, Any]] = []
        for idx, item in enumerate(raw_items):
            payload = getattr(item, "data", None) or getattr(item, "text", None) or {}
            if isinstance(payload, dict):
                option = dict(payload)
            else:
                option = {"raw": payload}
            option.setdefault("option_id", f"trivago_opt_{idx}")
            option["_tool_source"] = TRIVAGO_TOOL_NAME
            options.append(option)
        return options
