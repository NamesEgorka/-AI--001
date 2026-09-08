"""Локальный провайдер рейсов для MVP без внешних API и ключей."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from orchestrator.state import SearchResultSnapshot
from orchestrator.tracing import trace_tool_call


class FlightApiAdapterError(Exception):
    pass


@dataclass
class FlightApiAdapter:
    """Детерминированный источник демонстрационных вариантов перелёта.

    Имя класса сохранено для совместимости с текущим wiring оркестратора.
    В продакшене этот класс можно заменить отдельным внешним адаптером,
    не меняя контракт ``search_flights`` и формат snapshot.
    """

    provider_name: str = "mvp-local"

    def __post_init__(self) -> None:
        self.provider_name = os.getenv("FLIGHT_PROVIDER", self.provider_name)

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
        with trace_tool_call(
            trace_id=trace_id,
            turn_id=turn_id,
            session_id=session_id,
            tool_name="search-flight",
        ):
            options = self._build_options(
                origin=origin,
                destination=destination,
                date_from=date_from,
                passengers=passengers,
            )

        return SearchResultSnapshot(
            search_id=f"flight_{turn_id}",
            intent="SearchFlight",
            options=options,
        )

    @staticmethod
    def _build_options(
        *, origin: str, destination: str, date_from: str, passengers: int
    ) -> list[dict[str, Any]]:
        base_price = 185 + (sum(ord(char) for char in origin + destination) % 220)
        flights = (
            ("MVP Air", "MVP101", "08:15", "12:05", base_price),
            ("North Star", "NS204", "13:40", "17:35", base_price + 74),
            ("Open Skies", "OS330", "19:20", "23:25", base_price + 128),
        )
        return [
            {
                "option_id": f"flight_opt_{idx}",
                "flight_number": flight_number,
                "carrier": carrier,
                "origin": origin,
                "destination": destination,
                "departure": f"{date_from}T{departure}:00",
                "arrival": f"{date_from}T{arrival}:00",
                "price": float(price * max(passengers, 1)),
                "currency": "USD",
                "raw": {"provider": "mvp-local", "flight_number": flight_number},
                "_tool_source": "search-flight",
            }
            for idx, (carrier, flight_number, departure, arrival, price) in enumerate(flights)
        ]
