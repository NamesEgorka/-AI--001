"""
Клиент источника данных для SearchTrain.

ЧЕСТНО, по итогам поиска публичных MCP-серверов для ЖД (см. обсуждение
в чате перед этим коммитом): в отличие от Kiwi (рейсы) и trivago (отели),
для ЖД НЕ нашлось публичного no-key сервиса с тем же контрактом —
"поиск + цена + доступность мест", который можно было бы просто вызвать
по аналогии с KiwiFlightClient/TrivagoHotelClient. Что реально есть:

  - 12306 (Китай) — уже рассматривали и отклонили, слишком узкая юрисдикция
    для демо-агента, который ищет рейсы/отели глобально.
  - Deutsche Bahn Timetable MCP (несколько community-реализаций) — рабочий,
    без ключа, НО это API расписаний (станции/платформы/задержки), а не
    продажи билетов: там нет цены и мест в наличии, то есть он не даёт
    того, что нужно нашему SearchResultSnapshot для последующего
    select_option/check_policy/create_order пути.
  - SNCF (Франция) — есть MCP-обёртки, но требуют собственный API-ключ
    SNCF, а не публичный anonymous-доступ, как у Kiwi/trivago.

Поэтому SearchTrain реализован как честная заглушка (по аналогии с
tools/internal_api_client.py) — детерминированный, но правдоподобный
генератор вариантов, с тем же контрактом (SearchResultSnapshot,
option_id, _tool_source), чтобы весь путь ниже по графу
(select_option/check_policy/create_order) работал БЕЗ ИЗМЕНЕНИЙ, как и
для рейсов/отелей — ключевая находка архитектуры (см. HANDOFF.md).

Каждое место, которое нужно будет заменить на реальный вызов (когда/если
найдётся подходящий публичный ЖД-провайдер или появится ключ SNCF/другого
агрегатора), помечено `# TODO(train-api):`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from orchestrator.state import SearchResultSnapshot
from orchestrator.tracing import trace_tool_call

FAKE_TRAIN_TOOL_NAME = "search_trains_fake"

# Минимальный "справочник" операторов — только для того, чтобы заглушка
# выдавала разные, но воспроизводимые варианты, а не один и тот же option
# каждый раз. Никакого отношения к реальным тарифам не имеет.
_OPERATORS = ("EuroRail", "NightJet Express", "InterCity Line")
_TRAIN_CLASSES = ("economy", "comfort", "first")


class TrainSearchError(Exception):
    pass


@dataclass
class FakeTrainClient:
    """
    # TODO(train-api): заменить тело search_trains на реальный вызов, когда
    # появится подходящий публичный источник (или ключ SNCF/другого
    # агрегатора) — сохранить контракт метода (аргументы, возврат
    # SearchResultSnapshot) без изменений, чтобы Orchestrator.search_trains
    # и граф не пришлось трогать.
    """

    options_per_search: int = 3
    base_price_eur: int = 45

    async def search_trains(
        self,
        *,
        trace_id: str,
        turn_id: str,
        session_id: str,
        origin: str,
        destination: str,
        date_from: str,
        passengers: int = 1,
    ) -> SearchResultSnapshot:
        with trace_tool_call(
            trace_id=trace_id, turn_id=turn_id, session_id=session_id,
            tool_name=FAKE_TRAIN_TOOL_NAME,
        ):
            options = self._generate_options(
                origin=origin, destination=destination, date_from=date_from,
                passengers=passengers, turn_id=turn_id,
            )

        return SearchResultSnapshot(
            search_id=f"faketrain_{turn_id}",
            intent="SearchTrain",
            options=options,
        )

    def _generate_options(
        self, *, origin: str, destination: str, date_from: str, passengers: int,
        turn_id: str,
    ) -> list[dict[str, object]]:
        """
        Детерминированный (не случайный) генератор: одинаковый вход всегда
        даёт одинаковый выход — важно для воспроизводимых тестов/демо,
        в отличие от honest-random заглушки.
        """
        options: list[dict[str, object]] = []
        seed_base = f"{origin}|{destination}|{date_from}"

        for idx in range(self.options_per_search):
            seed = hashlib.sha256(f"{seed_base}|{idx}".encode()).hexdigest()
            operator = _OPERATORS[idx % len(_OPERATORS)]
            train_class = _TRAIN_CLASSES[idx % len(_TRAIN_CLASSES)]
            price = self.base_price_eur + (int(seed[:4], 16) % 60) + idx * 15
            departure_hour = 6 + (int(seed[4:6], 16) % 16)

            options.append({
                "option_id": f"train_opt_{turn_id}_{idx}",
                "operator": operator,
                "train_class": train_class,
                "origin": origin,
                "destination": destination,
                "date": date_from,
                "departure_time": f"{departure_hour:02d}:00",
                "price": price * max(passengers, 1),
                "currency": "EUR",
                "seats_available": 4 + (idx * 3),
                "_tool_source": FAKE_TRAIN_TOOL_NAME,
                # Явная метка, чтобы в трейсах/демо было видно, что это
                # НЕ реальный вызов, в отличие от Kiwi/trivago-ответов.
                "_is_stub": True,
            })

        return options
