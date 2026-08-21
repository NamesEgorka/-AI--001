"""
Юнит-тесты nlu/service.py — везде подставляем FakeStructuredLLM вместо
настоящего ChatAnthropic().with_structured_output(...), поэтому ни один
из этих тестов не делает реальный сетевой вызов и не требует
ANTHROPIC_API_KEY (тот же принцип, что и у FakeKiwiClient/FakeHotelClient
в tests/test_golden_dialogues.py — подмена на границе, а не мокание
внутренностей).
"""

from __future__ import annotations

from typing import Callable, Union

import pytest

from nlu.service import NLUService
from nlu_output import ExtractedEntity, IntentPrediction, NLUExtraction


class FakeStructuredLLM:
    """
    Фейковая реализация StructuredLLMClient. Принимает либо готовый
    NLUExtraction (всегда один и тот же ответ), либо callable, которому
    передаются собранные messages — удобно, когда тесту важно ПРОВЕРИТЬ,
    что именно отправили в LLM (system prompt, active_intent, history).
    """

    def __init__(self, response: Union[NLUExtraction, Callable[[list], NLUExtraction]]):
        self._response = response
        self.received_messages: list[list[dict]] = []

    async def ainvoke(self, messages: list[dict]) -> NLUExtraction:
        self.received_messages.append(messages)
        if callable(self._response):
            return self._response(messages)
        return self._response


def _make_extraction(raw_text: str = "неважно что тут", **overrides) -> NLUExtraction:
    defaults = dict(
        raw_text=raw_text,
        intent=IntentPrediction(name="SearchFlight", confidence=0.95),
        alternative_intents=[],
        intent_switch_detected=False,
        entities=[
            ExtractedEntity(
                slot_name="origin", value="LAX", raw_span="из LAX",
                confidence=0.9, source="current_utterance",
            ),
            ExtractedEntity(
                slot_name="destination", value="JFK", raw_span="в JFK",
                confidence=0.9, source="current_utterance",
            ),
            ExtractedEntity(
                slot_name="date_from", value="2026-08-07", raw_span="7 августа",
                confidence=0.9, source="current_utterance",
            ),
        ],
        missing_required_slots=[],
        clarification_needed=False,
        clarification_reason=None,
    )
    defaults.update(overrides)
    return NLUExtraction(**defaults)


@pytest.mark.asyncio
async def test_extract_returns_valid_nluoutput_with_generated_ids():
    fake = FakeStructuredLLM(_make_extraction())
    service = NLUService(llm=fake)

    result = await service.extract("найди рейс из LAX в JFK на 7 августа", session_id="s_1")

    assert result.session_id == "s_1"
    assert result.turn_id.startswith("t_")
    assert result.trace_id.startswith("tr_")
    assert result.intent.name == "SearchFlight"
    assert len(result.entities) == 3


@pytest.mark.asyncio
async def test_extract_uses_explicitly_provided_ids_instead_of_generating():
    fake = FakeStructuredLLM(_make_extraction())
    service = NLUService(llm=fake)

    result = await service.extract(
        "найди рейс", session_id="s_1", turn_id="t_fixed", trace_id="tr_fixed",
    )

    assert result.turn_id == "t_fixed"
    assert result.trace_id == "tr_fixed"


@pytest.mark.asyncio
async def test_extract_forces_raw_text_to_match_actual_input_not_model_echo():
    """
    NLUExtraction.raw_text ДОЛЖЕН быть в точности тем, что реально прислал
    пользователь — это факт, известный коду ДО вызова LLM, а не то, что
    модель "вспомнила" и повторила. Если модель вернула что-то другое
    (опечатка, перефраз, урезанная копия) — extract() обязан перезаписать
    поле реальным значением.
    """
    fake = FakeStructuredLLM(_make_extraction(raw_text="что-то совсем другое"))
    service = NLUService(llm=fake)

    real_text = "найди рейс из LAX в JFK на 7 августа"
    result = await service.extract(real_text, session_id="s_1")

    assert result.raw_text == real_text


@pytest.mark.asyncio
async def test_extract_passes_active_intent_as_system_hint():
    captured = {}

    def _capture(messages):
        captured["messages"] = messages
        return _make_extraction()

    fake = FakeStructuredLLM(_capture)
    service = NLUService(llm=fake)

    await service.extract(
        "а ещё и отель", session_id="s_1", active_intent="SearchFlight",
    )

    joined = " ".join(m["content"] for m in captured["messages"] if m["role"] == "system")
    assert "SearchFlight" in joined


@pytest.mark.asyncio
async def test_extract_includes_history_messages_for_anaphora():
    captured = {}

    def _capture(messages):
        captured["messages"] = messages
        return _make_extraction()

    fake = FakeStructuredLLM(_capture)
    service = NLUService(llm=fake)

    history = [
        {"role": "user", "content": "найди рейс LAX-JFK на 7 августа"},
        {"role": "assistant", "content": "нашёл 1 вариант"},
    ]
    await service.extract("забронируй ещё и отель на те же даты", session_id="s_1", history=history)

    assert history[0] in captured["messages"]
    assert history[1] in captured["messages"]
    # user-реплика идёт ПОСЛЕДНЕЙ, после всей history
    assert captured["messages"][-1]["content"] == "забронируй ещё и отель на те же даты"


@pytest.mark.asyncio
async def test_extract_propagates_clarification_needed_extraction_unchanged():
    """Честный проброс: если LLM сказала 'нужно уточнение' — extract() не
    должен ничего додумывать или скрывать это поле."""
    extraction = _make_extraction(
        entities=[],
        missing_required_slots=["destination", "date_from"],
        clarification_needed=True,
        clarification_reason="missing_required_slot",
        intent=IntentPrediction(name="SearchFlight", confidence=0.6),
    )
    fake = FakeStructuredLLM(extraction)
    service = NLUService(llm=fake)

    result = await service.extract("хочу куда-нибудь улететь", session_id="s_1")

    assert result.clarification_needed is True
    assert result.clarification_reason == "missing_required_slot"
    assert result.missing_required_slots == ["destination", "date_from"]
