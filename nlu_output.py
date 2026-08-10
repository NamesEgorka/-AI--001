"""
NLUOutput — контракт результата NLU-слоя, передаваемый в Orchestrator.

Использование с LangChain (structured output вместо ручного bind_tools/tool_calls):

    from langchain_anthropic import ChatAnthropic

    llm = ChatAnthropic(model="claude-sonnet-5")
    structured_llm = llm.with_structured_output(NLUOutput)

    result: NLUOutput = structured_llm.invoke(user_message)
    # result уже провалидированный NLUOutput, готовый для Orchestrator
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

# --- Справочники допустимых значений -----------------------------------

IntentName = Literal[
    "SearchFlight",
    "SearchTrain",
    "SearchHotel",
    "SelectOption",
    "RequestApproval",
    "CreateOrder",
    "CheckOrderStatus",
    "CancelOrder",
    "ExplainPolicy",
    "SmallTalk",
    "OutOfScope",
]

EntitySource = Literal["current_utterance", "resolved_from_context"]

ClarificationReason = Literal[
    "missing_required_slot",
    "ambiguous_entity",
    "conflicting_constraints",
    "low_intent_confidence",
]


# --- Вложенные модели -----------------------------------------------------

class IntentPrediction(BaseModel):
    """Основной распознанный intent с confidence."""

    name: IntentName
    confidence: float = Field(..., ge=0, le=1)


class AlternativeIntent(BaseModel):
    """Альтернативный кандидат intent'а (top-k), если основной неоднозначен."""

    name: IntentName
    confidence: float = Field(..., ge=0, le=1)


class ExtractedEntity(BaseModel):
    """
    Одна извлечённая сущность (слот).

    value — сырое значение, БЕЗ валидации через справочники/API.
    Финальная нормализация (IATA-код, точная дата и т.д.) — зона
    Orchestrator + Reference/Geo API, не NLU-слоя.
    """

    slot_name: str = Field(
        ..., description="например origin, destination, date_from, pax_count"
    )
    value: str = Field(
        ..., description="сырое извлечённое значение, без валидации через справочники"
    )
    raw_span: str = Field(
        ..., description="фрагмент исходного текста, из которого извлечено значение"
    )
    confidence: float = Field(..., ge=0, le=1)
    source: EntitySource = Field(
        ...,
        description="явная пометка, если значение взято из предыдущих реплик (anaphora)",
    )
    context_turn_id: Optional[str] = Field(
        default=None,
        description="id реплики-источника, если source = resolved_from_context",
    )

    @model_validator(mode="after")
    def _context_turn_id_required_for_context_source(self) -> "ExtractedEntity":
        if self.source == "resolved_from_context" and self.context_turn_id is None:
            raise ValueError(
                "context_turn_id обязателен, когда source = 'resolved_from_context'"
            )
        return self


class SafetyFlags(BaseModel):
    """Флаги безопасности/скоупа, выставляемые NLU-слоем."""

    out_of_scope: bool = False
    contains_pii: bool = False


# --- Часть, которую реально извлекает LLM -----------------------------------

class NLUExtraction(BaseModel):
    """
    То, что LLM извлекает из текста реплики.

    Намеренно НЕ содержит turn_id/session_id/trace_id — модель не видит эти
    идентификаторы в самой реплике, их знает Orchestrator ещё до вызова LLM.
    Просить модель их "угадывать" — источник случайных рассинхронов id.
    Именно этот класс передаётся в with_structured_output().
    """

    raw_text: str = Field(..., description="Исходная реплика пользователя, без изменений")

    intent: IntentPrediction
    alternative_intents: List[AlternativeIntent] = Field(default_factory=list)
    intent_switch_detected: bool = Field(
        default=False,
        description="true, если distinct от активного intent'а в текущем состоянии диалога",
    )

    entities: List[ExtractedEntity] = Field(default_factory=list)
    missing_required_slots: List[str] = Field(default_factory=list)

    clarification_needed: bool
    clarification_reason: Optional[ClarificationReason] = None

    safety_flags: SafetyFlags = Field(default_factory=SafetyFlags)

    # --- Внутренние guardrail-проверки консистентности самого объекта -----

    @model_validator(mode="after")
    def _clarification_reason_required_when_needed(self) -> "NLUExtraction":
        if self.clarification_needed and self.clarification_reason is None:
            raise ValueError(
                "clarification_reason обязателен, когда clarification_needed = True"
            )
        if not self.clarification_needed and self.clarification_reason is not None:
            raise ValueError(
                "clarification_reason должен быть None, когда clarification_needed = False"
            )
        return self

    @model_validator(mode="after")
    def _missing_slots_imply_clarification(self) -> "NLUExtraction":
        if self.missing_required_slots and not self.clarification_needed:
            raise ValueError(
                "если missing_required_slots не пуст, clarification_needed должен быть True"
            )
        return self


# --- Полный контракт, который уходит в Orchestrator --------------------------

class NLUOutput(NLUExtraction):
    """
    NLUExtraction + инфраструктурные id, подставляемые кодом (не моделью).
    Именно этот объект пишется в логи/трейсинг и передаётся дальше по пайплайну.
    """

    turn_id: str
    session_id: str
    trace_id: str

    @classmethod
    def from_extraction(
        cls, extraction: NLUExtraction, *, turn_id: str, session_id: str, trace_id: str
    ) -> "NLUOutput":
        return cls(
            turn_id=turn_id,
            session_id=session_id,
            trace_id=trace_id,
            **extraction.model_dump(),
        )


# --- Пример использования ---------------------------------------------------

if __name__ == "__main__":
    extraction = NLUExtraction(
        raw_text="а лучше сразу отель рядом забронируй",
        intent=IntentPrediction(name="SearchHotel", confidence=0.88),
        alternative_intents=[AlternativeIntent(name="CreateOrder", confidence=0.31)],
        intent_switch_detected=True,
        entities=[
            ExtractedEntity(
                slot_name="city",
                value="Санкт-Петербург",
                raw_span="рядом",
                confidence=0.7,
                source="resolved_from_context",
                context_turn_id="t_105",
            )
        ],
        missing_required_slots=["check_in", "check_out", "guests"],
        clarification_needed=True,
        clarification_reason="ambiguous_entity",
    )

    # так Orchestrator "дописывает" id поверх того, что реально извлекла модель
    example = NLUOutput.from_extraction(
        extraction, turn_id="t_212", session_id="s_88f2", trace_id="tr_c410"
    )
    print(example.model_dump_json(indent=2, exclude_none=False))
