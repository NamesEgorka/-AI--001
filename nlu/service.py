"""
NLUService — единственное место в проекте, где реально вызывается LLM для
понимания реплики пользователя (см. api/main.py, docstring модуля: раньше
там было явно написано "этот слой НЕ вызывает LLM сам" — с шагом 6 эта
граница просто переехала сюда, а не размылась).

Вход — сырой текст пользователя. Выход — NLUOutput (nlu_output.py),
провалидированный pydantic-объект, который дальше:
  - либо (clarification_needed=True) превращается в уточняющий вопрос
    пользователю, граф вообще не вызывается;
  - либо его entities идут напрямую в orchestrator/router.py:route()
    (тот уже давно принимает list[ExtractedEntity] — тот же тип, что
    отдаёт NLUExtraction.entities, никакой доп. конвертации не нужно).

LLM подставляется через Dependency Injection (тот же паттерн, что у
Orchestrator с internal_api/kiwi_client/train_client) — тесты
(tests/test_nlu_service.py) подсовывают Fake-реализацию протокола
StructuredLLMClient и НЕ дёргают реальный Anthropic API. Прод —
create_app() создаёт NLUService() без аргументов, что лениво строит
настоящий ChatAnthropic().with_structured_output(NLUExtraction)
(см. api/main.py — конструирование отложено до первого /message,
чтобы отсутствие ANTHROPIC_API_KEY не ломало приложение, если им никто
не пользуется, например в тестах /intent).
"""

from __future__ import annotations

import uuid
from typing import Any, Optional, Protocol

from nlu_output import ExtractedEntity, NLUExtraction, NLUOutput
from orchestrator.router import INTENT_SPECS


class StructuredLLMClient(Protocol):
    """
    Протокол под `ChatAnthropic(...).with_structured_output(NLUExtraction)`
    (langchain Runnable, у которого есть async .ainvoke(messages)).
    Позволяет подставить фейк в тестах, не поднимая реальный LLM-вызов.
    """

    async def ainvoke(self, messages: list[dict[str, str]]) -> NLUExtraction: ...


def _build_slot_glossary() -> str:
    """
    Собирает список допустимых slot_name на каждый intent ПРЯМО из
    orchestrator/router.py:INTENT_SPECS — то есть из источника правды,
    который реально используется роутером для валидации. Если кто-то
    добавит новый intent/слот в router.py и забудет обновить промпт —
    промпт обновится сам, а не рассинхронизируется молча.
    """
    lines = []
    for intent_name, spec in INTENT_SPECS.items():
        slots = ", ".join(spec.slot_map.keys())
        required = ", ".join(spec.required_slots) or "—"
        lines.append(f"  - {intent_name}: слоты [{slots}], обязательные [{required}]")
    return "\n".join(lines)


SYSTEM_PROMPT = f"""Ты — NLU-слой AI-агента для оформления командировок
(перелёты, отели, поезда). Твоя ЕДИНСТВЕННАЯ задача — понять, что хочет
пользователь, и извлечь из его реплики intent и сущности (слоты).

ВАЖНО (архитектурная граница): ты НЕ выполняешь действия, НЕ придумываешь
цены/рейсы/номера заказов и НЕ обращаешься к внешним системам. Только
классификация intent'а и извлечение сущностей из текста.

Допустимые intent'ы и их слоты:
{_build_slot_glossary()}

Кроме них допустимы (пока без обработчика в графе, но валидны как intent):
ExplainPolicy, SmallTalk, OutOfScope.

Правила:
1. Если реплика неоднозначна, не хватает обязательного слота, или ты не
   уверен(а) в intent'е — выставляй clarification_needed=true с честной
   причиной (missing_required_slot / ambiguous_entity /
   conflicting_constraints / low_intent_confidence) и НЕ выдумывай
   недостающие данные.
2. Значения слотов (value) — сырые, БЕЗ нормализации (не переводи города
   в IATA-коды, не досчитывай даты) — это отдельный шаг после тебя.
3. Если значение слота взято не из текущей реплики, а из более раннего
   контекста диалога (анафора: "туда же", "на те же даты") — обязательно
   выставляй source="resolved_from_context" и указывай context_turn_id.
4. Не приписывай пользователю intent, которого нет в его словах, даже
   если он "похож" на предыдущий активный intent.
"""


class NLUService:
    def __init__(
        self,
        llm: Optional[StructuredLLMClient] = None,
        *,
        model: str = "claude-sonnet-5",
    ) -> None:
        self._llm = llm or self._build_default_llm(model)

    @staticmethod
    def _build_default_llm(model: str) -> StructuredLLMClient:
        # Импорт внутри метода, а не на верхнем уровне модуля — чтобы
        # nlu_output.py/router.py можно было использовать (и тестировать)
        # без установленного langchain_anthropic, если LLM в конкретном
        # прогоне вообще не нужен (см. tests/test_nlu_service.py — там
        # всегда передаётся Fake, реальный импорт не срабатывает).
        from langchain_anthropic import ChatAnthropic

        base = ChatAnthropic(model=model)
        return base.with_structured_output(NLUExtraction)  # type: ignore[return-value]

    async def extract(
        self,
        raw_text: str,
        *,
        session_id: str,
        active_intent: Optional[str] = None,
        history: Optional[list[dict[str, str]]] = None,
        turn_id: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> NLUOutput:
        """
        history — предыдущие реплики этого же диалога (role/content),
        нужны модели для anaphora resolution (см. правило 3 в SYSTEM_PROMPT).
        Реальный источник history — checkpointer графа/DialogueState, а не
        сам NLUService — он ничего не хранит между вызовами (stateless,
        как и Orchestrator-клиенты).
        """
        messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        if active_intent:
            messages.append(
                {
                    "role": "system",
                    "content": f"Текущий активный intent в этом диалоге: {active_intent}.",
                }
            )
        messages.extend(history or [])
        messages.append({"role": "user", "content": raw_text})

        extraction: NLUExtraction = await self._llm.ainvoke(messages)

        # raw_text — фактическая правда, известная коду ДО вызова модели;
        # не полагаемся на то, что модель дословно эхом вернула его же
        # (см. NLUExtraction.raw_text docstring в nlu_output.py).
        if extraction.raw_text != raw_text:
            extraction = extraction.model_copy(update={"raw_text": raw_text})

        return NLUOutput.from_extraction(
            extraction,
            turn_id=turn_id or f"t_{uuid.uuid4().hex[:8]}",
            session_id=session_id,
            trace_id=trace_id or f"tr_{uuid.uuid4().hex[:8]}",
        )
