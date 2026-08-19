"""
FastAPI-обёртка поверх графа (orchestrator/graph.py) — шаг 5 из HANDOFF.md.

Три эндпоинта, ровно по числу разных "форм" хода диалога:

  POST /sessions/{session_id}/intent   — новый intent (первый ход ИЛИ
                                          продолжение уже начатого потока,
                                          например SelectOption после
                                          SearchFlight в том же session_id)
  POST /sessions/{session_id}/confirm  — ответ на interrupt() (да/нет на
                                          "подтвердите оформление заказа")
  GET  /sessions/{session_id}/state    — отладочный снимок состояния диалога

Осознанная граница ответственности (см. AGENTS.md/README про "LLM никогда
не источник фактов"): этот слой НЕ вызывает LLM сам. Тело запроса на
/intent — это уже посчитанный intent + слоты, то есть то, что в проде
отдал бы NLU-слой (см. nlu_output.py, там же пример вызова
ChatAnthropic.with_structured_output(NLUOutput)). Здесь эта граница
проведена явно эндпоинтом: где заканчивается "понять, что хочет
пользователь" и начинается "выполнить это по строгим правилам".
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from langgraph.types import Command
from pydantic import BaseModel, Field

from nlu_output import ExtractedEntity, IntentName
from orchestrator.core import Orchestrator
from orchestrator.graph import build_graph
from orchestrator.router import MissingRequiredSlotsError, RouterError, UnsupportedIntentError, route
from orchestrator.state import DialogueState


# --- Request/response DTO ---------------------------------------------------
#
# Намеренно ЛЕГЧЕ полного NLUOutput (без confidence/raw_span/source на
# каждый слот) — это HTTP-контракт "уже принятое решение", а не сырой
# выход модели. Кто угодно перед этим эндпоинтом (реальный NLU-сервис,
# ручной тест curl'ом, другой оркестратор) должен сначала решить, ЧТО
# это за intent и какие у него слоты — это же остаётся зоной NLU-слоя
# согласно архитектурным принципам (см. HANDOFF.md).

class IntentTurnRequest(BaseModel):
    intent: IntentName
    slots: dict[str, str] = Field(default_factory=dict)


class ConfirmRequest(BaseModel):
    confirmed: bool


class TurnResponse(BaseModel):
    session_id: str
    current_state: str
    awaiting_confirmation: bool = False
    confirmation_question: Optional[dict[str, Any]] = None
    final_result: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class SessionStateResponse(BaseModel):
    session_id: str
    current_state: str
    active_intent: Optional[str] = None
    last_search_options: Optional[list[dict[str, Any]]] = None
    policy_verdict: Optional[dict[str, Any]] = None
    order_draft: Optional[dict[str, Any]] = None
    awaiting_confirmation: bool = False


# --- Приложение --------------------------------------------------------------

def create_app(orchestrator: Optional[Orchestrator] = None) -> FastAPI:
    """
    Фабрика приложения — принимает готовый Orchestrator, чтобы тесты могли
    подставить фейковые клиенты (см. tests/test_api.py), а прод — реальные
    (Kiwi/trivago/FakeTrainClient/InternalApiClient по умолчанию, как и в
    Orchestrator()).
    """
    orch = orchestrator or Orchestrator()
    graph = build_graph(orch)

    app = FastAPI(
        title="Travel Agent Core API",
        description="HTTP-обёртка над LangGraph-агентом поиска командировок.",
    )

    def _thread_config(session_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": session_id}}

    def _entities_from_slots(slots: dict[str, str]) -> list[ExtractedEntity]:
        # confidence=1.0/source="current_utterance" — заглушка НАМЕРЕННО:
        # на этой границе (см. docstring модуля) слоты УЖЕ считаются
        # принятыми, а не гипотезой NLU с confidence < 1. Настоящий
        # confidence/raw_span/anaphora-resolution — забота NLU-слоя ДО
        # этого эндпоинта, не после.
        return [
            ExtractedEntity(
                slot_name=name, value=value, raw_span=value,
                confidence=1.0, source="current_utterance",
            )
            for name, value in slots.items()
        ]

    async def _has_pending_interrupt(session_id: str) -> bool:
        snapshot = await graph.aget_state(_thread_config(session_id))
        return bool(snapshot.tasks and snapshot.tasks[0].interrupts)

    def _response_from_result(session_id: str, result: dict[str, Any]) -> TurnResponse:
        ds: Optional[DialogueState] = result.get("dialogue_state")
        if "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            return TurnResponse(
                session_id=session_id,
                current_state=ds.current_state if ds else "unknown",
                awaiting_confirmation=True,
                confirmation_question=payload,
            )
        return TurnResponse(
            session_id=session_id,
            current_state=ds.current_state if ds else "unknown",
            final_result=result.get("final_result"),
            error=result.get("error"),
        )

    @app.post("/sessions/{session_id}/intent", response_model=TurnResponse)
    async def post_intent(session_id: str, body: IntentTurnRequest) -> TurnResponse:
        if await _has_pending_interrupt(session_id):
            # Явный, а не тихий конфликт: если граф стоит на паузе,
            # ждёт да/нет по уже начатому заказу, новый intent обязан
            # идти через /confirm (или через явную отмену потока — что
            # пока не реализовано, см. HANDOFF.md), а не молча подмешиваться
            # поверх незавершённого interrupt'а.
            raise HTTPException(
                status_code=409,
                detail=(
                    "Диалог стоит на паузе, ждёт подтверждения через "
                    "POST /sessions/{session_id}/confirm — новый intent "
                    "сейчас принять нельзя."
                ),
            )

        entities = _entities_from_slots(body.slots)
        try:
            decision = route(body.intent, entities)
        except MissingRequiredSlotsError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except UnsupportedIntentError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        except RouterError as exc:  # noqa: BLE001 — страховка на будущие подклассы
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        thread_config = _thread_config(session_id)
        graph_input: dict[str, Any] = {
            "intent_entry_node": decision.entry_node,
            # Явный сброс — иначе error/final_result персистентного канала
            # из ПРЕДЫДУЩЕГО хода того же thread_id "протёк" бы в ответ на
            # текущий, никак не относящийся к нему ход.
            "error": None,
            "final_result": None,
            **decision.graph_params,
        }
        # dialogue_state кладём только на самый первый ход этого session_id —
        # дальше он живёт в checkpointer'е графа (см. router.py docstring
        # и ручной прогон в чате перед этим коммитом).
        existing_state = await graph.aget_state(thread_config)
        if not existing_state.values.get("dialogue_state"):
            graph_input["dialogue_state"] = DialogueState(session_id=session_id)

        try:
            result = await graph.ainvoke(graph_input, config=thread_config)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Внутренняя ошибка графа: {exc}") from exc

        return _response_from_result(session_id, result)

    @app.post("/sessions/{session_id}/confirm", response_model=TurnResponse)
    async def post_confirm(session_id: str, body: ConfirmRequest) -> TurnResponse:
        if not await _has_pending_interrupt(session_id):
            raise HTTPException(
                status_code=409,
                detail="Нет ожидающего подтверждения interrupt'а для этой сессии.",
            )
        thread_config = _thread_config(session_id)
        try:
            result = await graph.ainvoke(Command(resume=body.confirmed), config=thread_config)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Внутренняя ошибка графа: {exc}") from exc
        return _response_from_result(session_id, result)

    @app.get("/sessions/{session_id}/state", response_model=SessionStateResponse)
    async def get_state(session_id: str) -> SessionStateResponse:
        snapshot = await graph.aget_state(_thread_config(session_id))
        ds: Optional[DialogueState] = snapshot.values.get("dialogue_state")
        if ds is None:
            raise HTTPException(status_code=404, detail="Сессия не найдена.")
        return SessionStateResponse(
            session_id=session_id,
            current_state=ds.current_state,
            active_intent=ds.active_intent,
            last_search_options=(ds.last_search_result.options if ds.last_search_result else None),
            policy_verdict=ds.policy_verdict,
            order_draft=ds.order_draft,
            awaiting_confirmation=bool(snapshot.tasks and snapshot.tasks[0].interrupts),
        )

    return app


# Точка входа для `uvicorn api.main:app` — реальные клиенты по умолчанию
# (Kiwi/trivago настоящие, InternalApiClient требует env-переменные —
# см. README.md, FakeTrainClient честная заглушка — см. HANDOFF.md).
app = create_app()
