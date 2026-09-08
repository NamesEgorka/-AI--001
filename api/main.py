"""
FastAPI-обёртка поверх графа (orchestrator/graph.py) и роутера
(orchestrator/router.py). Этот слой НЕ вызывает LLM сам для /intent —
там уже готовый intent+slots от вызывающего кода. /message — исключение
(шаг 6): там сырой текст пользователя, и NLU-слой (nlu/service.py)
вызывается прямо здесь, единственный раз во всём проекте.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from langgraph.types import Command
from pydantic import BaseModel, Field

from nlu.service import NLUService
from nlu_output import ExtractedEntity, IntentName
from orchestrator.core import Orchestrator
from orchestrator.graph import build_graph
from orchestrator.router import MissingRequiredSlotsError, RouterError, UnsupportedIntentError, route
from orchestrator.state import DialogueState, SearchResultSnapshot


class IntentTurnRequest(BaseModel):
    intent: IntentName
    slots: dict[str, str] = Field(default_factory=dict)


class MessageTurnRequest(BaseModel):
    raw_text: str


class ConfirmRequest(BaseModel):
    confirmed: bool


class ClarificationInfo(BaseModel):
    reason: str
    missing_slots: list[str] = Field(default_factory=list)
    detected_intent: Optional[str] = None
    intent_confidence: Optional[float] = None


class TurnResponse(BaseModel):
    session_id: str
    current_state: str
    awaiting_confirmation: bool = False
    confirmation_question: Optional[dict[str, Any]] = None
    message: Optional[str] = None
    options: Optional[list[dict[str, Any]]] = None
    final_result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    clarification: Optional[ClarificationInfo] = None


def _entities_from_slots(slots: dict[str, str]) -> list[ExtractedEntity]:
    return [
        ExtractedEntity(slot_name=name, value=value, raw_span=value, confidence=1.0, source="current_utterance")
        for name, value in slots.items()
    ]


def _format_flight_date(value: str) -> str:
    from datetime import datetime

    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime("%d.%m.%Y")
    except ValueError:
        return value.split("T", 1)[0]


def _format_flight_time(value: str) -> str:
    from datetime import datetime

    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime("%H:%M")
    except ValueError:
        return value.split("T", 1)[-1][:5] if "T" in value else value


def _public_search_options(
    search_result: Optional[SearchResultSnapshot],
) -> Optional[list[dict[str, Any]]]:
    if not search_result:
        return None

    if search_result.intent != "SearchFlight":
        # Формат опций для SearchHotel/SearchTrain пока не приведён к
        # единому виду с рейсами (другие имена полей у отеля/поезда) —
        # чтобы не гадать и не падать на несовпадении схемы, отдаём как
        # есть, без "красивого" переформатирования.
        return search_result.options

    return [
        {
            "option_id": option.get("option_id"),
            "flight_number": option.get("flight_number"),
            "carrier": option.get("carrier"),
            "route": f"{option.get('origin', '')} -> {option.get('destination', '')}",
            "date": _format_flight_date(str(option.get("departure") or "")),
            "departure_time": _format_flight_time(str(option.get("departure") or "")),
            "arrival_time": _format_flight_time(str(option.get("arrival") or "")),
            "price": option.get("price"),
            "currency": option.get("currency", "USD"),
        }
        for option in search_result.options
    ]


def _search_message(
    intent: Optional[str], options: Optional[list[dict[str, Any]]]
) -> Optional[str]:
    if not options:
        return "Подходящих вариантов не найдено."

    count = len(options)
    if count % 10 == 1 and count % 100 != 11:
        label = "вариант"
    elif count % 10 in (2, 3, 4) and count % 100 not in (12, 13, 14):
        label = "варианта"
    else:
        label = "вариантов"

    if intent != "SearchFlight":
        # Общий безопасный формат для отелей/поездов, пока у них нет
        # своего "красивого" форматирования (см. _public_search_options).
        return f"Нашёл {count} {label}. Подробности — в поле options."

    first = options[0]
    lines = [f"{first['route']} | {first['date']} | {count} {label}"]
    for index, option in enumerate(options, start=1):
        price = option.get("price")
        price_str = (
            (str(int(price)) if float(price).is_integer() else str(price))
            if price is not None
            else "?"
        )
        lines.append(
            f"{index}) {option.get('carrier') or '?'} {option.get('flight_number') or '?'} | "
            f"{option.get('departure_time') or '?'} - {option.get('arrival_time') or '?'} | "
            f"{price_str} {option.get('currency', 'USD')} | {option.get('option_id')}"
        )
    lines.append("Выберите рейс: выбираю <ID>")
    return "\n".join(lines)


def create_app(
    orchestrator: Optional[Orchestrator] = None,
    nlu_service: Optional[NLUService] = None,
    graph: Optional[Any] = None,
) -> FastAPI:
    """
    graph — опциональный уже построенный граф (например, с
    AsyncSqliteSaver для персистентности, см. run_server.py). Если не
    передан — строится здесь же с MemorySaver по умолчанию (старое
    поведение, используется тестами и обычным `uvicorn --reload`).
    """
    orch = orchestrator or Orchestrator()
    active_graph = graph or build_graph(orch)

    _nlu_holder: dict[str, NLUService] = {}

    def _get_nlu_service() -> NLUService:
        if "instance" not in _nlu_holder:
            _nlu_holder["instance"] = nlu_service or NLUService()
        return _nlu_holder["instance"]

    app = FastAPI(title="Travel Agent Core API")

    def _thread_config(session_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": session_id}}

    async def _has_pending_interrupt(session_id: str) -> bool:
        snapshot = await active_graph.aget_state(_thread_config(session_id))
        return bool(snapshot.next)

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

        search_result = ds.last_search_result if ds else None
        public_options = _public_search_options(search_result)
        return TurnResponse(
            session_id=session_id,
            current_state=ds.current_state if ds else "unknown",
            awaiting_confirmation=False,
            message=(
                _search_message(search_result.intent if search_result else None, public_options)
                if public_options is not None else None
            ),
            options=public_options,
            final_result=result.get("final_result"),
            error=result.get("error"),
        )

    async def _run_intent_turn(
        session_id: str, intent_name: str, entities: list[ExtractedEntity]
    ) -> TurnResponse:
        if await _has_pending_interrupt(session_id):
            raise HTTPException(
                status_code=409,
                detail=(
                    "Диалог стоит на паузе, ждёт подтверждения через "
                    "POST /sessions/{session_id}/confirm — новый intent "
                    "сейчас принять нельзя."
                ),
            )

        try:
            decision = route(intent_name, entities)
        except MissingRequiredSlotsError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except UnsupportedIntentError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        except RouterError as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        thread_config = _thread_config(session_id)
        graph_input: dict[str, Any] = {
            "intent_entry_node": decision.entry_node,
            "error": None,
            "final_result": None,
            **decision.graph_params,
        }
        existing_state = await active_graph.aget_state(thread_config)
        if not existing_state.values.get("dialogue_state"):
            graph_input["dialogue_state"] = DialogueState(session_id=session_id)

        try:
            result = await active_graph.ainvoke(graph_input, config=thread_config)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Внутренняя ошибка графа: {exc}") from exc

        return _response_from_result(session_id, result)

    @app.post("/sessions/{session_id}/intent", response_model=TurnResponse)
    async def post_intent(session_id: str, body: IntentTurnRequest) -> TurnResponse:
        entities = _entities_from_slots(body.slots)
        return await _run_intent_turn(session_id, body.intent, entities)

    @app.post("/sessions/{session_id}/message", response_model=TurnResponse)
    async def post_message(session_id: str, body: MessageTurnRequest) -> TurnResponse:
        if await _has_pending_interrupt(session_id):
            raise HTTPException(
                status_code=409,
                detail=(
                    "Диалог стоит на паузе, ждёт подтверждения через "
                    "POST /sessions/{session_id}/confirm — новую реплику "
                    "сейчас принять нельзя."
                ),
            )

        thread_config = _thread_config(session_id)
        existing_state = await active_graph.aget_state(thread_config)
        ds: Optional[DialogueState] = existing_state.values.get("dialogue_state")

        try:
            extraction = await _get_nlu_service().extract(
                body.raw_text, session_id=session_id, active_intent=ds.active_intent if ds else None,
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"NLU-слой не смог обработать реплику: {exc}") from exc

        if extraction.clarification_needed:
            return TurnResponse(
                session_id=session_id,
                current_state=ds.current_state if ds else "idle",
                clarification=ClarificationInfo(
                    reason=extraction.clarification_reason or "low_intent_confidence",
                    missing_slots=extraction.missing_required_slots,
                    detected_intent=extraction.intent.name,
                    intent_confidence=extraction.intent.confidence,
                ),
            )

        return await _run_intent_turn(session_id, extraction.intent.name, extraction.entities)

    @app.post("/sessions/{session_id}/confirm", response_model=TurnResponse)
    async def post_confirm(session_id: str, body: ConfirmRequest) -> TurnResponse:
        thread_config = _thread_config(session_id)
        if not await _has_pending_interrupt(session_id):
            raise HTTPException(
                status_code=409,
                detail="Нет диалога, ожидающего подтверждения, для этой сессии.",
            )
        try:
            result = await active_graph.ainvoke(Command(resume=body.confirmed), config=thread_config)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Внутренняя ошибка графа: {exc}") from exc
        return _response_from_result(session_id, result)

    @app.get("/sessions/{session_id}/state")
    async def get_state(session_id: str) -> dict[str, Any]:
        snapshot = await active_graph.aget_state(_thread_config(session_id))
        ds: Optional[DialogueState] = snapshot.values.get("dialogue_state")
        if ds is None:
            raise HTTPException(status_code=404, detail="Сессия не найдена.")
        return {
            "session_id": session_id,
            "current_state": ds.current_state,
            "active_intent": ds.active_intent,
            "last_search_options": ds.last_search_result.options if ds.last_search_result else None,
            "policy_verdict": ds.policy_verdict,
            "order_draft": ds.order_draft,
            "awaiting_confirmation": bool(snapshot.next),
        }

    return app


app = create_app()
