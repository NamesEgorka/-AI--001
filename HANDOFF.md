# HANDOFF — передача контекста для продолжения в новом чате

Прикрепите этот файл (и, если нужно, весь `travel_agent_core.zip`) в начале
нового диалога с Claude — этого достаточно, чтобы продолжить без повторных
объяснений.

## Контекст проекта

Вы готовитесь к вакансии AI Agent Engineer (компания разрабатывает
корпоративный AI-агент поиска авиа/жд/отелей поверх существующей OBT —
Online Booking Tool). Мы вместе строим Python + LangGraph реализацию как
практику и как портфолио-артефакт для интервью.

Репозиторий уже выложен на GitHub: `github.com/NamesEgorka/-AI--001`
(работа велась через GitHub Codespaces в браузере).

## Архитектурные принципы (не менять без явного решения)

- LLM никогда не источник фактов — только оркестратор дальше решает вызвать
  инструмент. Все цены/статусы/политика — только из tool result с меткой
  `_tool_source`.
- Guardrails реализованы в коде (`orchestrator/guardrails.py`), не только
  в тексте промпта: anti-hallucination на выбор варианта, source-of-truth
  проверка policy/approval, идемпотентность CreateOrder/CancelOrder.
- Явная таблица разрешённых переходов состояний (`orchestrator/transitions.py`)
  — LLM не может "перепрыгнуть" через policy_check к order_creating.
- `SearchResultSnapshot` — единый формат результата поиска для ЛЮБОГО домена
  (рейсы/отели/жд), поэтому select_option/check_policy/create_order
  переиспользуются без изменений между доменами (ключевая находка шага 3).

## Статус реализации: сделано (8 коммитов, 46/46 тестов)

| Intent | Orchestrator метод | Graph нода | Точка входа через router? | Реальный API |
|---|---|---|---|---|
| SearchFlight | ✅ `search_flights` | ✅ `search_flights` | ✅ | Kiwi.com MCP (`mcp.kiwi.com`, tool `search-flight`, без ключа) |
| SearchHotel | ✅ `search_hotels` | ✅ `search_hotels` | ✅ | trivago MCP (`mcp.trivago.com/mcp`, tool `search_hotels`, без ключа) |
| SearchTrain | ✅ `search_trains` | ✅ `search_trains` | ✅ | честная заглушка `FakeTrainClient` — публичного no-key API с ценой/местами не нашлось (DB — только расписания без цен и с ключом, SNCF — свой ключ, 12306 — не тот рынок) |
| SelectOption | ✅ `select_option` | ✅ `select_option` | ✅ (как ВТОРОЙ ход после SearchX в том же session_id) | anti-hallucination guardrail |
| CheckPolicyCompliance | ✅ `check_policy` | ✅ `check_policy` | — (внутренний шаг потока, не отдельный intent входа) | заглушка (internal API) |
| RequestApproval/CreateOrder | ✅ `create_order` | ✅ `create_order` + `await_user_confirmation` (interrupt) | — (только через `/confirm`, resume уже начатого потока) | заглушка (internal API) |
| CheckOrderStatus | ✅ `check_order_status` | ✅ `check_order_status` | ✅ (самостоятельный короткий путь, сразу в END) | заглушка (internal API) |
| CancelOrder | ✅ `cancel_order` | ✅ `cancel_order` | ✅ (самостоятельный короткий путь, сразу в END) | заглушка (internal API) |
| ExplainPolicy/SmallTalk/OutOfScope | ❌ не начато | ❌ | `router.route()` бросает `UnsupportedIntentError` → HTTP 501 | — |

**Шаг 5 (router + FastAPI) сделан:** `orchestrator/router.py` — маппинг
`intent + сырые слоты` → `(entry_node, типизированные graph_params)`, с
явной таблицей на каждый intent (например, слот `destination` для
`SearchFlight` и `SearchHotel` — РАЗНЫЕ поля `GraphState`, `destination`
vs `hotel_city` — намеренно не смешаны). `orchestrator/graph.py` теперь
ветвится от `START` по `state["intent_entry_node"]`
(`route_from_graph_state`), а не идёт одним жёстким путём.
`api/main.py` — тонкая FastAPI-обёртка (3 эндпоинта: `POST .../intent`,
`POST .../confirm`, `GET .../state`), `thread_id` = `session_id`,
персистентность между ходами — через `MemorySaver` графа (см.
`tests/test_api.py` — доказывает end-to-end HTTP-путь на фейковых
клиентах, включая продолжение SearchFlight → SelectOption → confirm в
ОДНОЙ HTTP-сессии).

## Известные технические особенности (не баги, осознанные решения)

- `tools/internal_api_client.py` — все 6 внутренних методов (профиль,
  политика, approval, заказы) представляют собой честные заглушки с
  `# TODO(internal-api):` — реальных эндпоинтов компании у нас нет и не
  может быть без доступа к их системам.
- `tools/kiwi_client.py` и `tools/trivago_client.py` — код написан по
  официальной документации, но НЕ протестирован вживую в моей (Claude)
  песочнице: сеть там ограничена белым списком доменов. В вашем Codespace
  сеть полная — Kiwi мы уже проверили вживую (работает, см. историю).
  Trivago — ещё не проверяли живьём, только код написан.
- Был найден и исправлен реальный дублирующийся метод `cancel_order` в
  `core.py` (два идентичных определения, тесты не ловили, т.к. оба работали
  одинаково) — хороший пример, что зелёные тесты ≠ отсутствие мёртвого кода.
- Было заражение кода автопереводом браузера Chrome (`graph` → `Графин`) —
  ОБЯЗАТЕЛЬНО держите автоперевод страницы отключённым для github.dev/
  vscode.dev, иначе редактор может незаметно испортить идентификаторы прямо
  во время просмотра файла.
- **Найден и исправлен реальный баг в `route_after_search` (шаг 5).**
  Изначально (унаследовано от старого одного жёсткого пути) эта функция
  сразу вела в ноду `select_option` после успешного поиска — работало
  только потому, что старый `demo_graph_run.py` передавал `option_id`
  заранее одним вызовом. По HTTP это ломалось незаметно: ручной прогон
  через `TestClient` показал, что `POST /intent {SearchFlight}` возвращал
  `200` вместе с полем `error: "option_id не указан"`, потому что граф
  в рамках ОДНОГО `graph.ainvoke()` пытался тут же выбрать вариант,
  которого пользователь физически ещё не мог назвать. Исправлено:
  `route_after_search` теперь всегда `END` после показа результатов —
  `SearchX` и `SelectOption` строго два разных хода диалога, как и было
  задумано в Intent Map. Заодно пришлось перевести `demo_graph_run.py`
  с одного вызова на три (`search → select → confirm`), чтобы демо
  честно отражало реальный HTTP-путь, а не маскировало его.

## Оставшийся план (по порядку, как договаривались "всё по порядку")

1. ~~CheckOrderStatus~~ ✅
2. ~~CancelOrder~~ ✅
3. ~~SearchHotel~~ ✅
4. ~~SearchTrain~~ ✅
5. ~~FastAPI-обёртка + intent-роутер~~ ✅
6. ~~Известные упрощения шага 5 / NLU-слой~~ ✅ — реализовано как
   `nlu/service.py` (`NLUService`, DI-паттерн как у `Orchestrator`,
   `ChatAnthropic(...).with_structured_output(NLUExtraction)`) +
   новый эндпоинт `POST /sessions/{id}/message` (сырой текст →
   `NLUService.extract()` → либо `clarification` в ответе без захода в
   граф, либо `NLUExtraction.entities` напрямую в уже существующий
   `orchestrator/router.py:route()` — конвертация не нужна, тип тот же
   `list[ExtractedEntity]`). `/intent` остался нетронутым для
   ручных/скриптовых вызовов. Общее тело обоих эндпоинтов вынесено в
   `_run_intent_turn()` (см. `api/main.py`).

   Попутно найден и исправлен реальный баг: `DialogueState.active_intent`
   был объявлен в `state.py`, но НИКОГДА не устанавливался ни в одном
   методе `Orchestrator` — подсказка активного intent'а для NLU (нужна
   для anaphora/"туда же", "на те же даты") была бы no-op. Исправлено:
   `active_intent` теперь выставляется в начале `search_flights` /
   `search_hotels` / `search_trains` / `check_order_status` /
   `cancel_order` и сбрасывается в `None` на успешных терминальных
   переходах (`order_confirmed`, `cancelled`, `idle` после
   `CheckOrderStatus`) — но НЕ на `order_failed`/`cancel_failed`,
   намеренно: пользователь может захотеть повторить попытку, и подсказка
   там ещё уместна.

   Тесты: `tests/test_nlu_service.py` (юнит, `FakeStructuredLLM`, без
   реального Anthropic API) + `tests/test_api.py` (`/message`
   end-to-end через тот же `FakeStructuredLLM`, включая проверку, что
   `active_intent` реально доезжает до NLU через HTTP на втором ходу
   одной и той же сессии).

7. **Известные упрощения шага 6, которые стоит закрыть дальше:**
   - `NLUService.extract()` принимает `history` явным списком сообщений,
     но НИКТО пока не собирает эту историю из `DialogueState`/checkpointer'а
     и не передаёт её в `/message` — сейчас anaphora-контекст ограничен
     только `active_intent` (одна строка), полной истории реплик пока нет.
   - Промпт (`nlu/service.py:SYSTEM_PROMPT`) не протестирован на реальных
     ответах `ChatAnthropic` — только структура вызова (через
     `FakeStructuredLLM`). Качество извлечения intent/entities на живых
     репликах нужно проверить вручную с реальным `ANTHROPIC_API_KEY`
     (в моей песочнице ключа нет, сеть на `api.anthropic.com` разрешена,
     но без ключа реальный вызов не сделать).
   - `intent_switch_detected` и `alternative_intents` из `NLUExtraction`
     сейчас никак не используются в `api/main.py` — они долетают до
     `NLUOutput`, но `_run_intent_turn` их просто игнорирует.
   - Нет эндпоинта "явно прервать/отменить ожидающий interrupt" — если
     `POST /confirm` не пришёл, сессия так и висит в
     `approval_pending`/паузе (не баг, а нереализованная часть; сейчас
     `POST .../intent` в этом состоянии просто вернёт 409).
   - `MemorySaver` — состояние диалога живёт только в памяти процесса
     (см. также пункт про Redis/Postgres в README.md) — под нагрузкой/
     рестарт процесса потеряет все сессии.

## Как продолжить technически

Рабочий процесс, который сложился и хорошо работает:
1. Claude пишет код у себя в песочнице, сразу гоняет `pytest` — показывает
   реальный проходящий вывод, а не просто код.
2. Собирает **патч-архив** только с изменёнными файлами (не весь проект
   каждый раз) — экономит время на перенос.
3. Даёт команды `unzip -o patch.zip -d /tmp/patchN` + `cp` — НЕ heredoc
   через терминал (несколько раз обрывался при длинной вставке) и НЕ прямая
   правка в редакторе (риск порчи автопереводом браузера).
4. Пользователь применяет патч в Codespace, гоняет тесты, коммитит и
   пушит на GitHub с содержательным сообщением коммита.

## Рабочее окружение

- GitHub Codespaces, репозиторий `NamesEgorka/-AI--001`, ветка `master`.
- Зависимости: `pydantic`, `httpx`, `mcp`, `pytest`, `pytest-asyncio`,
  `langgraph`, `fastapi`, `uvicorn`, `langchain-anthropic` (все уже в
  `requirements.txt`).
- Для реального (не через FakeStructuredLLM) вызова `/message` нужен
  `ANTHROPIC_API_KEY` в окружении Codespace — без него `NLUService()` по
  умолчанию упадёт при первом обращении к `/message` (но НЕ при старте
  приложения — см. ленивую инициализацию в `api/main.py:create_app`).
- Запуск тестов: `PYTHONPATH=. pytest tests/ -v`
- Запуск демо графа: `PYTHONPATH=. python3 demo_graph_run.py`
