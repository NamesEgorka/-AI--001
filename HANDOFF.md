# HANDOFF — AI-агент определения командировок

Python + LangGraph + FastAPI. Тестовый/портфолио-проект — не подключён к
реальным системам компании, использует честные локальные заглушки там,
где нет доступа к реальным внешним API.

## Статус: MVP готов и покрыт тестами (58/58)

Полный цикл диалога работает end-to-end через HTTP: сырая реплика
пользователя → NLU (Gemini) → граф (LangGraph) → guardrails → результат,
с персистентностью между перезапусками сервера.

---

## Архитектура

```
Пользователь
    │
    ▼
POST /sessions/{id}/message  (сырой текст)
    │
    ▼
NLUService (nlu/service.py) ──── Gemini (gemini-3.6-flash)
    │  intent + entities, либо clarification_needed
    ▼
orchestrator/router.py:route()  — типизация слотов по intent'у
    │
    ▼
orchestrator/graph.py  (LangGraph StateGraph)
    │  search → select_option → check_policy → interrupt() → create_order
    ▼
orchestrator/core.py:Orchestrator  — вся бизнес-логика и guardrails
    │
    ▼
tools/*  — клиенты внешних/внутренних систем (сейчас — заглушки)
```

Персистентность: `langgraph.checkpoint.sqlite.aio.AsyncSqliteSaver`
(файл `sessions.db`) — диалоги переживают перезапуск процесса.
Проверено вживую: остановка `run_server.py`, повторный запуск,
`GET /sessions/{id}/state` возвращает сохранённое состояние.

---

## Что реализовано

### Ядро оркестратора (`orchestrator/core.py`, `orchestrator/guardrails.py`)
- Intent'ы: `SearchFlight`, `SearchHotel`, `SearchTrain`, `SelectOption`,
  `CheckOrderStatus`, `CancelOrder`, `CreateOrder`
- Guardrails: idempotency (повторный `create_order` с тем же ключом не
  дублирует заказ), policy-check перед подтверждением, запрет
  пропускать шаги state machine (нельзя из `idle` сразу в
  `order_confirmed` и т.п.)
- `DialogueState.active_intent` — подсказка активного intent'а,
  выставляется в начале каждого intent-метода и сбрасывается на
  успешных терминальных переходах (`order_confirmed`, `cancelled`,
  `idle` после `CheckOrderStatus`) — **не** сбрасывается на
  `*_failed`, чтобы подсказка осталась актуальной при повторной попытке

### Граф (`orchestrator/graph.py`)
- Точка входа в граф — параметр `intent_entry_node`, а не жёсткий путь
  (переключение произошло на шаге 5)
- `route_after_search`: после показа результатов поиска граф
  **останавливается** (END) и ждёт отдельный ход `SelectOption` — это
  было исправлено (раньше граф пытался сразу выбрать вариант в одном
  вызове с поиском, что при реальном HTTP-использовании приводило к
  ложной ошибке `option_id не указан` на каждый успешный поиск)
- `build_graph(orchestrator, checkpointer=None)` — checkpointer
  передаётся снаружи (по умолчанию `MemorySaver`, для тестов и обычной
  разработки; `AsyncSqliteSaver` — для персистентного запуска)

### FastAPI (`api/main.py`)
- `POST /sessions/{id}/intent` — вход с уже готовым `intent` + `slots`
  (для ручных/скриптовых вызовов, минуя NLU)
- `POST /sessions/{id}/message` — вход с сырым текстом, сам вызывает
  NLU-слой; если `clarification_needed` — граф не трогается, сразу
  возвращается уточняющий вопрос
- `POST /sessions/{id}/confirm` — подтверждение/отказ на `interrupt()`
- `GET /sessions/{id}/state` — снимок состояния сессии
- `TurnResponse.message` — человекочитаемый текст результата поиска.
  **Красиво отформатирован только для `SearchFlight`** (маршрут, даты,
  цены, ID вариантов, подсказка "выбираю <ID>"); для `SearchHotel`/
  `SearchTrain` — безопасный общий текст с количеством вариантов без
  предположений о структуре полей (во избежание падений на
  несовпадении схемы данных между доменами)
- `create_app(orchestrator=None, nlu_service=None, graph=None)` — все
  три зависимости внедряются DI-паттерном, как и везде в проекте

### NLU (`nlu/service.py`)
- `NLUService` — обёртка над `ChatGoogleGenerativeAI(...).with_structured_output(NLUExtraction)`
- **Провайдер — Google Gemini** (`gemini-3.6-flash`), не Anthropic.
  Причина смены: изначально был Anthropic Claude, но проект осознанно
  переведён на Gemini. `provider="anthropic"` оставлен в коде как
  опция, если понадобится сравнить модели
- LLM подставляется через DI — тесты используют `FakeStructuredLLM`,
  реальный вызов API не требуется для прогона тестов
- Промпт (`SYSTEM_PROMPT`) собирает список допустимых intent'ов и
  слотов **прямо из `router.py:INTENT_SPECS`** — не может рассинхронизироваться
  с реальной логикой роутера
- Живьём проверено на реальных репликах через Gemini API — извлечение
  intent/entities работает корректно для полных и неполных запросов

### Данные о рейсах (`tools/flight_api_adapter.py`)
- **`kiwi_client.py` удалён.** Причина: у Kiwi нет открытой регистрации
  для новых пользователей — реальная интеграция технически была почти
  готова (баг с распаковкой кортежа `streamable_http_client` был найден
  и исправлен), но стала недостижима из-за закрытой регистрации
- `FlightApiAdapter` — локальный детерминированный генератор данных
  (`provider_name = "mvp-local"`), без сети и ключей. Цены
  детерминированы по паре origin/destination — одинаковый запрос даёт
  одинаковый результат
- Параметр в `Orchestrator.__init__` называется `flight_client`
  (переименовано из `kiwi_client`, чтобы имя не вводило в заблуждение)

### Персистентность (`run_server.py`)
- `python3 run_server.py` — запуск с `AsyncSqliteSaver` (файл
  `sessions.db`, путь настраивается через `SESSIONS_DB_PATH`)
- Обычный `uvicorn api.main:app --reload` по-прежнему работает и даёт
  in-memory поведение (для быстрой разработки с hot-reload)
- **Ограничение:** `run_server.py` не поддерживает `--reload`
  (программный запуск `uvicorn.Server` этого не даёт) — для разработки
  использовать обычный `uvicorn --reload`, `run_server.py` — только
  для проверки/использования персистентности

---

## Тесты

```bash
PYTHONPATH=. pytest tests/ -v
```
58 тестов, все зелёные: ядро оркестратора, guardrails, роутер, граф
через HTTP (`test_api.py`), NLU-слой (`test_nlu_service.py`,
`FakeStructuredLLM`, без реальных вызовов API).

---

## Как запустить

```bash
pip install -r requirements.txt
```

Ключ Gemini — через GitHub Codespaces Secrets (`GOOGLE_API_KEY`) или
`export GOOGLE_API_KEY="..."` в том же терминале, где будет запускаться
сервер (переменные окружения не передаются между вкладками терминала).

```bash
# Разработка, hot-reload, in-memory (сессии сотрутся при рестарте)
uvicorn api.main:app --reload

# Персистентный запуск (сессии переживают рестарт, sessions.db)
python3 run_server.py
```

**Важно:** порт 8000 в Codespaces по умолчанию публичный — держите его
**Private** во вкладке "Порты" (иначе в логи будет постоянно сыпаться
шум от автоматических сканеров интернета — не баг, но раздражает и
небезопасно для чего угодно ценнее тестового MVP).

---

## Осознанные ограничения (НЕ делаем сейчас — это тестовый проект)

Явное решение: пока используется как тестовый/портфолио-бот, не для
реальной работы, следующее НЕ подключается:

1. **Реальные внутренние API компании** — профиль пользователя,
   тревел-политика, согласование, создание/статус/отмена заказа. Все
   пять — заглушка `InternalApiClient`
2. **Реальный внешний flight-провайдер** — нет доступного поставщика с
   открытой регистрацией на момент разработки; `FlightApiAdapter`
   остаётся локальным
3. **Postgres/Redis** — SQLite выбран как достаточный для этого
   масштаба, не требует поднимать отдельный сервис

---

## Известные технические ограничения (можно закрыть дальше, не блокеры)

- NLU не получает полную историю реплик диалога — только
  `active_intent` (одна строка-подсказка). Anaphora resolution
  ("туда же", "на те же даты") работает хуже, чем могла бы с полной
  историей
- `intent_switch_detected` и `alternative_intents` из `NLUExtraction`
  долетают до `NLUOutput`, но нигде не используются в `/message`
- `message`/`options` формат красив только для `SearchFlight` —
  `SearchHotel`/`SearchTrain` получают общий безопасный, но не
  "красивый" текст
- Chrome автоперевод страницы **обязательно держать выключенным** для
  `github.dev`/`vscode.dev` — ранее уже был случай порчи идентификаторов
  в коде через автоперевод браузера при простом просмотре файла

---

## История ключевых решений (кратко)

1. Ядро оркестратора + guardrails + 4 intent'а с честными
   заглушками/реальными MCP-клиентами, 24 теста
2. FastAPI-обёртка + intent-роутер поверх графа; найден и исправлен
   баг двухходового поиска (`route_after_search`)
3. NLU-слой (`NLUService` + `POST /message`); найден и исправлен баг
   с никогда не устанавливаемым `active_intent`
4. Провайдер LLM: Anthropic Claude → Google Gemini
   (`gemini-3.6-flash`)
5. `kiwi_client.py` удалён (нет регистрации для новых пользователей) →
   `FlightApiAdapter` (локальный MVP-провайдер); добавлен
   человекочитаемый `message`/`options` в ответах API
6. Персистентность: `MemorySaver` → `AsyncSqliteSaver`
   (`run_server.py`), проверено вживую переживание рестарта процесса
