# Travel Agent Core — Orchestrator

Кодовое ядро для travel-агента, вынесенное из LangSmith Fleet-прототипа.
Реализует то, что раньше существовало только как текст в `AGENTS.md`:
жёсткие guardrails, детерминированную state machine, идемпотентность и
трейсинг — всё в виде исполняемого и протестированного кода.

## Что реально готово и протестировано (17/17 тестов проходят)

- **State machine** (`orchestrator/state.py`, `orchestrator/transitions.py`) —
  явная таблица разрешённых переходов между состояниями диалога. Прыжки
  вида "пропустить policy_check и сразу создать заказ" технически
  невозможны — блокируются до вызова любого инструмента.
- **Guardrails как код** (`orchestrator/guardrails.py`):
  - anti-hallucination на выбор варианта (`SelectOption` возможен только
    для `option_id`, реально присутствующего в последнем результате поиска,
    с проверкой TTL);
  - source-of-truth валидация (вердикт по политике/approval обязан быть
    помечен как пришедший от конкретного tool'а, а не сгенерирован LLM);
  - идемпотентность `CreateOrder`/`CancelOrder` — защита от двойного
    заказа при retry на сетевом сбое.
- **Трейсинг** (`orchestrator/tracing.py`) — структурированные JSON-логи
  каждого tool call с `trace_id`/`turn_id`/latency/статусом.
- **Orchestrator** (`orchestrator/core.py`) — связывает всё выше в рабочий
  путь: `search_flights → select_option → check_policy → create_order`.
- **Реальный клиент поиска рейсов** (`tools/kiwi_client.py`) — публичный
  MCP-сервер Kiwi.com, ключ не нужен.
- **Тесты**:
  - `tests/test_guardrails.py` — adversarial-тесты (симуляция галлюцинаций,
    попыток обойти проверки, повторных запросов);
  - `tests/test_golden_dialogues.py` — сценарные тесты полного пути на
    замоканных клиентах, без реальной сети.

## Что НЕ готово — и почему я не мог это сделать

**Внутренние API компании** (`tools/internal_api_client.py`) — интерфейс
и вся инфраструктура (retries, тайминг, трейсинг, обязательная метка
`_tool_source`) готовы, но тело метода `_http_call` — заглушка. Я не
знаю реальных эндпоинтов, схем данных и способа авторизации ваших систем
профиля, тревел-политики, approval и заказов — это физически невозможно
написать без доступа к вашей внутренней документации/бэкенду.

Каждое место, требующее правки, помечено `# TODO(internal-api):` в коде.
По моей оценке — это правки уровня "подставить реальный URL и путь",
не переписывание архитектуры.

**Не протестировано вживую**: `tools/kiwi_client.py` написан по
официальной документации Kiwi (Streamable HTTP MCP, инструмент
`search-flight`), но в текущей песочнице сеть ограничена белым списком
доменов, `mcp.kiwi.com` туда не входит — реальный вызов ни разу не
выполнялся. Структура ответа (`_parse_options`) может потребовать
подгонки под фактический формат — сверьте при первом реальном запуске.

## Что осталось за рамками этой сборки (не блокер для первого релиза)

- Мониторинг метрик по intent (успех/fallback/эскалация) — трейсинг есть,
  дашборд поверх него — нет.
- Полный LangGraph-граф со всеми нодами из Intent Map (сейчас реализован
  путь SearchFlight → SelectOption → CheckPolicyCompliance → CreateOrder;
  SearchTrain/SearchHotel/CancelOrder/CheckOrderStatus — по аналогии,
  но не написаны явно).
- PII-обработка (паспортные/платёжные данные) — в тестах и коде такие
  поля нигде не появляются намеренно; когда дойдёте до реальных заказов,
  нужно отдельно продумать, что логируется, а что нет.
- Redis/Postgres вместо in-memory `IdempotencyStore` — интерфейс уже
  рассчитан на замену (см. `TODO(prod)` в `guardrails.py`).

## Запуск

```bash
pip install -r requirements.txt
export INTERNAL_TRAVEL_API_BASE_URL="https://your-internal-mcp-gateway"
export INTERNAL_TRAVEL_API_TOKEN="..."
PYTHONPATH=. pytest tests/ -v
```

## Дальнейшие шаги (по приоритету)

1. Бэкенд-команда закрывает `TODO(internal-api)` в `internal_api_client.py`
   для всех 6 внутренних методов.
2. Живой тест `kiwi_client.py` против реального `mcp.kiwi.com` в вашем
   окружении (не ограниченном белым списком доменов).
3. Достроить LangGraph-обёртку поверх `Orchestrator` для остальных intent'ов
   (SearchTrain, SearchHotel, CancelOrder, CheckOrderStatus) по аналогии
   с уже реализованным путём.
4. Заменить `IdempotencyStore` на Redis-версию перед реальной нагрузкой.
