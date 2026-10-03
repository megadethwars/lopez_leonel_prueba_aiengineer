# TiendaHogar — Agente de soporte al cliente

Agente mínimo de soporte para garantías, devoluciones, envíos y reembolsos, y para consultar el estado de pedidos. Usa RAG sobre los 5 documentos de política, la tool `consultar_estado_pedido` y guardrails que escalan a un humano los casos que el agente no debe resolver.

**Stack:** Python 3.11+ · LangGraph (orquestación) · Claude vía SDK oficial `anthropic` (LLM) · fastembed (embeddings locales multilingües) · FastAPI (microservicio) · pytest.

La arquitectura y las decisiones técnicas están en [SUBMISSION.md](SUBMISSION.md).

## Arquitectura (hexagonal / ports & adapters)

El núcleo (dominio + casos de uso) no conoce ni el LLM, ni la base de datos, ni HTTP: define **ports** (interfaces) y la infraestructura se conecta mediante **adapters**. Para integrar otro microservicio, API, base de datos o broker se escribe un adapter nuevo y se conecta en `bootstrap.py`, sin tocar el núcleo. `tests/test_architecture.py` verifica automáticamente estas reglas de dependencia.

```
data/knowledge_base/                 # base de conocimiento: los 5 documentos, tal cual (.txt)
skills/atencion_al_cliente/SKILL.md  # reglas de conversación: tono, empatía, saludos, formato
src/tiendahogar_agent/
  domain/                  # NÚCLEO — reglas de negocio puras, sin frameworks ni I/O
    models.py              #   Document, Order, Escalation, EscalationCategory
    guardrails.py          #   reglas de escalamiento (entrada/salida), montos > $500
  ports/                   # CONTRATOS (typing.Protocol) que el núcleo necesita
    llm.py                 #   LLMPort + errores del proveedor (LLMError…)
    retrieval.py           #   RetrieverPort
    knowledge.py           #   KnowledgeSourcePort
    orders.py              #   OrderRepositoryPort
    skills.py              #   SkillRepositoryPort
    notifications.py       #   EscalationNotifierPort
  application/             # CASOS DE USO — dependen solo de domain + ports
    support_agent.py       #   grafo LangGraph: guardrail → RAG → agente ⇄ tools → guardrail de salida
    order_status.py        #   consultar_estado_pedido(order_id: str) -> dict
    tools.py               #   ToolRegistry + esquemas de tools (consultar pedido, escalar)
    prompts.py             #   reglas de seguridad + skill → system prompt
  adapters/
    inbound/               # ENTRADA — cómo llegan las peticiones
      http/api.py          #   FastAPI: POST /chat, GET /health, GET / (chat web), /docs
      cli.py               #   chat en terminal
    outbound/              # SALIDA — implementaciones de los ports
      llm/                 #   AnthropicLLM (Claude, SDK oficial)
      retrieval/           #   HybridRetriever (denso + léxico) + FastEmbedEmbedder
      knowledge/           #   FileSystemKnowledgeSource (.txt)
      orders/              #   InMemoryOrderRepository (tabla mock)
      skills/              #   FileSystemSkillRepository (SKILL.md)
      notifications/       #   LoggingEscalationNotifier
  bootstrap.py             # COMPOSITION ROOT — conecta cada port con su adapter
  main.py                  # entrypoint ASGI (uvicorn)
  __main__.py              # entrypoint CLI (python -m tiendahogar_agent)
  config.py · logging_config.py
tests/                     # pytest (no requieren API key)
```

| Port | Adapter actual | Adapter en producción (ejemplo) |
|---|---|---|
| `LLMPort` | `AnthropicLLM` | Claude en Microsoft Foundry |
| `KnowledgeSourcePort` | `FileSystemKnowledgeSource` | Tablas Delta en Unity Catalog |
| `RetrieverPort` | `HybridRetriever` (en memoria) | Mosaic AI Vector Search |
| `OrderRepositoryPort` | `InMemoryOrderRepository` | API del OMS detrás de Apigee |
| `SkillRepositoryPort` | `FileSystemSkillRepository` | Servicio de gestión de prompts |
| `EscalationNotifierPort` | `LoggingEscalationNotifier` | Productor Kafka `support.escalation.created` |

## Requisitos

- Python 3.11 o superior.
- Una API key de Anthropic solo para usar el agente real. Los tests no la necesitan.
- La primera ejecución descarga el modelo de embeddings (unos 220 MB, desde HuggingFace) y lo cachea.

## Instalación

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate      Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # y completa ANTHROPIC_API_KEY
```

### Variables de entorno

| Variable | Requerida | Default | Descripción |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Sí, para el agente | — | API key de Anthropic |
| `ANTHROPIC_MODEL` | No | `claude-opus-5-5` | Modelo de Claude |
| `ANTHROPIC_EFFORT` | No | `low` | Esfuerzo de razonamiento (`low`…`max`) |
| `RETRIEVAL_TOP_K` | No | `3` | Máximo de documentos por pregunta |
| `RETRIEVAL_THRESHOLD` | No | `0.25` | Score híbrido mínimo para usar un documento |
| `RETRIEVAL_ALPHA` | No | `0.7` | Peso del componente denso en el score híbrido |
| `KNOWLEDGE_BASE_DIR` | No | `data/knowledge_base` | Carpeta con los documentos `.txt` |
| `SKILLS_DIR` | No | `skills` | Carpeta de skills de conversación |
| `CONVERSATION_SKILL` | No | `atencion_al_cliente` | Skill que se carga en el system prompt |
| `LOG_LEVEL` | No | `INFO` | `DEBUG` agrega los scores de todos los documentos en cada búsqueda |
| `LOG_FORMAT` | No | `text` | `json`: una línea JSON por evento (para Azure Monitor, Datadog, ELK) |
| `JUDGE_MODEL` | No | `claude-sonnet-5-5` | Modelo del juez en las pruebas de integración |
| `DEMO_API_KEY` | No | — | Si se define, `POST /chat` exige el header `X-API-Key` |

## Correr los tests

```bash
pytest tests/
```

Los tests usan el retriever real (embeddings locales) y adapters falsos para el LLM y el notificador, así que no hacen llamadas a la API de Anthropic. Si no activaste el venv: `python -m pytest tests/`.

### Pruebas de integración end-to-end (Claude real + juez LLM)

```bash
pytest -m integration
```

Requieren `ANTHROPIC_API_KEY` y **consumen tokens** (unos USD 0.50–0.70 por corrida completa, estimado). Sin la key se omiten. `pytest tests/` no las ejecuta: están excluidas por defecto.

- **`test_e2e_agent.py`:** 19 conversaciones reales por HTTP (`/chat` → LangGraph → Claude → RAG/tools). Cada caso pasa dos filtros: verificaciones deterministas sobre el JSON (¿escaló?, ¿llamó la tool?, ¿usó el documento correcto?) y un **juez LLM** (`claude-sonnet-5-5`, distinto del agente) que califica una rúbrica criterio por criterio, siempre con la fidelidad a la base de conocimiento.
- **`test_judge_calibration.py`:** verifica que el juez repruebe respuestas malas conocidas (vacía, dato incorrecto, beneficio inventado, reembolso aprobado, intento de manipular al juez) y apruebe una correcta.
- Al terminar se genera `reports/llm_eval_report.md` (y `.json`) con el veredicto y la justificación de cada criterio.
- El modelo del juez se cambia con `JUDGE_MODEL`.


## Correr el agente

**Microservicio + chat web:**

RECORDATORIO: Instalar un ambiente virtual primero (recomendado)

```bash
uvicorn tiendahogar_agent.main:app --app-dir src --port 8000
```

- Chat web: http://localhost:8000/
- OpenAPI / Swagger: http://localhost:8000/docs

```bash
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"message": "¿Dónde está mi pedido ORD-1001?"}'
```

Respuesta (resumida):

```json
{
  "session_id": "…",               // reenvíalo para mantener la conversación
  "answer": "Tu Refrigeradora está En tránsito…",
  "escalated": false,
  "escalation": null,              // {category, reason, source} si se escaló
  "sources": [{"doc_id": "doc3", "title": "Tiempos de envío", "score": 0.41}],
  "tool_calls": [{"tool": "consultar_estado_pedido", "input": {"order_id": "ORD-1001"}, "output": {…}}]
}
```

**Terminal:**

```bash
# Linux/macOS
PYTHONPATH=src python -m tiendahogar_agent
# Windows PowerShell
$env:PYTHONPATH="src"; python -m tiendahogar_agent
```

**Docker:**

```bash
docker build -t tiendahogar-agent .
docker run -p 8000:8000 --env-file .env tiendahogar-agent
```

## Preguntas de ejemplo

| Pregunta | Comportamiento esperado |
|---|---|
| ¿Cuánto dura la garantía de una licuadora? | RAG → Doc 1 → "6 meses" |
| ¿Dónde está mi pedido ORD-1001? | Tool → En tránsito, 3 días hábiles |
| ¿Y el pedido ORD-9999? | Tool → no encontrado (no inventa) |
| Quiero un reembolso de $800 | Guardrail → escala a soporte@tiendahogar.example |
| Un empleado me trató mal | Guardrail → escala a soporte@tiendahogar.example |
| ¿Venden televisores? | Sin documentos relevantes → "no tengo información" (sin llamar al LLM) |
