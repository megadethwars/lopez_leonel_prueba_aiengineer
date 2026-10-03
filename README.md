# TiendaHogar — Agente de soporte al cliente

Agente mínimo de soporte para garantías, devoluciones, envíos y reembolsos, y para consultar el estado de pedidos. Usa RAG sobre los 5 documentos de política, la tool `consultar_estado_pedido` y guardrails que escalan a un humano los casos que el agente no debe resolver.

**Stack:** Python 3.11+ · LangGraph (orquestación) · Claude vía SDK oficial `anthropic` (LLM) · fastembed (embeddings locales multilingües) · FastAPI (microservicio) · pytest.

La arquitectura y las decisiones técnicas están en [SUBMISSION.md](SUBMISSION.md).

## Estructura

```
data/knowledge_base/  # base de conocimiento: los 5 documentos, tal cual (1 archivo .txt por documento)
src/tiendahogar_agent/
  knowledge_base.py   # carga los documentos de data/knowledge_base/
  orders.py           # tool consultar_estado_pedido(order_id: str) -> dict + tabla mock
  retriever.py        # RAG: embeddings + léxico (híbrido), top-k y umbral
  guardrails.py       # reglas de escalamiento (entrada/salida) + tool escalar_a_humano
  llm.py              # cliente Claude (SDK anthropic) detrás de una interfaz inyectable
  graph.py            # grafo LangGraph: guardrail → retrieve → agent ⇄ tools → output guardrail
  api.py              # FastAPI: POST /chat, GET /health, GET / (chat web), /docs (OpenAPI)
  cli.py              # chat en terminal
tests/                # pytest (no requieren API key)
```

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
| `DEMO_API_KEY` | No | — | Si se define, `POST /chat` exige el header `X-API-Key` |

## Correr los tests

```bash
pytest tests/
```

Los tests usan el retriever real (embeddings locales) y un LLM falso guionizado, así que no hacen llamadas a la API de Anthropic. Si no activaste el venv: `python -m pytest tests/`.

## Correr el agente

**Microservicio + chat web:**

```bash
uvicorn tiendahogar_agent.api:app --app-dir src --port 8000
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
PYTHONPATH=src python -m tiendahogar_agent.cli
# Windows PowerShell
$env:PYTHONPATH="src"; python -m tiendahogar_agent.cli
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
