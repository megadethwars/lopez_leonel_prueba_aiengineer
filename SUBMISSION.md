# SUBMISSION — Prueba técnica AI Engineer Senior (TiendaHogar)

## Arquitectura propuesta y justificación

```mermaid
flowchart LR
    U[Cliente<br/>web / curl / CLI] -->|POST /chat| API[FastAPI<br/>auth X-API-Key opcional]
    API --> G

    subgraph G[Grafo LangGraph · estado por session_id]
        IG{{Guardrail de entrada<br/>reglas deterministas}}
        R[Retrieve<br/>híbrido denso + léxico<br/>top-k=3 · umbral 0.25]
        A[Agente<br/>Claude + system prompt]
        T[Tools]
        OG{{Guardrail de salida}}
        E[Escalamiento<br/>respuesta plantilla]
        N[Sin contexto<br/>respuesta plantilla]

        IG -->|reembolso >$500 · queja trato<br/>disputa facturación · legal| E
        IG -->|ok| R
        R -->|0 docs y sin intención de pedido| N
        R -->|docs como contexto| A
        A -->|tool_use| T
        T --> A
        A -->|end_turn| OG
    end

    T --> P[(consultar_estado_pedido<br/>tabla mock)]
    T --> H[escalar_a_humano<br/>soporte@tiendahogar.example]
    R --> KB[(5 documentos<br/>índice en memoria)]
```

**Componentes**

| Componente | Responsabilidad |
|---|---|
| `api.py` (FastAPI) | Contrato HTTP (`/chat`, `/health`, OpenAPI), API key opcional para la demo pública y mapeo de errores del proveedor a 429/502/503. |
| `graph.py` (LangGraph) | Orquestación explícita con nodos y rutas condicionales, y memoria por `session_id` mediante un checkpointer. |
| `guardrails.py` | Tres capas: (1) reglas regex antes del LLM; (2) tool `escalar_a_humano` + system prompt para paráfrasis que las reglas no ven; (3) un control de salida que bloquea cualquier respuesta que "apruebe" un reembolso y garantiza que toda escalación incluya el canal humano. |
| `retriever.py` | Índice vectorial en memoria más score léxico IDF, con top-k y umbral. |
| `orders.py` | La tool con la firma exacta `consultar_estado_pedido(order_id: str) -> dict` y un estado explícito `"no encontrado"`. |
| `llm.py` | Claude vía SDK oficial `anthropic` detrás de la interfaz `LLMClient`. En los tests se inyecta un LLM falso. |

**Por qué este diseño**

- **Lo crítico no depende del LLM.** Los casos que el negocio prohíbe resolver (Doc 4 y Doc 5) se detectan con reglas deterministas antes de llamar al modelo. Son auditables, testeables, sin costo de tokens y sin latencia. El LLM es la segunda red para lo que las reglas no capturan, y el guardrail de salida es la tercera.
- **Anti-alucinación por construcción.** (a) Si ningún documento supera el umbral y la pregunta no trata de un pedido, se responde con una plantilla sin invocar al LLM. (b) Si hay contexto, el system prompt restringe las fuentes de verdad a `<contexto>` y a los resultados de las tools. (c) La tool nunca fabrica datos de un pedido inexistente.
- **LangGraph y no una cadena lineal.** El flujo tiene ramas (escalar, sin contexto, bucle de tools) que conviene que sean explícitas, visibles y testeables por nodo. Además trae checkpointer para multi-turno y se mapea directo a un despliegue productivo.
- **SDK de Anthropic dentro de los nodos, sin capa de abstracción LangChain.** Da control total del request: tools con `strict: true`, `effort`, `fallbacks` server-side y bloques de thinking. El historial es append-only, un requisito de los modelos Claude actuales para conservar el razonamiento entre turnos, y además mantiene válido el prompt cache.
- **Modelo:** `claude-opus-5-5` con `effort: low`, configurable por variable de entorno. Para un chat de soporte con respuestas cortas, un esfuerzo bajo da buena calidad con menos latencia y costo. Se activa `fallbacks: "default"` para que un rechazo del clasificador de seguridad se reintente en otro modelo en lugar de dejar al cliente sin respuesta.

**Trade-offs por el límite de tiempo**

- Índice y memoria de conversación en proceso (`InMemorySaver`): se pierden al reiniciar y no escalan horizontalmente. Es suficiente para el prototipo.
- Guardrail de entrada basado en regex: es preciso y explicable, pero de recall limitado ante paráfrasis. Por eso existe la capa 2.
- No hay streaming ni evaluación automática con LLM-as-judge. La suite de pruebas valida la orquestación con un LLM falso.

## Decisiones técnicas de RAG

- **Chunking:** no se aplicó. Cada documento es un único párrafo de 30 a 60 palabras con un solo tema, así que partirlo solo fragmentaría el contexto (p. ej., separar "Reembolsos mayores a $500…" de su política) sin mejorar la precisión. Se indexa `título + contenido` como una unidad: 1 documento = 1 chunk.
  *Con miles de documentos:* chunking por estructura (secciones y encabezados) con un tamaño objetivo de unos 300 a 500 tokens y un solapamiento de 10 a 15%. Cada chunk llevaría metadatos (doc_id, sección, vigencia, país, tipo de producto) para filtrar antes de la búsqueda. Además: búsqueda híbrida (BM25 + vectores) en un vector store gestionado, un reranker (cross-encoder) sobre los top-20/50, y re-indexación incremental cuando cambia un documento.
- **Embeddings:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` vía `fastembed` (ONNX). Motivos: es **multilingüe** (el corpus y las preguntas están en español), corre **local en CPU** sin API key ni costo (los tests corren offline una vez cacheado), pesa unos 220 MB y genera 384 dimensiones, así que es rápido. Se descartó `multilingual-e5-large` porque pesa unos 2.2 GB y su ganancia de calidad no se justifica para 5 documentos.
  En la calibración, MiniLM puntuaba bajo las consultas cortas ("¿Garantía de una licuadora?" → 0.24), así que el score final es **híbrido**: `0.7 · coseno + 0.3 · léxico`. El componente léxico es la coincidencia de términos del dominio ponderada por IDF, con stemming por prefijo. Así, el top-1 fue correcto en 18 de 18 preguntas de prueba.
- **Threshold de recuperación:** top-k = 3 con umbral de score híbrido de **0.25**, calibrado empíricamente: las preguntas válidas quedaron en ≥ 0.28 y las claramente fuera de dominio ("receta de pastel", "¿venden televisores?", "hola") en ≤ 0.18. El umbral es deliberadamente **orientado a recall**: algunos casos limítrofes que comparten vocabulario ("¿cuánto cuesta una refrigeradora?") sí pasan, y ahí la precisión la garantiza el LLM, instruido para responder solo con el contexto y decir "no tengo información" si la respuesta no está.
  **Si ningún documento supera el umbral:** si la pregunta no menciona un pedido (`ORD-…`, "pedido", "orden") y es el primer turno, se responde con una plantilla fija ("no tengo información sobre eso…" + canal de soporte) **sin llamar al LLM**. Si menciona un pedido, o la conversación ya tiene contexto previo, pasa al agente con `<contexto>(sin documentos relevantes)</contexto>`, para que pueda usar la tool o el historial pero no inventar políticas.

## Pruebas automatizadas

Comando exacto (desde la raíz del repo, con dependencias instaladas):

```bash
pytest tests/
```

48 tests en unos 5 segundos. No requieren API key: usan el retriever real con embeddings locales y un LLM falso guionizado.

| Archivo | Qué cubre |
|---|---|
| `test_retrieval.py` | El top-1 es el documento correcto para preguntas claras de cada política (incluida una de **garantía** que verifica "6 meses") y para consultas cortas. Se respetan top-k y umbral. Las preguntas fuera de dominio no recuperan nada. |
| `test_orders.py` | `ORD-1001` devuelve exactamente su fila; un pedido entregado no tiene fecha; los IDs se normalizan. IDs **inválidos** (`ORD-9999`, vacío, inyección) → `encontrado=False`, `estado="no encontrado"` y **ningún** dato fabricado. |
| `test_guardrails.py` | Se **activa el escalamiento** para reembolsos > $500 (incluye "$1,250.00" y "2 mil"), quejas de trato, disputas de facturación y temas legales. No se activa para preguntas normales, reembolsos ≤ $500 ni IDs de pedido confundidos con montos. El guardrail de salida bloquea "tu reembolso ha sido aprobado". |
| `test_agent_graph.py` | Recorrido end-to-end del grafo: el escalamiento ocurre **sin llamar al LLM**; la tool alimenta al LLM con los datos reales; un pedido inexistente llega como "no encontrado"; las preguntas fuera de dominio no llegan al LLM; el contexto RAG se envía en el prompt; la tool `escalar_a_humano` fuerza el canal humano; el historial es append-only entre turnos. |
| `test_api.py` | Contrato HTTP de `/health` y `/chat`, y exigencia de `X-API-Key` cuando se configura. |

## Cómo mapearías esto a producción

| Prototipo | Stack Grupo Mariposa |
|---|---|
| Agente LangGraph en FastAPI | Agente desplegado en **Microsoft Foundry**: el grafo se empaqueta como agente hospedado (contenedor) o se reexpresa con Foundry Agent Service. Claude se consume desde Foundry (cliente `AnthropicFoundry` del mismo SDK, sin cambios en la lógica). Como los `fallbacks` server-side no existen en Foundry, se usaría el middleware de fallback del SDK. Identidad gestionada (Entra ID) en lugar de API keys. |
| 5 documentos en memoria + fastembed | **Databricks**: los documentos fuente como tablas Delta en **Unity Catalog** (gobierno, linaje y permisos por grupo). Un pipeline de ingesta hace el chunking y los embeddings y alimenta un índice de **Mosaic AI Vector Search** con *Delta Sync* (re-indexación automática), usando búsqueda híbrida y filtros por metadatos. El retriever del agente pasa a ser un cliente de ese índice o una tool expuesta vía Unity Catalog functions. El umbral se re-calibra con un set de evaluación en MLflow. |
| `consultar_estado_pedido` con tabla mock | Llamada al OMS real expuesta y protegida en **Apigee**: OAuth2 o mTLS, cuotas, rate limiting, caching corto y versionado. El agente solo conoce el proxy de Apigee, y el propio `/chat` del agente se publica también detrás de Apigee para los canales (web, app, WhatsApp). |
| Escalamiento como respuesta con email | Además de responder, el nodo de escalamiento publica un evento `support.escalation.created` en **Kafka** (categoría, motivo, `session_id` y transcript con PII enmascarada). Lo consume el sistema de ticketing o CRM, que asigna un agente humano o supervisor (p. ej., aprobación de reembolsos > $500). Otros eventos: `support.conversation.completed` para analítica y evaluación offline, y `kb.document.updated` para disparar la re-indexación. |
| `InMemorySaver` | Checkpointer persistente (Postgres, Cosmos DB o Redis) para conversaciones multi-instancia y auditoría. |
| Logs locales | Trazas OpenTelemetry/MLflow por nodo (retrieval, scores, tools, guardrails), dashboards de tasa de escalamiento y "sin respuesta", y evaluación continua (groundedness y precisión de escalamiento) en Databricks. |
| Guardrails regex | Se mantienen como primera capa determinista y se complementan con Azure AI Content Safety / Prompt Shields para prompt injection y contenido dañino, más un clasificador de intención entrenado con datos reales. |

## Limitaciones conocidas

- **Guardrail por reglas:** no detecta paráfrasis sin palabras clave ("el total no cuadra con lo que pagué"), montos escritos en letras ("ochocientos dólares") ni otros idiomas. Lo mitiga la tool `escalar_a_humano` del LLM, pero esa capa es probabilística. Puede haber falsos positivos (p. ej., "demanda" en otro sentido).
- **Reembolsos > $500 sin monto explícito:** si el cliente pide el reembolso de una refrigeradora sin decir el precio, el agente no conoce el valor (no hay catálogo de precios) y depende del LLM. Supuesto: "remitir al canal humano" para reembolsos > $500 = soporte@tiendahogar.example (Doc 4 pide un supervisor humano pero no define un canal distinto).
- **Umbral calibrado con pocas preguntas** (unas 27, escritas por mí). Con tráfico real se debe recalibrar con un set etiquetado. Las preguntas limítrofes que comparten vocabulario pasan al LLM y dependen de su obediencia al system prompt.
- **Pruebas con LLM falso:** los tests validan la orquestación, no la calidad de las respuestas de Claude. No se incluyó una evaluación automática (LLM-as-judge) de groundedness.
- **Estado en memoria:** conversaciones e índice se pierden al reiniciar, no se comparten entre réplicas y no tienen TTL ni límite de longitud de historial.
- **Sin autenticación de usuario final:** cualquiera con la API key de demo puede consultar cualquier `order_id`. En producción, la tool debe validar que el pedido pertenece al cliente autenticado.
- **Fechas:** la garantía y la ventana de 30 días dependen de la fecha de compra, que el sistema no conoce. El agente solo puede explicar la política.

## Tiempo invertido

Aproximadamente **[COMPLETAR] horas**.
