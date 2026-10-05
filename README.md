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
    conversations.py       #   ConversationRepositoryPort
  application/             # CASOS DE USO — dependen solo de domain + ports
    support_agent.py       #   grafo LangGraph: guardrail → RAG → agente ⇄ tools → guardrail de salida
    chat_service.py        #   ChatService: turno del agente + historial de conversaciones
    order_status.py        #   consultar_estado_pedido(order_id: str) -> dict
    tools.py               #   ToolRegistry + esquemas de tools (consultar pedido, escalar)
    prompts.py             #   reglas de seguridad + skill → system prompt
  adapters/
    inbound/               # ENTRADA — cómo llegan las peticiones
      http/api.py          #   FastAPI: POST /chat, /conversations, GET /health, GET / (chat web), /docs
      cli.py               #   chat en terminal
    outbound/              # SALIDA — implementaciones de los ports
      llm/                 #   AnthropicLLM (Claude, SDK oficial)
      retrieval/           #   HybridRetriever (denso + léxico) + FastEmbedEmbedder
      knowledge/           #   FileSystemKnowledgeSource (.txt)
      orders/              #   InMemoryOrderRepository (tabla mock)
      skills/              #   FileSystemSkillRepository (SKILL.md)
      notifications/       #   LoggingEscalationNotifier
      conversations/       #   JsonFileConversationRepository (un JSON por conversación)
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
| `ConversationRepositoryPort` | `JsonFileConversationRepository` (`conversations/*.json`) | Cosmos DB / PostgreSQL |

## Requisitos

- Python 3.11 o superior.
- Una API key de Anthropic solo para usar el agente real. Los tests no la necesitan.
- La primera ejecución descarga el modelo de embeddings (unos 220 MB, desde HuggingFace) y lo cachea.

## Instalación

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate      Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt
```

### Configurar la API key (archivo `.env`)

El agente necesita una API key de Anthropic, que se configura en un archivo `.env` en la raíz del proyecto (al mismo nivel que este README).

> **Importante:** el `.env` se crea **a partir de la plantilla [`.env.example`](.env.example)**, que ya incluye todas las variables disponibles con sus valores por defecto.

1. Copia la plantilla como `.env`:

   ```bash
   # Linux/macOS
   cp .env.example .env
   # Windows (PowerShell)
   Copy-Item .env.example .env
   ```

2. Abre el `.env` y escribe tu propia API key en la variable `ANTHROPIC_API_KEY`:

   ```
   ANTHROPIC_API_KEY=sk-ant-tu-api-key-aqui
   ```

- La API key se obtiene en la [Claude Console](https://platform.claude.com), en **Settings → API Keys**.
- El archivo `.env` está en `.gitignore`: **nunca lo subas al repositorio ni pegues la key en el código** (ni en `.env.example`, que sí se versiona).
- Los unit tests (`pytest tests/`) no necesitan la key. Sin ella, el servidor arranca, pero las preguntas que requieren al LLM responden `503 LLM no configurado`.

### Variables de entorno

| Variable | Requerida | Default | Descripción |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Sí, para el agente | — | API key de Anthropic |


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

Forma recomendada (incluye el `.env` y el volumen del historial):

```bash
docker compose up --build
```

El historial de conversaciones **no se empaqueta en la imagen** (son datos de clientes y la imagen es inmutable): `docker-compose.yml` monta la carpeta `conversations/` del proyecto dentro del contenedor, así que sobrevive a reinicios y a reconstrucciones de la imagen.

> ⚠️ Si se ejecuta la imagen **sin volumen** (por ejemplo con el botón *Run* de Docker Desktop sin configurar *Volumes*, o con `docker run` sin `-v`), cada contenedor nuevo empieza con el historial vacío.

Con `docker run` directamente:

```bash
docker build -t tiendahogar-agent .
# Linux/macOS
docker run -p 8000:8000 --env-file .env -v "$(pwd)/conversations:/app/conversations" tiendahogar-agent
# Windows (PowerShell)
docker run -p 8000:8000 --env-file .env -v "${PWD}/conversations:/app/conversations" tiendahogar-agent
```

#sin historial de converaciones

# Windows (PowerShell)
docker run -p 8000:8000 --env-file .env tiendahogar-agent


### Historial de conversaciones

Cada conversación se guarda como `conversations/<id>.json` (pregunta, respuesta, fuentes, tools y escalamiento de cada turno). El chat web muestra el historial en el panel izquierdo: se puede abrir, continuar o eliminar cada conversación, y la URL (`/#<id>`) conserva la conversación activa al recargar la página. Si el servidor se reinicia, al continuar una conversación el agente recupera el contexto desde el historial.

| Endpoint | Descripción |
|---|---|
| `GET /conversations` | Lista de conversaciones (más reciente primero) |
| `GET /conversations/{id}` | Conversación completa con todos sus mensajes |
| `DELETE /conversations/{id}` | Elimina la conversación |

Los JSON contienen datos de clientes: están en `.gitignore` (solo se versiona la carpeta vacía).

## Despliegue de la demo en AWS (Lightsail)

Para mostrar la demo al cliente desde internet con recursos mínimos: **una instancia AWS Lightsail `micro_3_0`** (1 GB RAM, 2 vCPU, 40 GB SSD, ~USD 7/mes cobrados por hora) con Docker Compose, publicada por HTTPS a través de **API Gateway (HTTP API)**, que en una demo cuesta prácticamente nada (~USD 1 por millón de peticiones). El historial de conversaciones vive en el disco de la instancia.

```bash
bash deploy/aws/deploy.sh             # crea la infraestructura y despliega (o actualiza el código si ya existe)
bash deploy/aws/destroy.sh            # elimina todo: API Gateway, instancia, IP estática y key pair (deja de cobrar)
bash deploy/aws/destroy.sh --app-only # solo detiene el contenedor y borra la imagen (la instancia sigue)
```

En Windows se ejecutan desde **Git Bash**, o desde PowerShell con `bash deploy/aws/deploy.sh`.

**Requisitos:** AWS CLI con credenciales configuradas (`aws configure`) y el archivo `.env` con `ANTHROPIC_API_KEY`.

**Qué crea `deploy.sh`:**

| Recurso | Detalle |
|---|---|
| Instancia Lightsail `tiendahogar-demo` | Ubuntu 24.04 · `micro_3_0` · Docker + 2 GB de swap (instalados al arrancar con `user-data.sh`) |
| IP estática `tiendahogar-demo-ip` | Gratis mientras está asociada; `destroy.sh` la libera (sin asociar sí cobra) |
| Key pair `tiendahogar-demo-key` | Llave privada en `~/.ssh/` (fuera del repo y de carpetas sincronizadas) |
| Firewall | Puerto 80 público; SSH (22) solo desde la IP de quien despliega |
| API Gateway `tiendahogar-demo-api` | URL pública `https://<id>.execute-api.<región>.amazonaws.com` con HTTPS. Si la instancia se recrea, `deploy.sh` actualiza el backend y la URL no cambia |

- Genera una **`DEMO_API_KEY`** aleatoria (la imprime al final) para que nadie pueda usar el endpoint, y gastar tokens de Anthropic, sin ella. Se ingresa en el chat web: icono de engrane → "API key de la demo". Se conserva entre despliegues.
- La imagen se construye **en la instancia**, así que no hace falta un registry (ECR).
- `destroy.sh` descarga primero el historial a `conversations-aws-backup-<fecha>/` (ignorado por git).
- Todo es configurable por variables de entorno: `AWS_REGION`, `BUNDLE_ID`, `APP_NAME` (ver `deploy/aws/config.sh`).
- Si la AWS CLI falla con `SSL: CERTIFICATE_VERIFY_FAILED` (redes con proxy o antivirus que inspeccionan TLS), define `AWS_CA_BUNDLE` con un bundle que incluya los certificados raíz de tu sistema.

> Limitaciones de la demo: API Gateway corta las peticiones a los **30 segundos** (las respuestas del agente tardan entre 3 y 10 s). La instancia también responde por HTTP directo en su IP; para algo más que una demo convendría cerrar el puerto 80 a todo lo que no sea API Gateway (o usar un load balancer con certificado) y un dominio propio.

### Cómo entrar a la demo desplegada (qué clave usar)

El chat web pide una clave en **icono de engrane → "API key de la demo"**. Esa clave es la **`DEMO_API_KEY`** que imprime `deploy.sh` al terminar:

```
  Chat web:      https://<id>.execute-api.us-east-1.amazonaws.com/
  DEMO_API_KEY:  <32 caracteres hexadecimales>
```

> ⚠️ **No se debe pegar la API key de Anthropic** (`sk-ant-…`). Son dos claves distintas e independientes: la `DEMO_API_KEY` **no es un sufijo ni una parte** de la key de Anthropic, es una clave aleatoria generada por `deploy.sh`. Si se pega la key de Anthropic, el chat responde *"Se requiere una API key válida"* (HTTP 401).

| Clave | Formato | Para qué sirve | Dónde va |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | `sk-ant-…` (larga) | Que el servidor pueda llamar a Claude | Solo en el archivo `.env`. **Nunca en el navegador** |
| `DEMO_API_KEY` | 32 caracteres hexadecimales (p. ej. `3f9a…c21e`) | Que solo quien la tenga pueda usar la demo (y gastar tokens) | En el chat web (engrane) o en el header `X-API-Key` de la API |

**Si se perdió la `DEMO_API_KEY`:** se puede volver a ver ejecutando de nuevo `bash deploy/aws/deploy.sh` (la conserva y la imprime al final), o consultándola en la instancia:

```bash
ssh -i ~/.ssh/tiendahogar-demo-key.pem ubuntu@<IP> "grep DEMO_API_KEY /home/ubuntu/app/.env"
```

**Pasos para probar en el navegador:**

1. Abrir la URL del chat web.
2. Engrane → pegar la `DEMO_API_KEY` en "API key de la demo" → Enter (se guarda en el navegador; al hacerlo se carga el historial a la izquierda).
3. Escribir una pregunta o hacer clic en una de las tarjetas de ejemplo.

Para la API: abrir `/docs`, pulsar **Authorize**, pegar la misma `DEMO_API_KEY` y usar **Try it out** en `POST /chat`.

## Preguntas de ejemplo

| Pregunta | Comportamiento esperado |
|---|---|
| ¿Cuánto dura la garantía de una licuadora? | RAG → Doc 1 → "6 meses" |
| ¿Dónde está mi pedido ORD-1001? | Tool → En tránsito, 3 días hábiles |
| ¿Y el pedido ORD-9999? | Tool → no encontrado (no inventa) |
| Quiero un reembolso de $800 | Guardrail → escala a soporte@tiendahogar.example |
| Un empleado me trató mal | Guardrail → escala a soporte@tiendahogar.example |
| ¿Venden televisores? | Sin documentos relevantes → "no tengo información" (sin llamar al LLM) |
