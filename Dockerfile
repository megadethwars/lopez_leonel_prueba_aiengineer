FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    FASTEMBED_CACHE_PATH=/app/.models

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY data ./data
COPY skills ./skills
# Pre-descarga el modelo de embeddings en la imagen (arranque rápido, sin red en runtime).
RUN python -c "from tiendahogar_agent.bootstrap import build_retriever; build_retriever()"

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
CMD ["sh", "-c", "uvicorn tiendahogar_agent.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
