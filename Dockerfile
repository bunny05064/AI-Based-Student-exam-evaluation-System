# Production image: Gunicorn + core deps + Tesseract for OCR.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_MODE=production \
    MODEL_PRELOAD=true \
    FLASK_APP=run.py

WORKDIR /app

# Tesseract system binary + the pytesseract Python binding the app imports.
# The apt package alone only provides the binary: without pytesseract,
# OCR_ENGINE=tesseract raises ImportError at request time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt pytesseract

COPY . .

EXPOSE 8000

# Apply migrations before serving. `create_all()` is skipped in production, so
# without this a fresh Postgres volume has no tables and every DB route 500s.
RUN printf '%s\n' \
    '#!/bin/sh' \
    'set -e' \
    'echo "Applying database migrations..."' \
    'flask db upgrade' \
    'exec "$@"' \
    > /app/docker-entrypoint.sh \
    && chmod +x /app/docker-entrypoint.sh

# Gunicorn: 2 workers (sync) - tweak per available vCPU/RAM.
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120", "run:app"]