FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SETUPTOOLS_SCM_PRETEND_VERSION=3.7.0 \
    PORT=8080

WORKDIR /opt/paddleocr

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml setup.py MANIFEST.in README.md ./
COPY paddleocr ./paddleocr
COPY iquash_service ./iquash_service

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/ && \
    python -m pip install ".[doc-parser]" && \
    python -m pip install -r iquash_service/requirements.txt

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=5 \
  CMD curl --fail http://127.0.0.1:$PORT/health || exit 1

CMD ["sh", "-c", "uvicorn iquash_service.app:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
