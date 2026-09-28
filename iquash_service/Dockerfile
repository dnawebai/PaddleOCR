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

# Keep the heavyweight OCR dependency layer independent from the small iQuash
# API wrapper so service-code edits do not reinstall Paddle/PaddleX.
COPY pyproject.toml setup.py MANIFEST.in README.md ./
COPY paddleocr ./paddleocr
COPY iquash_service/requirements.txt /tmp/iquash-requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/ && \
    python -m pip install "." && \
    python -m pip install -r /tmp/iquash-requirements.txt && \
    python - <<'PY'
from paddleocr import PaddleOCR
PaddleOCR(
    text_detection_model_name="PP-OCRv6_tiny_det",
    text_recognition_model_name="PP-OCRv6_tiny_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    text_det_limit_side_len=1280,
    text_det_limit_type="max",
    text_recognition_batch_size=16,
    engine="paddle",
)
print("iQuash tiny OCR models cached")
PY

COPY iquash_service ./iquash_service

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=5 \
  CMD curl --fail http://127.0.0.1:$PORT/health || exit 1

CMD ["sh", "-c", "uvicorn iquash_service.app:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
