import asyncio
import json
import os
import hashlib
import hmac
import tempfile
from pathlib import Path
from threading import Lock
from typing import Any, Literal

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel
from paddleocr import PaddleOCR

MAX_UPLOAD_BYTES = int(os.getenv("IQUASH_OCR_MAX_UPLOAD_BYTES", str(30 * 1024 * 1024)))
API_KEY_SHA256 = os.getenv("IQUASH_OCR_API_KEY_SHA256", "").strip().lower()

app = FastAPI(title="iQuash PaddleOCR Service", version="1.0.0", docs_url=None, redoc_url=None)

_pipeline: PaddleOCR | None = None
_pipeline_init_lock = Lock()
_inference_lock = asyncio.Lock()


class PageResult(BaseModel):
    page_index: int
    text: str
    markdown: str | None = None
    layout_labels: list[str] = []
    raw: dict[str, Any] | None = None


class ParseResponse(BaseModel):
    engine: str
    document_type: Literal["pdf", "image"]
    page_count: int
    pages: list[PageResult]


def require_api_key(authorization: str | None = Header(default=None)) -> None:
    if not API_KEY_SHA256:
        raise HTTPException(status_code=503, detail="OCR service API key hash is not configured")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    digest = hashlib.sha256(authorization[7:].encode()).hexdigest()
    if not hmac.compare_digest(digest, API_KEY_SHA256):
        raise HTTPException(status_code=401, detail="Unauthorized")


def get_pipeline() -> PaddleOCR:
    global _pipeline
    if _pipeline is None:
        with _pipeline_init_lock:
            if _pipeline is None:
                _pipeline = PaddleOCR(
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
    return _pipeline


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    for attr in ("json", "to_dict", "dict"):
        candidate = getattr(value, attr, None)
        try:
            converted = candidate() if callable(candidate) else candidate
            if isinstance(converted, str):
                converted = json.loads(converted)
            if isinstance(converted, dict):
                return converted
        except Exception:
            pass
    try:
        converted = dict(value)
        if isinstance(converted, dict):
            return converted
    except Exception:
        pass
    return {"repr": repr(value)}


def _payload(raw: dict[str, Any]) -> dict[str, Any]:
    nested = raw.get("res")
    return nested if isinstance(nested, dict) else raw


def _page_text(payload: dict[str, Any]) -> str:
    texts = payload.get("rec_texts")
    if isinstance(texts, list):
        cleaned = [str(item).strip() for item in texts if str(item).strip()]
        if cleaned:
            return "\n".join(cleaned)
    overall = payload.get("overall_ocr_res")
    if isinstance(overall, dict):
        texts = overall.get("rec_texts")
        if isinstance(texts, list):
            cleaned = [str(item).strip() for item in texts if str(item).strip()]
            if cleaned:
                return "\n".join(cleaned)
    for key in ("markdown_text", "text", "content"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _page_markdown(payload: dict[str, Any]) -> str | None:
    return None


def _layout_labels(payload: dict[str, Any]) -> list[str]:
    return []


def _run_pipeline(path: str, include_raw: bool) -> list[PageResult]:
    results = get_pipeline().predict(input=path)
    pages: list[PageResult] = []
    for index, result in enumerate(results):
        raw = _as_dict(result)
        payload = _payload(raw)
        page_index = payload.get("page_index")
        try:
            normalized_page_index = int(page_index) if page_index is not None else index
        except (TypeError, ValueError):
            normalized_page_index = index
        pages.append(
            PageResult(
                page_index=normalized_page_index,
                text=_page_text(payload),
                markdown=_page_markdown(payload),
                layout_labels=_layout_labels(payload),
                raw=raw if include_raw else None,
            )
        )
    return pages


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "service": "iquash-paddleocr",
        "engine": "PaddleOCR PP-OCRv6 tiny",
        "model_loaded": _pipeline is not None,
        "auth_configured": bool(API_KEY_SHA256),
    }


@app.post("/v1/parse", response_model=ParseResponse, dependencies=[Depends(require_api_key)])
async def parse_document(file: UploadFile = File(...), include_raw: bool = False) -> ParseResponse:
    mime = (file.content_type or "").lower()
    is_pdf = mime == "application/pdf"
    is_image = mime.startswith("image/")
    if not (is_pdf or is_image):
        raise HTTPException(status_code=415, detail="Only PDF and image files are supported")

    suffix = Path(file.filename or "").suffix.lower() or (".pdf" if is_pdf else ".jpg")
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds OCR upload limit")

    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            temp_path = tmp.name
        async with _inference_lock:
            pages = await asyncio.to_thread(_run_pipeline, temp_path, include_raw)
        if not pages:
            raise HTTPException(status_code=422, detail="No document pages could be parsed")
        return ParseResponse(
            engine="PaddleOCR PP-OCRv6 tiny",
            document_type="pdf" if is_pdf else "image",
            page_count=len(pages),
            pages=pages,
        )
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
