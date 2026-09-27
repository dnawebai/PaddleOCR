# iQuash PaddleOCR Service

Private OCR/document-parsing service for iQuash, built on this PaddleOCR fork.

## Endpoints

- `GET /health` — service health.
- `POST /v1/parse` — authenticated multipart upload of a PDF or image.

Authentication uses `Authorization: Bearer $IQUASH_OCR_API_KEY`.

The response includes page count, per-page OCR text, layout labels and optional raw PaddleOCR output.

## Runtime

The service uses **PP-StructureV3** with document orientation, unwarping, text-line orientation, seal recognition, tables and region detection. Formula/chart recognition are disabled for the legal-intake workload to reduce compute.

Required environment:

- `IQUASH_OCR_API_KEY` — long random secret shared only with the iQuash server.
- `PORT` — supplied by the hosting platform.
- `IQUASH_OCR_MAX_UPLOAD_BYTES` — optional; defaults to 30 MiB.

## Docker

```bash
docker build -f iquash_service/Dockerfile -t iquash-paddleocr .
docker run --rm -p 8080:8080 -e IQUASH_OCR_API_KEY=replace-me iquash-paddleocr
```

The first OCR request can take longer while PaddleOCR models are initialized/downloaded.
