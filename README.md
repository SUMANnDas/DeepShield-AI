# DeepShield AI

DeepShield AI is a research prototype for multimodal media forensics. Its outputs are probabilistic assessments, not certified forensic conclusions. The initial backend foundation provides configuration, logging, health checks, and API documentation; it does not yet analyze media.

## Local setup

From the project root in PowerShell:

    python --version
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install -r requirements-dev.txt
    Copy-Item .env.example .env
    python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000

Open http://127.0.0.1:8000/docs for the API schema. The app reads .env with python-dotenv; environment variables supplied by the process take precedence.

## Health endpoints

- GET /api/v1/health/live reports that the process is running.
- GET /api/v1/health/ready reports that the initial API foundation has loaded.
- /docs, /redoc, and /openapi.json are enabled in development by default and can be disabled with DEEPSHIELD_DOCS_ENABLED=false.

Readiness is intentionally shallow at this stage: there is no database, job worker, media storage, or model runtime wired into the service yet.

## Tests

    python -m unittest discover -s backend/tests -v

## Current scope

Phase 1 establishes the backend and development environment only. Dataset acquisition, ML models, upload/analysis endpoints, database integration, reports, and the frontend are later phases. Local data, .env, model weights, and generated reports are excluded from Git by .gitignore.

## Phase 3: CIFAKE image detection

The image branch now includes manifest-validated CIFAKE v3 loading, scratch-trained ResNet50, EfficientNet-B0, and small Vision Transformer pipelines, validation-based checkpoint selection, isolated one-time test evaluation, Grad-CAM/ViT patch attribution, and 32×32 single-image inference. See [the Phase 3 results and run documentation](docs/models/CIFAKE_IMAGE_DETECTION.md).

These results cover AI-generated versus real CIFAKE images only. They do not establish facial deepfake or video manipulation detection, and CIFAKE's 32×32 resolution limits forensic detail.
