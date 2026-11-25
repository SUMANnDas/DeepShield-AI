# DeepShield AI

DeepShield AI is a local research prototype for image and video media analysis. Its outputs are model-based indicators, not certified forensic conclusions.

## Local setup

From the project root in PowerShell:

    python --version
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install -r requirements-dev.txt
    Copy-Item .env.example .env
    python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000

The active interface uses framework-free HTML, CSS, and JavaScript. In a second PowerShell terminal from the project root, run:

    python -m http.server 5173 --bind 127.0.0.1 --directory frontend/vanilla

Open http://127.0.0.1:5173. API documentation is at http://127.0.0.1:8000/docs. The app reads .env with python-dotenv; environment variables supplied by the process take precedence.

## Health endpoints

- GET /api/v1/health/live reports that the process is running.
- GET /api/v1/health/ready reports that the initial API foundation has loaded.
- /docs, /redoc, and /openapi.json are enabled in development by default and can be disabled with DEEPSHIELD_DOCS_ENABLED=false.

The API also serves local image analysis, bounded video metadata/frame sampling, model-performance and system-health records, persistent analysis history, and JSON/printable reports. Video classification remains unavailable because there is no validated video checkpoint.

## Tests

    .\.venv\Scripts\python.exe -m unittest discover -s backend/tests -v
    npm --prefix frontend test -- --reporter=dot

## Current scope

Local data, .env, model weights, and generated reports are excluded from Git by .gitignore. The active vanilla interface is in `frontend/vanilla/`; the existing React/Vite source is preserved but is not used by this launch command.

## Phase 3: CIFAKE image detection

The image branch now includes manifest-validated CIFAKE v3 loading, scratch-trained ResNet50, EfficientNet-B0, and small Vision Transformer pipelines, validation-based checkpoint selection, isolated one-time test evaluation, Grad-CAM/ViT patch attribution, and 32×32 single-image inference. See [the Phase 3 results and run documentation](docs/models/CIFAKE_IMAGE_DETECTION.md).

These results cover AI-generated versus real CIFAKE images only. They do not establish facial deepfake or video manipulation detection, and CIFAKE's 32×32 resolution limits forensic detail. Resized uploads from other source dimensions receive an inconclusive assessment. External generalization datasets have not been downloaded; see [the dataset research notes](docs/datasets/GENERALIZATION_RESEARCH.md).
