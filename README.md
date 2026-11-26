# DeepShield AI

DeepShield AI is a local research prototype for image and video media analysis. Its outputs are model-based indicators, not certified forensic conclusions.

## Local setup

The primary interface is the framework-free HTML, CSS, and JavaScript app in `frontend/vanilla/`. FastAPI serves the UI and API from one local origin. The launcher uses the existing project `.venv`; it does not install packages or change the machine execution policy.

From the project root in PowerShell:

    .\start_deepshield.ps1

Open <http://127.0.0.1:8000>. API documentation is at <http://127.0.0.1:8000/docs>. Stop the server with Ctrl+C. The launcher expects `.venv\Scripts\python.exe` to exist and reports a clear error if it does not.

The backend can also be started directly with the environment's Python:

    .\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000

For frontend-only work, serve the vanilla files separately in a second PowerShell terminal (the backend must also be running):

    .\.venv\Scripts\python.exe -m http.server 5173 --bind 127.0.0.1 --directory frontend/vanilla

Then open <http://127.0.0.1:5173>. The default backend CORS allowlist includes this local development origin. The app reads `.env` with python-dotenv; process environment variables take precedence. Copy `.env.example` to `.env` only when local overrides are needed.

For a fresh environment, create a separate `.venv` and install the CUDA-matched PyTorch/Torchvision builds before the remaining pinned project dependencies. Do not replace or modify the existing environment as part of application startup.

## Health endpoints

- GET /api/v1/health/live reports that the process is running.
- GET /api/v1/health/ready reports that the initial API foundation has loaded.
- /docs, /redoc, and /openapi.json are enabled in development by default and can be disabled with DEEPSHIELD_DOCS_ENABLED=false.

The API also serves local image analysis with live Grad-CAM / ViT patch-attribution previews, local video facial manipulation analysis (two validated `RealTimeFaceSwap-10k` checkpoints plus bounded metadata/frame sampling), model-performance and system-health records, persistent analysis history, and JSON/printable reports. Video uploads are limited to 200 MB, 10 minutes, and 16 megapixels per frame; at most six frames are sampled and reduced to 320-pixel previews for inspection alongside the up-to-12-frame face-track classifier.

## Tests

    .\.venv\Scripts\python.exe -m unittest discover -s backend/tests -v
    npm --prefix frontend test -- --reporter=dot

## Current scope

Local data, .env, model weights, and generated reports are excluded from Git by .gitignore. The active vanilla interface is in `frontend/vanilla/`; the existing React/Vite source is preserved but is not used by this launch command.

## Phase 3: CIFAKE image detection

The image branch now includes manifest-validated CIFAKE v3 loading, scratch-trained ResNet50, EfficientNet-B0, and small Vision Transformer pipelines, validation-based checkpoint selection, isolated one-time test evaluation, Grad-CAM/ViT patch attribution, and 32×32 single-image inference. See [the Phase 3 results and run documentation](docs/models/CIFAKE_IMAGE_DETECTION.md).

These results cover AI-generated versus real CIFAKE images only. They do not establish facial deepfake or video manipulation detection, and CIFAKE's 32×32 resolution limits forensic detail. Resized uploads from other source dimensions receive an inconclusive assessment. External generalization datasets have not been downloaded; see [the dataset research notes](docs/datasets/GENERALIZATION_RESEARCH.md).

## Phase 4: Video facial manipulation pipeline

Phase 4 includes bounded video inspection, face tracking/cropping, lossless deterministic crop caching, ImageNet-normalized sequence datasets, EfficientNet-B0 frame/BiGRU/Transformer architectures, budgeted training and grouped evaluation CLIs (`scripts/training/video/train.py`, `scripts/training/video/evaluate.py`), open Kaggle acquisition tooling (`scripts/datasets/rtfs10k.py`), FaceForensics++ c23 preflight/preparation tooling (`scripts/datasets/ffpp_c23.py`), and automatic backend/UI video prediction wiring via `backend/services/video_model_registry.py`.

A double-disjoint 1,200-video subset of `RealTimeFaceSwap-10k` (`CC BY-SA 4.0`, `840` train / `180` validation / `180` test across `original`, `inswapper`, and `uniface`) was acquired under `C:\Users\User\DeepShield-Data\VideoDetection` (~`9.3 GB` total used out of the `100 GB` hard cap) and used to train and evaluate two scratch-initialized models on the RTX 5070 Laptop GPU (`EfficientNet-B0 Frame Baseline`: `80.45%` accuracy, `0.8720` ROC-AUC; `EfficientNet-B0 + BiGRU`: `77.09%` accuracy, `0.8280` ROC-AUC). See [the video model pipeline guide](docs/models/VIDEO_MODEL_PIPELINE.md) and [the video deepfake documentation](docs/models/VIDEO_DEEPFAKE_DETECTION.md).
