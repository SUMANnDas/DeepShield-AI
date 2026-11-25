# DeepShield AI application guide

## Local development

Use two PowerShell terminals from `C:\Users\User\Desktop\DeepShield-AI`.

Backend terminal:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

Vanilla frontend terminal:

```powershell
python -m http.server 5173 --bind 127.0.0.1 --directory frontend/vanilla
```

Open `http://127.0.0.1:5173`. The frontend sends API requests to `http://127.0.0.1:8000/api/v1`; API docs are at `http://127.0.0.1:8000/docs`. Stop each server with Ctrl+C. The earlier React/Vite frontend remains in the project but is not active in this workflow.

The existing virtual environment contains the project's PyTorch/CUDA stack. The active frontend uses native browser modules and requires no added packages. For a fresh Python environment, install the CUDA-matched PyTorch and Torchvision builds from the official PyTorch selector before installing the project requirements; do not replace the existing `.venv`.

## Application behavior

- Image inference loads one selected CIFAKE v3 checkpoint and its immutable per-run configuration snapshot. It applies EXIF orientation and aspect-preserving fit with training-mean padding for non-native dimensions. A source image other than 32×32 receives an `INCONCLUSIVE` assessment while its raw exploratory score remains available. Probabilities are not calibrated forensic certainty.
- Model metrics come from saved CIFAKE v3 held-out test evaluation records and are displayed as benchmark results only. The application does not rerun test evaluation.
- Video uploads are limited to 200 MB, inspected for metadata, and sampled at up to six frames. The application does not claim a real/fake video prediction because no validated video checkpoint is available.
- Image and video media are not retained. Video uses a unique temporary file that the application removes after processing. SQLite stores filename, SHA-256, metadata, status, and supported model outputs under ignored `storage/app/`.
- Printable reports are served as HTML with a browser print/save-to-PDF action; JSON reports can be downloaded directly.
- The frontend API URL setting is restricted to loopback addresses so media cannot be sent to a remote endpoint.
- API CORS defaults to `http://localhost:5173` and `http://127.0.0.1:5173`. Set `DEEPSHIELD_CORS_ORIGINS` to a comma-separated local-origin allowlist if needed.

## Main API routes

- `GET /api/v1/health/live` and `/api/v1/health/ready`
- `GET /api/v1/system`, `/models`, `/performance`, `/dashboard`
- `POST /api/v1/analysis/images` (multipart `file`, `model_id`)
- `POST /api/v1/analysis/videos` (multipart `file`; metadata and frame previews only)
- `GET /api/v1/history`, `/history/{analysis_id}`
- `GET /api/v1/reports/{analysis_id}.json` and `.html`

Interactive OpenAPI documentation is available at `/docs`.
