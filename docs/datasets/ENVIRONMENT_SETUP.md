# Dataset Processing Environment

## Approved dependency set

Dataset tooling was installed only into the project virtual environment at `C:\Users\User\Desktop\DeepShield-AI\.venv` with Python 3.12.4 (Windows x64). No global Python, CUDA driver, or system configuration was changed.

Direct packages from `requirements-datasets.txt` and its `requirements.txt` include:

| Package | Installed version | Python 3.12 status | Compatibility notes |
|---|---:|---|---|
| fastapi | 0.141.1 | Compatible | Existing backend dependency. |
| pydantic | 2.13.4 | Compatible | Uses the matching pydantic-core wheel. |
| python-dotenv | 1.2.4 | Compatible | No known conflict with the selected packages. |
| uvicorn | 0.52.4 | Compatible | Existing ASGI server dependency. |
| numpy | 2.5.5 | Compatible | SciPy requires NumPy >=2.0,<2.8; selected version satisfies it. |
| scipy | 1.18.1 | Compatible | Published Python requirement is >=3.12; selected NumPy constraint is satisfied. |
| opencv-python-headless | 4.14.0.94 | Compatible | Use the headless distribution only; other OpenCV wheels provide the same `cv2` module. |
| soundfile | 0.14.0 | Compatible | Supports Python >=3.10; Windows wheel supports the project platform and FLAC decoding. |

Resolved runtime packages present in the environment: `annotated-doc==0.0.7`, `annotated-types==0.8.0`, `anyio==4.16.0`, `cffi==2.1.1`, `click==8.5.1`, `h11==0.16.0`, `idna==3.20`, `pydantic_core==2.46.4`, `pycparser==3.12`, `starlette==1.10.0`, `typing-inspection==0.4.5`, and `typing_extensions==4.16.0`. `requirements-dev.txt` also declares `httpx==0.28.1`, but it is not installed in `.venv`; it was outside the approved dependency set and was not added.

`pip check` reported no broken requirements. Direct package imports succeeded. No incompatible dependency constraints were found. The main packaging caution is to avoid installing another OpenCV distribution alongside `opencv-python-headless`; all expose the same `cv2` import. As with any NumPy 2 environment, add future binary extensions only after checking their NumPy 2 wheel/ABI support.

Compatibility metadata references: [FastAPI](https://pypi.org/project/fastapi/), [Pydantic](https://pypi.org/project/pydantic/), [python-dotenv](https://pypi.org/project/python-dotenv/), [Uvicorn](https://pypi.org/project/uvicorn/), [NumPy](https://pypi.org/project/numpy/), [SciPy](https://pypi.org/project/scipy/), [OpenCV headless](https://pypi.org/project/opencv-python-headless/), and [SoundFile](https://pypi.org/project/soundfile/).

## Verification performed

- Virtual environment Python: 3.12.4.
- `python -m pip check`: passed, no broken requirements.
- Direct imports for FastAPI, Pydantic, dotenv, Uvicorn, NumPy, SciPy, OpenCV, and SoundFile: passed.
- SoundFile FLAC format capability check: passed.
- Uvicorn startup smoke test: passed; `/api/v1/health/live` returned HTTP 200 and `/docs` returned HTTP 200.
- Dataset preprocessing test module: 9 passed using synthetic in-memory/on-disk fixtures, including a regression test for unequal split ratios. These tests do not validate CIFAKE media.
- Full unittest discovery: dataset tests passed; the Phase 1 health-test module could not import because `httpx` is absent. Installing that development-only test dependency was not approved. A separate live Uvicorn smoke check passed.

Dataset acquisition and real-release validation remain pending per-dataset approval.
