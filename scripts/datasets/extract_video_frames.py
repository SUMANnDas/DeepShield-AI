"""Extract bounded samples of frames and emit video plus frame manifests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.datasets.video_pipeline import main

if __name__ == "__main__":
    main()
