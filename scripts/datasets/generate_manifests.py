"""Create an image, video, or audio manifest and preprocess its media."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("modality", choices=("images", "videos", "audio"))
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    if args.modality == "images":
        from scripts.datasets.image_pipeline import main as run
    elif args.modality == "videos":
        from scripts.datasets.video_pipeline import main as run
    else:
        from scripts.datasets.audio_pipeline import main as run
    run()


if __name__ == "__main__":
    main()
