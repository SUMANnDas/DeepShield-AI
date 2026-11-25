"""Create deterministic train/validation/test JSONL files without group leakage."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.datasets.split_pipeline import create_group_splits

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    print(json.dumps(create_group_splits(args.manifest, args.output_prefix, config_path=args.config), indent=2))
