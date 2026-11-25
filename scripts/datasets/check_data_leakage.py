"""Check split manifests for source, speaker, identity, exact, and perceptual leakage."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.datasets.leakage_pipeline import check_split_leakage

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--perceptual-distance", type=int, default=6)
    args = parser.parse_args()
    print(json.dumps(check_split_leakage(args.manifests, args.report, perceptual_distance=args.perceptual_distance), indent=2))
