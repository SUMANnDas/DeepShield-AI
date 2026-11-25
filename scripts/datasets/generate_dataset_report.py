"""Generate JSON/CSV/SVG quality reports from a JSON Lines manifest."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.datasets.report_pipeline import generate_dataset_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--media-root", type=Path)
    args = parser.parse_args()
    print(json.dumps(generate_dataset_report(args.manifest, args.output_prefix, args.media_root), indent=2))
