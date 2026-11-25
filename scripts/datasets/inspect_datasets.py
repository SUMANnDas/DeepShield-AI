"""Inventory local dataset directory sizes and file extensions without opening media."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path


def inspect(root: Path) -> dict[str, object]:
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(root)
    sizes: Counter[str] = Counter()
    files: Counter[str] = Counter()
    dirs: set[str] = set()
    for current, subdirs, filenames in os.walk(root, followlinks=False):
        subdirs[:] = [name for name in subdirs if not (Path(current) / name).is_symlink()]
        relative_dir = Path(current).relative_to(root).as_posix()
        dirs.add("." if relative_dir == "." else relative_dir)
        for filename in filenames:
            if filename == ".gitkeep":
                continue
            path = Path(current) / filename
            if path.is_symlink():
                continue
            extension = path.suffix.lower() or "[no extension]"
            try:
                sizes[extension] += path.stat().st_size
                files[extension] += 1
            except OSError:
                continue
    return {
        "root": str(root.resolve()),
        "directory_count": len(dirs),
        "file_count": sum(files.values()),
        "total_bytes": sum(sizes.values()),
        "by_extension": {extension: {"files": files[extension], "bytes": sizes[extension]} for extension in sorted(files)},
        "directories": sorted(dirs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, nargs="?", default=Path("datasets/raw"))
    args = parser.parse_args()
    print(json.dumps(inspect(args.path), indent=2))


if __name__ == "__main__":
    main()
