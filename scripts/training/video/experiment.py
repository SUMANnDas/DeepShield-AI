from __future__ import annotations
from datetime import datetime,timezone
from uuid import uuid4
from pathlib import Path
import json
from .storage_budget import VideoStorageBudget

def create_run(root, config, budget=None):
    active_budget = budget or VideoStorageBudget()
    run_root = active_budget.validate_location(Path(root))
    rid = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex[:8]
    path = active_budget.validate_destination(run_root / rid)
    config_path = path / 'config.json'
    payload = json.dumps(config, indent=2) + '\n'
    active_budget.require_capacity(
        len(payload.encode('utf-8')),
        purpose='writing a video experiment configuration',
    )
    path.mkdir(parents=True, exist_ok=False)
    active_budget.validate_destination(config_path)
    with config_path.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(payload)
    return rid, path
