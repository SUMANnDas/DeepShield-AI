from __future__ import annotations
from datetime import datetime,timezone
from uuid import uuid4
from pathlib import Path
from .utils import write_json_exclusive

def create_run(root,config):
    rid=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid4().hex[:8]; path=Path(root)/rid;path.mkdir(parents=True,exist_ok=False);write_json_exclusive(path/'config.json',config);return rid,path
