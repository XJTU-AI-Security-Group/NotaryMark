import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def _to_jsonable(value: Any):
    """Recursively convert values into JSON-serializable primitives."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def save_run_config(log_dir: str | Path, script_name: str, params: dict) -> str:
    """Persist a timestamped run configuration snapshot for reproducibility."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "script": script_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "argv": sys.argv,
        "params": _to_jsonable(params),
    }

    timestamped_path = log_dir / f"{script_name}_run_{timestamp}.json"
    latest_path = log_dir / f"{script_name}_latest.json"

    for path in (timestamped_path, latest_path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    return str(timestamped_path)
