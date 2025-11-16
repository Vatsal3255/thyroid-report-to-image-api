# backend/app/config_store.py
import json
from pathlib import Path
from typing import Optional

CONFIG_PATH = Path(__file__).parent / "config.json"

def save_api_key(key: str) -> None:
    CONFIG_PATH.write_text(json.dumps({"GENAI_API_KEY": key.strip()}))

def load_api_key() -> Optional[str]:
    if CONFIG_PATH.exists():
        data = json.loads(CONFIG_PATH.read_text() or "{}")
        return data.get("GENAI_API_KEY")
    return None
