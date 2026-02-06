from __future__ import annotations

import json
import os
from pathlib import Path


def load_api_key(
    env_var: str = "GEMINI_API_KEY",
    secret_path: str | Path = ".secret",
    *,
    prompt: bool = True,
) -> str:
    """Load API key from env var, shared .secret file, or prompt and save."""
    value = os.getenv(env_var)
    if value:
        return value.strip()

    secret_file = Path(secret_path)
    if secret_file.exists():
        stored = _read_secret_value(secret_file, env_var)
        if stored:
            return stored

    if not prompt:
        return ""

    key = input(f"Enter API key for {env_var}: ").strip()
    if not key:
        raise ValueError("No API key provided.")

    _write_secret_value(secret_file, env_var, key)
    return key


def _read_secret_value(secret_file: Path, env_var: str) -> str:
    raw = secret_file.read_text(encoding="utf-8").strip()
    if not raw:
        return ""

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        if env_var == "GEMINI_API_KEY":
            return raw
        return ""

    if not isinstance(payload, dict):
        raise ValueError("Secret file must contain a JSON object when using structured keys.")

    value = payload.get(env_var)
    if not isinstance(value, str):
        return ""
    return value.strip()


def _write_secret_value(secret_file: Path, env_var: str, key: str) -> None:
    payload: dict[str, str] = {}

    if secret_file.exists():
        raw = secret_file.read_text(encoding="utf-8").strip()
        if raw:
            try:
                existing = json.loads(raw)
            except json.JSONDecodeError:
                payload["GEMINI_API_KEY"] = raw
            else:
                if not isinstance(existing, dict):
                    raise ValueError("Secret file must contain a JSON object when using structured keys.")
                for name, value in existing.items():
                    if isinstance(name, str) and isinstance(value, str) and value.strip():
                        payload[name] = value.strip()

    payload[env_var] = key
    secret_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
