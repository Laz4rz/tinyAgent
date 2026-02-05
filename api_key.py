from __future__ import annotations

import os
from pathlib import Path


def load_api_key(
    env_var: str = "GEMINI_API_KEY",
    secret_path: str | Path = ".secret",
    *,
    prompt: bool = True,
) -> str:
    """Load API key from env var, .secret file, or prompt and save."""
    value = os.getenv(env_var)
    if value:
        return value.strip()

    secret_file = Path(secret_path)
    if secret_file.exists():
        stored = secret_file.read_text(encoding="utf-8").strip()
        if stored:
            return stored

    if not prompt:
        return ""

    key = input(f"Enter API key for {env_var}: ").strip()
    if not key:
        raise ValueError("No API key provided.")

    secret_file.write_text(key, encoding="utf-8")
    return key
