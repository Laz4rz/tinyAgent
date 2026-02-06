from __future__ import annotations

KNOWN_MODEL_IDS_BY_PROVIDER: dict[str, tuple[str, ...]] = {
    "google": (
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    ),
}

DEFAULT_MODEL_BY_PROVIDER: dict[str, str] = {
    "google": "gemini-2.5-flash",
}


def known_models(provider: str) -> list[str]:
    model_ids = KNOWN_MODEL_IDS_BY_PROVIDER.get(provider)
    if model_ids is None:
        raise ValueError(f"Unknown provider: {provider}")
    return list(model_ids)


def default_model(provider: str) -> str:
    model_id = DEFAULT_MODEL_BY_PROVIDER.get(provider)
    if model_id is None:
        raise ValueError(f"Unknown provider: {provider}")
    return model_id


def list_live_models(provider: str, api_key: str) -> list[str]:
    if provider == "google":
        return _list_google_models(api_key)
    raise ValueError(f"Live model listing is not implemented for provider: {provider}")


def _list_google_models(api_key: str) -> list[str]:
    from google import genai

    client = genai.Client(api_key=api_key)
    try:
        names: set[str] = set()
        for model in client.models.list():
            full_name = model.name
            if not full_name.startswith("models/gemini"):
                continue
            supported_actions = model.supported_actions
            if supported_actions and "generateContent" not in supported_actions:
                continue
            names.add(full_name.removeprefix("models/"))

        if not names:
            raise RuntimeError("No Gemini models with generateContent support were returned.")

        return sorted(names)
    finally:
        client.close()
