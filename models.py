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
    "openai": (
        # Chat Completions-oriented IDs from OpenAI docs (checked 2026-02-06).
        "gpt-5.2-chat-latest",
        "gpt-5.2",
        "gpt-5.1-chat-latest",
        "gpt-5.1",
        "gpt-5-chat-latest",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "o3",
        "o4-mini",
        "gpt-4o",
        "gpt-4o-mini",
    ),
}


def known_models(provider: str) -> list[str]:
    model_ids = KNOWN_MODEL_IDS_BY_PROVIDER.get(provider)
    if model_ids is None:
        raise ValueError(f"Unknown provider: {provider}")
    return list(model_ids)


def list_live_models(provider: str, api_key: str) -> list[str]:
    if provider == "google":
        return _list_google_models(api_key)
    if provider == "openai":
        return _list_openai_models(api_key)
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


def _list_openai_models(api_key: str) -> list[str]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    try:
        names = sorted(model.id for model in client.models.list().data)
        if not names:
            raise RuntimeError("No OpenAI models were returned.")
        return names
    finally:
        client.close()
