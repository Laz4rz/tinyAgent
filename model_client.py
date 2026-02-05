from __future__ import annotations

import mimetypes
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal


Role = Literal["user", "model", "tool", "system"]


@dataclass
class MessagePart:
    kind: Literal["text", "image"]
    text: str | None = None
    data: bytes | None = None
    mime_type: str | None = None


@dataclass
class Message:
    role: Role
    parts: list[MessagePart] = field(default_factory=list)


class BaseModelClient(ABC):
    def __init__(self, model: str, system_prompt: str | None = None) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.history: list[Message] = []
        self.tools: list[Any] = []

    def set_system_prompt(self, prompt: str | None) -> None:
        self.system_prompt = prompt

    def add_text(self, text: str, *, role: Role = "user") -> None:
        self.history.append(Message(role=role, parts=[MessagePart(kind="text", text=text)]))

    def add_image(self, path: str | Path, *, role: Role = "user", mime_type: str | None = None) -> None:
        file_path = Path(path)
        data = file_path.read_bytes()
        resolved_mime = mime_type or _guess_mime_type(file_path)
        self.history.append(
            Message(
                role=role,
                parts=[MessagePart(kind="image", data=data, mime_type=resolved_mime)],
            )
        )

    def add_tool(self, tool: Any) -> None:
        self.tools.append(tool)

    def set_tools(self, tools: Iterable[Any]) -> None:
        self.tools = list(tools)

    def clear_history(self) -> None:
        self.history.clear()

    def get_history(self) -> list[Message]:
        return list(self.history)

    @abstractmethod
    def generate(self, **kwargs) -> Any:
        raise NotImplementedError


class GeminiClient(BaseModelClient):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "gemini-2.5-flash",
        system_prompt: str | None = None,
        tools: Iterable[Any] | None = None,
        **client_kwargs: Any,
    ) -> None:
        super().__init__(model=model, system_prompt=system_prompt)
        from google import genai

        self._genai = genai
        self._client = genai.Client(api_key=api_key, **client_kwargs)
        if tools:
            self.set_tools(tools)

    def generate(
        self,
        *,
        store_response: bool = True,
        return_response: bool = False,
        **config_kwargs: Any,
    ) -> Any:
        from google.genai import types

        if self.system_prompt is not None and "system_instruction" not in config_kwargs:
            config_kwargs["system_instruction"] = self.system_prompt
        if self.tools and "tools" not in config_kwargs:
            config_kwargs["tools"] = self.tools

        contents = self._to_contents()
        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        if config is None:
            response = self._client.models.generate_content(model=self.model, contents=contents)
        else:
            response = self._client.models.generate_content(
                model=self.model, contents=contents, config=config
            )

        if return_response and not store_response:
            return response

        text = response.text or ""
        if store_response and text:
            self.add_text(text, role="model")

        return response if return_response else text

    def close(self) -> None:
        self._client.close()

    def _to_contents(self) -> list[Any]:
        from google.genai import types

        contents: list[Any] = []
        for message in self.history:
            parts: list[Any] = []
            for part in message.parts:
                if part.kind == "text":
                    parts.append(types.Part.from_text(text=part.text or ""))
                elif part.kind == "image":
                    parts.append(
                        types.Part.from_bytes(
                            data=part.data or b"",
                            mime_type=part.mime_type or "application/octet-stream",
                        )
                    )
            contents.append(types.Content(role=message.role, parts=parts))
        return contents


def _guess_mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"

    guessed, _ = mimetypes.guess_type(str(path))
    if guessed:
        return guessed
    raise ValueError(f"Unsupported image type for {path.name}. Use PNG or JPG.")
