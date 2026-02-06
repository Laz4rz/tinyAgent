from __future__ import annotations

import io
import mimetypes
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Literal

from tool_schema import make_tool_schema


Role = Literal["user", "model", "tool", "system"]
ImageFormat = Literal["jpeg", "png"]

_ROLE_COLORS: dict[Role, str] = {
    "user": "\033[38;5;39m",
    "model": "\033[38;5;82m",
    "tool": "\033[38;5;214m",
    "system": "\033[38;5;244m",
}
_ANSI_RESET = "\033[0m"


@dataclass
class ImageSettings:
    scale: float = 0.25
    format: ImageFormat | None = None
    quality: int = 85


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


@dataclass
class ConversationHistory:
    messages: list[Message] = field(default_factory=list)

    def __iter__(self) -> Iterator[Message]:
        return iter(self.messages)

    def __len__(self) -> int:
        return len(self.messages)

    def __getitem__(self, index: int) -> Message:
        return self.messages[index]

    def append(self, message: Message) -> None:
        self.messages.append(message)

    def add_text(self, text: str, *, role: Role = "user") -> None:
        self.append(Message(role=role, parts=[MessagePart(kind="text", text=text)]))

    def add_image(self, data: bytes, mime_type: str, *, role: Role = "user") -> None:
        self.append(Message(role=role, parts=[MessagePart(kind="image", data=data, mime_type=mime_type)]))

    def clear(self) -> None:
        self.messages.clear()

    def copy(self) -> list[Message]:
        return list(self.messages)

    def render(self, *, color: bool = True) -> str:
        if not self.messages:
            return "(history is empty)"

        lines: list[str] = []
        for message_index, message in enumerate(self.messages, start=1):
            rendered_role = _render_role(message.role, color=color)
            lines.append(f"{message_index}. role: {rendered_role}")

            for part_index, part in enumerate(message.parts, start=1):
                if part.kind == "text":
                    if part.text is None:
                        raise ValueError("Text message part is missing text.")
                    lines.extend(_render_text_part(part_index, part.text))
                    continue

                if part.kind == "image":
                    if part.mime_type is None:
                        raise ValueError("Image message part is missing mime_type.")
                    lines.append(f"   - input {part_index} (image)")
                    lines.append(f"     type: {part.mime_type}")
                    lines.append(f"     compression: {_compression_label(part.mime_type)}")
                    continue

                raise ValueError(f"Unsupported message part kind: {part.kind}")

        return "\n".join(lines)


class BaseModelClient(ABC):
    def __init__(self, model: str, system_prompt: str | None = None) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.history = ConversationHistory()
        self.tools: list[Any] = []

    def set_system_prompt(self, prompt: str | None) -> None:
        self.system_prompt = prompt

    def add_text(self, text: str, *, role: Role = "user") -> None:
        self.history.add_text(text, role=role)

    def add_image(
        self,
        path: str | Path,
        *,
        role: Role = "user",
        mime_type: str | None = None,
        settings: ImageSettings | None = None,
    ) -> None:
        file_path = Path(path)
        resolved_mime = mime_type or _guess_mime_type(file_path)
        data = file_path.read_bytes()
        resolved_settings = settings or ImageSettings()
        data, resolved_mime = _preprocess_image(
            data,
            mime_type=resolved_mime,
            settings=resolved_settings,
        )

        data, resolved_mime = self._prepare_image_for_provider(
            data,
            mime_type=resolved_mime,
            settings=resolved_settings,
        )
        self.history.add_image(data, resolved_mime, role=role)

    def add_tool(self, tool: Any) -> None:
        self.tools.append(tool)

    def set_tools(self, tools: Iterable[Any]) -> None:
        self.tools = list(tools)

    def set_tool_functions(self, tool_functions: Iterable[Callable[..., Any]]) -> None:
        self.set_tools(self._build_provider_tools(tool_functions))

    def clear_history(self) -> None:
        self.history.clear()

    def get_history(self) -> list[Message]:
        return self.history.copy()

    def format_history(self, *, color: bool = True) -> str:
        return self.history.render(color=color)

    def generate(
        self,
        *,
        store_response: bool = True,
        return_response: bool = False,
        **config_kwargs: Any,
    ) -> Any:
        request_config = self._build_request_config(config_kwargs)
        encoded_history = self._encode_history()
        response = self._call_model(encoded_history=encoded_history, request_config=request_config)

        if return_response and not store_response:
            return response

        text = self._extract_text(response)
        if store_response and text:
            self.add_text(text, role="model")

        return response if return_response else text

    def _prepare_image_for_provider(
        self,
        data: bytes,
        *,
        mime_type: str,
        settings: ImageSettings | None,
    ) -> tuple[bytes, str]:
        return data, mime_type

    @abstractmethod
    def _build_request_config(self, config_kwargs: dict[str, Any]) -> Any:
        raise NotImplementedError

    @abstractmethod
    def _encode_history(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def _call_model(self, *, encoded_history: Any, request_config: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def _extract_text(self, response: Any) -> str:
        raise NotImplementedError

    def _build_provider_tools(self, tool_functions: Iterable[Callable[..., Any]]) -> list[Any]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement callable tool conversion."
        )


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

        self._client = genai.Client(api_key=api_key, **client_kwargs)
        if tools:
            self.set_tools(tools)

    def close(self) -> None:
        self._client.close()

    def _build_request_config(self, config_kwargs: dict[str, Any]) -> Any:
        from google.genai import types

        merged = dict(config_kwargs)
        if self.system_prompt is not None and "system_instruction" not in merged:
            merged["system_instruction"] = self.system_prompt
        if self.tools and "tools" not in merged:
            merged["tools"] = self.tools

        if not merged:
            return None
        return types.GenerateContentConfig(**merged)

    def _encode_history(self) -> list[Any]:
        from google.genai import types

        contents: list[Any] = []
        for message in self.history:
            parts: list[Any] = []
            for part in message.parts:
                if part.kind == "text":
                    if part.text is None:
                        raise ValueError("Text message part is missing text.")
                    parts.append(types.Part.from_text(text=part.text))
                    continue
                if part.kind == "image":
                    if part.data is None or part.mime_type is None:
                        raise ValueError("Image message part is missing data or mime_type.")
                    parts.append(types.Part.from_bytes(data=part.data, mime_type=part.mime_type))
                    continue
                raise ValueError(f"Unsupported message part kind: {part.kind}")
            contents.append(types.Content(role=_gemini_role(message.role), parts=parts))
        return contents

    def _call_model(self, *, encoded_history: list[Any], request_config: Any) -> Any:
        if request_config is None:
            return self._client.models.generate_content(model=self.model, contents=encoded_history)
        return self._client.models.generate_content(
            model=self.model,
            contents=encoded_history,
            config=request_config,
        )

    def _extract_text(self, response: Any) -> str:
        lines: list[str] = []
        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) or []
            for part in parts:
                text = getattr(part, "text", None)
                if text:
                    lines.append(text)
        return "\n".join(lines).strip()

    def _build_provider_tools(self, tool_functions: Iterable[Callable[..., Any]]) -> list[Any]:
        from google.genai import types

        declarations: list[Any] = []
        for fn in tool_functions:
            schema = make_tool_schema(fn)
            declarations.append(
                types.FunctionDeclaration(
                    name=schema["name"],
                    description=schema["description"],
                    parameters=_json_schema_to_gemini(schema["parameters"]),
                )
            )
        return [types.Tool(function_declarations=declarations)]


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


def _gemini_role(role: Role) -> str:
    if role in {"user", "model"}:
        return role
    if role in {"tool", "system"}:
        return "user"
    raise ValueError(f"Unsupported history role for Gemini encoding: {role}")


def _json_schema_to_gemini(schema: dict[str, Any]) -> Any:
    from google.genai import types

    schema_type = schema["type"]
    kwargs: dict[str, Any] = {}

    description = schema.get("description")
    if description:
        kwargs["description"] = description

    if schema_type == "object":
        properties = schema.get("properties", {})
        kwargs["properties"] = {name: _json_schema_to_gemini(value) for name, value in properties.items()}
        required = schema.get("required", [])
        if required:
            kwargs["required"] = required
    elif schema_type == "array":
        items = schema.get("items")
        if items is None:
            raise ValueError("Array schema is missing `items`.")
        kwargs["items"] = _json_schema_to_gemini(items)

    return types.Schema(type=schema_type, **kwargs)


def _render_role(role: Role, *, color: bool) -> str:
    if not color:
        return role
    return f"{_ROLE_COLORS[role]}{role}{_ANSI_RESET}"


def _render_text_part(part_index: int, text: str) -> list[str]:
    lines = text.splitlines() or [""]
    rendered = [f"   - input {part_index} (text)", f"     content: {lines[0]}"]
    for line in lines[1:]:
        rendered.append(f"              {line}")
    return rendered


def _compression_label(mime_type: str) -> str:
    if mime_type == "image/png":
        return "lossless"
    if mime_type == "image/jpeg":
        return "lossy"
    return "unknown"


def _preprocess_image(data: bytes, *, mime_type: str, settings: ImageSettings) -> tuple[bytes, str]:
    from PIL import Image

    if settings.scale <= 0:
        raise ValueError("Image scale must be > 0.")

    source = io.BytesIO(data)
    with Image.open(source) as image:
        image.load()
        if settings.scale != 1.0:
            resized = (
                max(1, int(image.width * settings.scale)),
                max(1, int(image.height * settings.scale)),
            )
            image = image.resize(resized, Image.Resampling.LANCZOS)

        output_format = _resolve_output_format(settings.format, mime_type)
        if output_format == "JPEG" and image.mode not in {"RGB", "L"}:
            image = image.convert("RGB")

        buffer = io.BytesIO()
        save_kwargs: dict[str, Any] = {"optimize": True}
        if output_format == "JPEG":
            save_kwargs["quality"] = settings.quality

        image.save(buffer, format=output_format, **save_kwargs)

    output_mime = "image/jpeg" if output_format == "JPEG" else "image/png"
    return buffer.getvalue(), output_mime


def _resolve_output_format(requested_format: ImageFormat | None, mime_type: str) -> str:
    if requested_format == "jpeg":
        return "JPEG"
    if requested_format == "png":
        return "PNG"

    if mime_type == "image/jpeg":
        return "JPEG"
    if mime_type == "image/png":
        return "PNG"
    raise ValueError(f"Unsupported image mime type for preprocessing: {mime_type}")
