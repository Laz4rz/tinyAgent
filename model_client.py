from __future__ import annotations

import base64
import io
import json
import mimetypes
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Literal

from tool_schema import make_tool_schema

if TYPE_CHECKING:
    from debug_recorder import DebugRecorder


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
    max_width: int | None = None
    max_height: int | None = None
    format: ImageFormat | None = None
    quality: int = 85


@dataclass
class MessagePart:
    kind: Literal["text", "image", "thinking_summary", "tool_call", "tool_result"]
    text: str | None = None
    data: bytes | None = None
    mime_type: str | None = None
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_call_id: str | None = None
    tool_output: str | None = None
    item_id: str | None = None
    thought_signature: bytes | None = None
    encrypted_content: str | None = None


@dataclass
class Message:
    role: Role
    parts: list[MessagePart] = field(default_factory=list)


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict[str, Any]
    call_id: str | None = None
    thought_signature: bytes | None = None


@dataclass(frozen=True)
class ThinkingSummary:
    text: str
    item_id: str | None = None
    thought_signature: bytes | None = None
    encrypted_content: str | None = None


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

    def add_model_turn(
        self,
        *,
        text: str,
        thinking_summaries: list[ThinkingSummary],
        tool_calls: list[ToolCall],
    ) -> None:
        parts: list[MessagePart] = []
        for summary in thinking_summaries:
            parts.append(
                MessagePart(
                    kind="thinking_summary",
                    text=summary.text,
                    item_id=summary.item_id,
                    thought_signature=summary.thought_signature,
                    encrypted_content=summary.encrypted_content,
                )
            )

        if text:
            parts.append(MessagePart(kind="text", text=text))

        for tool_call in tool_calls:
            parts.append(
                MessagePart(
                    kind="tool_call",
                    tool_name=tool_call.name,
                    tool_args=tool_call.args,
                    tool_call_id=tool_call.call_id,
                    thought_signature=tool_call.thought_signature,
                )
            )

        if parts:
            self.append(Message(role="model", parts=parts))

    def add_tool_result(
        self,
        *,
        name: str,
        result: str,
        call_id: str | None,
        role: Role = "user",
    ) -> None:
        self.append(
            Message(
                role=role,
                parts=[
                    MessagePart(
                        kind="tool_result",
                        tool_name=name,
                        tool_output=result,
                        tool_call_id=call_id,
                    )
                ],
            )
        )

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

                if part.kind == "thinking_summary":
                    if part.text is None:
                        raise ValueError("Thinking-summary part is missing text.")
                    lines.append(f"   - input {part_index} (thinking_summary)")
                    if part.item_id:
                        lines.append(f"     item_id: {part.item_id}")
                    if part.encrypted_content:
                        lines.append("     encrypted_content: present")
                    lines.append(f"     content: {part.text}")
                    continue

                if part.kind == "tool_call":
                    if part.tool_name is None or part.tool_args is None:
                        raise ValueError("Tool-call part is missing name or args.")
                    lines.append(f"   - input {part_index} (tool_call)")
                    lines.append(f"     name: {part.tool_name}")
                    if part.tool_call_id:
                        lines.append(f"     call_id: {part.tool_call_id}")
                    if part.thought_signature:
                        lines.append("     thought_signature: present")
                    lines.append(f"     args: {json.dumps(part.tool_args)}")
                    continue

                if part.kind == "tool_result":
                    if part.tool_name is None or part.tool_output is None:
                        raise ValueError("Tool-result part is missing name or output.")
                    lines.append(f"   - input {part_index} (tool_result)")
                    lines.append(f"     name: {part.tool_name}")
                    if part.tool_call_id:
                        lines.append(f"     call_id: {part.tool_call_id}")
                    lines.append(f"     output: {part.tool_output}")
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

    def close(self) -> None:
        return

    def add_text(self, text: str, *, role: Role = "user") -> None:
        self.history.add_text(text, role=role)

    def add_model_turn(
        self,
        *,
        text: str,
        thinking_summaries: list[ThinkingSummary],
        tool_calls: list[ToolCall],
    ) -> None:
        self.history.add_model_turn(
            text=text,
            thinking_summaries=thinking_summaries,
            tool_calls=tool_calls,
        )

    def add_tool_result(
        self,
        *,
        name: str,
        result: str,
        call_id: str | None,
        role: Role = "user",
    ) -> None:
        self.history.add_tool_result(
            name=name,
            result=result,
            call_id=call_id,
            role=role,
        )

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

    def extract_response_text(self, response: Any) -> str:
        return self._extract_text(response)

    def extract_tool_calls(self, response: Any) -> list[ToolCall]:
        return self._extract_tool_calls(response)

    def extract_thinking_summaries(self, response: Any) -> list[ThinkingSummary]:
        return self._extract_thinking_summaries(response)

    def generate(
        self,
        *,
        store_response: bool = True,
        return_response: bool = False,
        debug_recorder: DebugRecorder | None = None,
        debug_turn: int | None = None,
        debug_screenshot_path: str | None = None,
        **config_kwargs: Any,
    ) -> Any:
        request_config = self._build_request_config(config_kwargs)
        encoded_history = self._encode_history()
        if debug_recorder is not None:
            if debug_turn is None:
                raise ValueError("debug_turn is required when debug_recorder is provided.")
            debug_recorder.record_request(
                turn=debug_turn,
                provider=self.__class__.__name__.removesuffix("Client").lower(),
                model=self.model,
                encoded_history=encoded_history,
                request_config=request_config,
                screenshot_path=debug_screenshot_path,
            )
        response = self._call_model(encoded_history=encoded_history, request_config=request_config)
        if debug_recorder is not None:
            debug_recorder.record_response(turn=debug_turn, response=response)

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

    def tool_call_request_config(self, *, allowed_function_names: Iterable[str]) -> dict[str, Any]:
        return {}

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

    @abstractmethod
    def _extract_tool_calls(self, response: Any) -> list[ToolCall]:
        raise NotImplementedError

    def _extract_thinking_summaries(self, response: Any) -> list[ThinkingSummary]:
        return []

    def _build_provider_tools(self, tool_functions: Iterable[Callable[..., Any]]) -> list[Any]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement callable tool conversion."
        )


class GeminiClient(BaseModelClient):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str,
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
        if "thinking_config" not in merged:
            merged["thinking_config"] = types.ThinkingConfig(include_thoughts=True)

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
                if part.kind == "thinking_summary":
                    if part.text is None:
                        raise ValueError("Thinking-summary part is missing text.")
                    parts.append(
                        types.Part(
                            text=part.text,
                            thought=True,
                            thought_signature=part.thought_signature,
                        )
                    )
                    continue
                if part.kind == "tool_call":
                    if part.tool_name is None or part.tool_args is None:
                        raise ValueError("Tool-call part is missing name or args.")
                    function_call_part = types.Part.from_function_call(
                        name=part.tool_name,
                        args=part.tool_args,
                    )
                    if part.thought_signature is not None:
                        function_call_part.thought_signature = part.thought_signature
                    parts.append(function_call_part)
                    continue
                if part.kind == "tool_result":
                    if part.tool_name is None or part.tool_output is None:
                        raise ValueError("Tool-result part is missing name or output.")
                    parts.append(
                        types.Part.from_function_response(
                            name=part.tool_name,
                            response={"result": part.tool_output},
                        )
                    )
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
        for candidate in response.candidates or []:
            for part in candidate.content.parts or []:
                text = part.text
                if text and not part.thought:
                    lines.append(text)
        return "\n".join(lines).strip()

    def _extract_tool_calls(self, response: Any) -> list[ToolCall]:
        calls: list[ToolCall] = []
        for candidate in response.candidates or []:
            for part in candidate.content.parts or []:
                function_call = part.function_call
                if function_call is None:
                    continue
                name = function_call.name
                if not name:
                    raise ValueError("Gemini function call is missing `name`.")
                calls.append(
                    ToolCall(
                        name=name,
                        args=_normalize_tool_args(function_call.args),
                        thought_signature=part.thought_signature,
                    )
                )
        return calls

    def _extract_thinking_summaries(self, response: Any) -> list[ThinkingSummary]:
        summaries: list[ThinkingSummary] = []
        for candidate in response.candidates or []:
            for part in candidate.content.parts or []:
                if part.thought and (part.text or part.thought_signature):
                    summaries.append(
                        ThinkingSummary(
                            text=part.text or "",
                            thought_signature=part.thought_signature,
                        )
                    )
        return summaries

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

    def tool_call_request_config(self, *, allowed_function_names: Iterable[str]) -> dict[str, Any]:
        from google.genai import types

        names = list(allowed_function_names)
        return {
            "tool_config": types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY",
                    allowed_function_names=names,
                )
            )
        }


class OpenAIClient(BaseModelClient):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str,
        system_prompt: str | None = None,
        tools: Iterable[Any] | None = None,
        **client_kwargs: Any,
    ) -> None:
        super().__init__(model=model, system_prompt=system_prompt)
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, **client_kwargs)
        if tools:
            self.set_tools(tools)

    def close(self) -> None:
        self._client.close()

    def _build_request_config(self, config_kwargs: dict[str, Any]) -> dict[str, Any]:
        merged = dict(config_kwargs)
        if self.system_prompt is not None and "instructions" not in merged:
            merged["instructions"] = self.system_prompt
        if self.tools and "tools" not in merged:
            merged["tools"] = self.tools
        if "store" not in merged:
            merged["store"] = False
        if "reasoning" not in merged:
            merged["reasoning"] = {"summary": "detailed"}
        include_items = merged.get("include")
        if include_items is None:
            normalized_include: list[str] = []
        elif isinstance(include_items, list):
            normalized_include = [str(item) for item in include_items]
        else:
            raise ValueError("OpenAI request config `include` must be a list of strings.")
        if "reasoning.encrypted_content" not in normalized_include:
            normalized_include.append("reasoning.encrypted_content")
        merged["include"] = normalized_include
        return merged

    def _encode_history(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for message_index, message in enumerate(self.history, start=1):
            if not message.parts:
                continue

            role = _openai_role(message.role)
            buffered_content: list[dict[str, Any]] = []

            def flush_buffered_message() -> None:
                nonlocal buffered_content
                if not buffered_content:
                    return

                if len(buffered_content) == 1 and buffered_content[0]["type"] == "input_text":
                    content: str | list[dict[str, Any]] = buffered_content[0]["text"]
                else:
                    content = list(buffered_content)

                items.append({"type": "message", "role": role, "content": content})
                buffered_content = []

            for part_index, part in enumerate(message.parts, start=1):
                if part.kind == "text":
                    if part.text is None:
                        raise ValueError("Text message part is missing text.")
                    buffered_content.append({"type": "input_text", "text": part.text})
                    continue

                if part.kind == "image":
                    if part.data is None or part.mime_type is None:
                        raise ValueError("Image message part is missing data or mime_type.")
                    encoded = base64.b64encode(part.data).decode("ascii")
                    data_url = f"data:{part.mime_type};base64,{encoded}"
                    buffered_content.append({"type": "input_image", "image_url": data_url})
                    continue

                flush_buffered_message()

                if part.kind == "thinking_summary":
                    reasoning_item: dict[str, Any] = {"type": "reasoning"}

                    if part.encrypted_content is not None:
                        reasoning_item["encrypted_content"] = part.encrypted_content

                    if part.text:
                        reasoning_item["summary"] = [
                            {
                                "type": "summary_text",
                                "text": part.text,
                            }
                        ]
                    else:
                        reasoning_item["summary"] = []

                    items.append(reasoning_item)
                    continue

                if part.kind == "tool_call":
                    if part.tool_name is None or part.tool_args is None:
                        raise ValueError("Tool-call part is missing name or args.")
                    call_id = part.tool_call_id or f"call_{message_index}_{part_index}"
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": call_id,
                            "name": part.tool_name,
                            "arguments": json.dumps(part.tool_args),
                        }
                    )
                    continue

                if part.kind == "tool_result":
                    if part.tool_output is None:
                        raise ValueError("Tool-result part is missing output.")
                    call_id = part.tool_call_id or f"call_{message_index}_{part_index}"
                    items.append(
                        {
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": part.tool_output,
                        }
                    )
                    continue

                raise ValueError(f"Unsupported message part kind: {part.kind}")

            flush_buffered_message()

        return items

    def _call_model(self, *, encoded_history: list[dict[str, Any]], request_config: dict[str, Any]) -> Any:
        return self._client.responses.create(
            model=self.model,
            input=encoded_history,
            **request_config,
        )

    def _extract_text(self, response: Any) -> str:
        try:
            output_text = response.output_text
        except AttributeError:
            output_text = None
        if output_text:
            return str(output_text).strip()

        try:
            output_items = response.output
        except AttributeError:
            output_items = None
        if output_items is None:
            return _extract_openai_chat_completion_text(response)

        lines: list[str] = []
        for item in output_items or []:
            item_type = item["type"] if isinstance(item, dict) else item.type
            if item_type != "message":
                continue

            content = item["content"] if isinstance(item, dict) else item.content
            for part in content or []:
                part_type = part["type"] if isinstance(part, dict) else part.type
                if part_type != "output_text":
                    continue
                text = part["text"] if isinstance(part, dict) else part.text
                if text:
                    lines.append(text)
        return "\n".join(lines).strip()

    def _extract_tool_calls(self, response: Any) -> list[ToolCall]:
        try:
            output_items = response.output
        except AttributeError:
            output_items = None
        if output_items is None:
            return _extract_openai_chat_completion_tool_calls(response)

        calls: list[ToolCall] = []
        for item in output_items or []:
            item_type = item["type"] if isinstance(item, dict) else item.type
            if item_type != "function_call":
                continue

            name = item["name"] if isinstance(item, dict) else item.name
            if not name:
                raise ValueError("OpenAI function call is missing function name.")
            raw_arguments = item["arguments"] if isinstance(item, dict) else item.arguments
            call_id = item["call_id"] if isinstance(item, dict) else item.call_id
            calls.append(
                ToolCall(
                    name=name,
                    args=_parse_openai_tool_arguments(raw_arguments),
                    call_id=call_id,
                )
            )
        return calls

    def _extract_thinking_summaries(self, response: Any) -> list[ThinkingSummary]:
        try:
            output_items = response.output
        except AttributeError:
            return []

        summaries: list[ThinkingSummary] = []
        for item in output_items or []:
            item_type = item["type"] if isinstance(item, dict) else item.type
            if item_type != "reasoning":
                continue
            item_id = item.get("id") if isinstance(item, dict) else item.id
            encrypted_content = (
                item.get("encrypted_content")
                if isinstance(item, dict)
                else item.encrypted_content
            )
            summary_parts = item["summary"] if isinstance(item, dict) else item.summary
            summary_texts: list[str] = []
            for part in summary_parts or []:
                part_type = part["type"] if isinstance(part, dict) else part.type
                if part_type != "summary_text":
                    continue
                text = part["text"] if isinstance(part, dict) else part.text
                if text:
                    summary_texts.append(text)
            if not summary_texts and encrypted_content is None:
                continue
            summaries.append(
                ThinkingSummary(
                    text="\n".join(summary_texts).strip(),
                    item_id=item_id,
                    encrypted_content=encrypted_content,
                )
            )
        return summaries

    def _build_provider_tools(self, tool_functions: Iterable[Callable[..., Any]]) -> list[Any]:
        declarations: list[dict[str, Any]] = []
        for fn in tool_functions:
            schema = make_tool_schema(fn)
            declarations.append(
                {
                    "type": "function",
                    "name": schema["name"],
                    "description": schema["description"],
                    "parameters": schema["parameters"],
                }
            )
        return declarations

    def tool_call_request_config(self, *, allowed_function_names: Iterable[str]) -> dict[str, Any]:
        names = list(allowed_function_names)
        if len(names) == 1:
            return {
                "tool_choice": {
                    "type": "function",
                    "name": names[0],
                }
            }
        return {"tool_choice": "required"}


def build_model_client(
    *,
    provider: str,
    api_key: str,
    model: str,
    system_prompt: str | None = None,
) -> BaseModelClient:
    if provider == "google":
        return GeminiClient(api_key=api_key, model=model, system_prompt=system_prompt)
    if provider == "openai":
        return OpenAIClient(api_key=api_key, model=model, system_prompt=system_prompt)
    raise ValueError(f"Unsupported provider: {provider}")


def _normalize_tool_args(args: Any) -> dict[str, Any]:
    if args is None:
        return {}
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        payload = json.loads(args)
        if not isinstance(payload, dict):
            raise ValueError("Tool args JSON payload must decode to an object.")
        return payload

    message_to_dict = None
    try:
        from google.protobuf.json_format import MessageToDict
    except ModuleNotFoundError:
        pass
    else:
        message_to_dict = MessageToDict

    if message_to_dict is not None:
        try:
            payload = message_to_dict(args)
        except (AttributeError, TypeError, ValueError):
            payload = None
        if payload is not None:
            if not isinstance(payload, dict):
                raise ValueError("Structured tool args payload must decode to an object.")
            return payload

    try:
        payload = dict(args)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unsupported tool args payload: {args!r}") from exc

    if not isinstance(payload, dict):
        raise ValueError("Tool args payload must be an object.")
    return payload


def _parse_openai_tool_arguments(arguments: Any) -> dict[str, Any]:
    if arguments is None or arguments == "":
        return {}
    if isinstance(arguments, dict):
        return arguments
    payload = json.loads(arguments)
    if not isinstance(payload, dict):
        raise ValueError("OpenAI function arguments must decode to a JSON object.")
    return payload


def _extract_openai_chat_completion_text(response: Any) -> str:
    lines: list[str] = []
    for choice in response.choices or []:
        message = choice.message
        if message is None:
            continue

        content = message.content
        if isinstance(content, str):
            if content:
                lines.append(content)
            continue

        if content is None:
            continue

        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                else:
                    text = item.text
                if text:
                    lines.append(text)
            continue

        raise ValueError(f"Unsupported OpenAI message content payload: {content!r}")

    return "\n".join(lines).strip()


def _extract_openai_chat_completion_tool_calls(response: Any) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for choice in response.choices or []:
        message = choice.message
        if message is None:
            continue
        for tool_call in message.tool_calls or []:
            function = tool_call.function
            if function is None:
                raise ValueError("OpenAI tool call is missing `function`.")
            name = function.name
            if not name:
                raise ValueError("OpenAI tool call is missing function name.")
            calls.append(
                ToolCall(
                    name=name,
                    args=_parse_openai_tool_arguments(function.arguments),
                    call_id=None,
                )
            )
    return calls


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


def _openai_role(role: Role) -> str:
    if role == "user":
        return "user"
    if role == "model":
        return "assistant"
    if role == "system":
        return "system"
    if role == "tool":
        return "user"
    raise ValueError(f"Unsupported history role for OpenAI encoding: {role}")


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
    if settings.max_width is not None and settings.max_width <= 0:
        raise ValueError("Image max_width must be > 0 when provided.")
    if settings.max_height is not None and settings.max_height <= 0:
        raise ValueError("Image max_height must be > 0 when provided.")

    has_target_resolution = settings.max_width is not None or settings.max_height is not None
    if has_target_resolution and settings.scale != 1.0:
        raise ValueError("Image scale cannot be combined with max_width/max_height target downscaling.")

    source = io.BytesIO(data)
    with Image.open(source) as image:
        image.load()
        if has_target_resolution:
            width_ratio = settings.max_width / image.width if settings.max_width is not None else 1.0
            height_ratio = settings.max_height / image.height if settings.max_height is not None else 1.0
            resize_ratio = min(width_ratio, height_ratio, 1.0)
            if resize_ratio < 1.0:
                resized = (
                    max(1, int(image.width * resize_ratio)),
                    max(1, int(image.height * resize_ratio)),
                )
                image = image.resize(resized, Image.Resampling.LANCZOS)
        elif settings.scale != 1.0:
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
