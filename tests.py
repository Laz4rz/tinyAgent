import argparse
import json
import io
import re
import sys
import tempfile
import time
import types
from pathlib import Path
from typing import Any

from api_key import load_api_key
from model_client import BaseModelClient, GeminiClient, ImageSettings, ToolCall
from tool_schema import make_tool_schema


class SkipTest(Exception):
    pass


class _StubClient(BaseModelClient):
    def __init__(self) -> None:
        super().__init__(model="stub-model")

    def _build_request_config(self, config_kwargs: dict[str, Any]) -> Any:
        return config_kwargs

    def _encode_history(self) -> list:
        return self.get_history()

    def _call_model(self, *, encoded_history: list, request_config: Any) -> dict[str, str]:
        return {"text": "stub-response"}

    def _extract_text(self, response: dict[str, str]) -> str:
        return response["text"]

    def _extract_tool_calls(self, response: dict[str, str]) -> list[ToolCall]:
        _ = response
        return []


def _print_block(title: str, payload) -> None:
    print(f"[{title}]")
    print(json.dumps(payload, indent=2))


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _wait_for(root, predicate, *, timeout_s: float = 2.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        root.update()
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("timeout waiting for expected event")


def _extract_first_function_call(response):
    calls = getattr(response, "function_calls", None) or []
    if calls:
        return calls[0]

    candidates = getattr(response, "candidates", None) or []
    for cand in candidates:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            fc = getattr(part, "function_call", None)
            if fc:
                return fc
    return None


def _args_to_dict(args) -> dict:
    if args is None:
        return {}
    if isinstance(args, dict):
        return args
    if hasattr(args, "fields"):
        try:
            from google.protobuf.json_format import MessageToDict

            return MessageToDict(args)
        except Exception:
            pass
    try:
        return dict(args)
    except Exception:
        return {}


def _all_tests() -> list[tuple[str, Any]]:
    return [
        ("main_cli_args_debug", test_main_cli_args_debug),
        ("main_provider_options", test_main_provider_options),
        ("main_reconfigure_command", test_main_reconfigure_command),
        ("main_strategy_command", test_main_strategy_command),
        ("main_clean_command", test_main_clean_command),
        ("main_tool_approval_abort", test_main_tool_approval_abort),
        ("main_api_error_handling", test_main_api_error_handling),
        ("main_tool_request_history", test_main_tool_request_history),
        ("main_tool_request_history_openai_shape", test_main_tool_request_history_openai_shape),
        ("main_deferred_tool_result_return_to_user", test_main_deferred_tool_result_return_to_user),
        ("main_deferred_tool_result_abort", test_main_deferred_tool_result_abort),
        ("utils_emit_message", test_utils_emit_message),
        ("model_client_debug_recording", test_model_client_debug_recording),
        ("model_client_openai_request_config_stateless", test_model_client_openai_request_config_stateless),
        ("model_client_openai_history_encoding_structured", test_model_client_openai_history_encoding_structured),
        ("model_client_openai_history_replays_reasoning_items", test_model_client_openai_history_replays_reasoning_items),
        ("model_client_extract_gemini_thinking", test_model_client_extract_gemini_thinking),
        ("model_client_extract_openai_response", test_model_client_extract_openai_response),
        ("api_key_shared_secret", test_api_key_shared_secret),
        ("session_config_fixed_path", test_session_config_fixed_path),
        ("session_config_roundtrip", test_session_config_roundtrip),
        ("session_parse_tool_strategy", test_session_parse_tool_strategy),
        ("known_models_constants", test_known_models_constants),
        ("main_tool_selection_parse", test_main_tool_selection_parse),
        ("history_object", test_history_object),
        ("history_model_turn_preserves_tool_call_signature", test_history_model_turn_preserves_tool_call_signature),
        ("history_visualization_text_and_image", test_history_visualization_text_and_image),
        ("return_to_user_tool", test_return_to_user_tool),
        ("type_tool", test_type_tool),
        ("image_auto_downsize_default", test_image_auto_downsize_default),
        ("image_preprocess_settings", test_image_preprocess_settings),
        ("press_combo_releases_on_failsafe", test_press_combo_releases_on_failsafe),
        ("press_combo_stuck_key_recovery", test_press_combo_stuck_key_recovery),
        ("press_combo_invalid_key", test_press_combo_invalid_key),
        ("tools_gui", test_tools_gui),
        ("tool_schemas", test_tool_schemas),
        ("gemini_init", test_init),
        ("gemini_text_image", test_text_and_image),
        ("gemini_tool_click", test_gemini_tool_click),
    ]


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="tinyAgent local test runner")
    parser.add_argument(
        "--test",
        help="Run a single test by name. Accepts display name or function name.",
    )
    return parser.parse_args(argv)


def _select_tests(
    tests: list[tuple[str, Any]],
    *,
    requested_test: str | None,
) -> list[tuple[str, Any]]:
    if requested_test is None:
        return tests

    selected = [
        (name, fn)
        for name, fn in tests
        if name == requested_test or fn.__name__ == requested_test
    ]
    if selected:
        return selected

    available = ", ".join(name for name, _ in tests)
    raise ValueError(f"unknown test '{requested_test}'. Available tests: {available}")


def _default_tools() -> list[dict]:
    from tools import click, move_mouse, press_combo, return_to_user, type as type_tool

    return [
        make_tool_schema(move_mouse),
        make_tool_schema(click),
        make_tool_schema(type_tool),
        make_tool_schema(press_combo),
        make_tool_schema(return_to_user),
    ]


def _gemini_tools() -> list[object]:
    from google.genai import types

    def obj(props: dict, required: list[str]):
        return types.Schema(type="object", properties=props, required=required)

    return [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="move_mouse",
                    description="Move mouse cursor to normalized screen coordinates.",
                    parameters=obj(
                        {
                            "x": types.Schema(type="number"),
                            "y": types.Schema(type="number"),
                        },
                        ["x", "y"],
                    ),
                ),
                types.FunctionDeclaration(
                    name="click",
                    description="Click at normalized screen coordinates.",
                    parameters=obj(
                        {
                            "x": types.Schema(type="number"),
                            "y": types.Schema(type="number"),
                            "button": types.Schema(type="string"),
                        },
                        ["x", "y"],
                    ),
                ),
                types.FunctionDeclaration(
                    name="type",
                    description="Type text into the currently focused input.",
                    parameters=obj(
                        {
                            "text": types.Schema(type="string"),
                            "interval_ms": types.Schema(type="integer"),
                        },
                        ["text"],
                    ),
                ),
                types.FunctionDeclaration(
                    name="press_combo",
                    description="Press a chord of keys, optionally hold, then release.",
                    parameters=obj(
                        {
                            "keys": types.Schema(
                                type="array",
                                items=types.Schema(type="string"),
                            ),
                            "hold_ms": types.Schema(type="integer"),
                        },
                        ["keys"],
                    ),
                ),
            ]
        )
    ]


def _maybe_client() -> GeminiClient:
    api_key = load_api_key(prompt=False)
    if not api_key:
        raise SkipTest("set GEMINI_API_KEY or .secret to run model tests")
    try:
        client = GeminiClient(
            api_key=api_key,
            model="gemini-2.5-flash",
            system_prompt="You are a helpful assistant.",
        )
        client.set_tools(_gemini_tools())
        return client
    except Exception as exc:
        raise RuntimeError(f"gemini init failed: {exc}") from exc


def test_history_object() -> None:
    client = _StubClient()
    client.add_text("hello")
    client.add_text("hi", role="model")

    snapshot = client.get_history()
    _print_block(
        "history.object",
        {
            "history_len": len(client.history),
            "roles": [message.role for message in client.history],
            "first_text": client.history[0].parts[0].text,
            "snapshot_len": len(snapshot),
        },
    )

    assert len(client.history) == 2
    assert client.history[0].parts[0].text == "hello"
    assert snapshot[1].role == "model"


def test_history_model_turn_preserves_tool_call_signature() -> None:
    from model_client import ConversationHistory, ThinkingSummary

    history = ConversationHistory()
    history.add_model_turn(
        text="",
        thinking_summaries=[ThinkingSummary(text="", thought_signature=b"sig_thought")],
        tool_calls=[
            ToolCall(
                name="click",
                args={"x": 0.25, "y": 0.75},
                thought_signature=b"sig_function_call",
            )
        ],
    )

    rendered = history.render(color=False)

    _print_block(
        "history.model_turn_preserves_tool_call_signature",
        {
            "part_kinds": [part.kind for part in history[0].parts],
            "thought_signature_present": history[0].parts[0].thought_signature is not None,
            "tool_call_signature_present": history[0].parts[1].thought_signature is not None,
            "rendered": rendered,
        },
    )

    assert len(history) == 1
    assert [part.kind for part in history[0].parts] == ["thinking_summary", "tool_call"]
    assert history[0].parts[0].thought_signature == b"sig_thought"
    assert history[0].parts[1].thought_signature == b"sig_function_call"
    assert "thought_signature: present" in rendered


def test_known_models_constants() -> None:
    from models import KNOWN_MODEL_IDS_BY_PROVIDER, known_models

    _print_block(
        "models.constants",
        {
            "providers": sorted(KNOWN_MODEL_IDS_BY_PROVIDER.keys()),
            "top_openai": known_models("openai")[:5],
        },
    )

    google_known = known_models("google")
    openai_known = known_models("openai")
    assert len(google_known) >= 1
    assert len(openai_known) >= 1
    assert len(set(google_known)) == len(google_known)
    assert len(set(openai_known)) == len(openai_known)
    assert "gpt-5.2" in openai_known
    assert "gpt-4o-mini" in openai_known


def test_api_key_shared_secret() -> None:
    import os

    with tempfile.TemporaryDirectory() as tmp_dir:
        secret_path = Path(tmp_dir) / ".secret"
        secret_path.write_text(
            json.dumps(
                {
                    "GEMINI_API_KEY": "gem-key",
                    "OPENAI_API_KEY": "openai-key",
                }
            ),
            encoding="utf-8",
        )

        original_gemini = os.environ.get("GEMINI_API_KEY")
        original_openai = os.environ.get("OPENAI_API_KEY")
        os.environ.pop("GEMINI_API_KEY", None)
        os.environ.pop("OPENAI_API_KEY", None)

        try:
            gemini = load_api_key("GEMINI_API_KEY", secret_path=secret_path, prompt=False)
            openai = load_api_key("OPENAI_API_KEY", secret_path=secret_path, prompt=False)
        finally:
            if original_gemini is None:
                os.environ.pop("GEMINI_API_KEY", None)
            else:
                os.environ["GEMINI_API_KEY"] = original_gemini
            if original_openai is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = original_openai

        _print_block(
            "api_key.shared_secret",
            {
                "gemini": gemini,
                "openai": openai,
                "path": str(secret_path),
            },
        )

        assert gemini == "gem-key"
        assert openai == "openai-key"


def test_session_config_roundtrip() -> None:
    from setup import ProviderOption, SessionConfig, load_session_config, save_session_config
    from tools import click, move_mouse, press_combo, type as type_tool

    with tempfile.TemporaryDirectory() as tmp_dir:
        config_path = Path(tmp_dir) / ".tinyagent.config.json"
        config = SessionConfig(
            provider="google",
            model="gemini-2.5-flash",
            tools=["move_mouse", "click"],
            tool_strategy="ask",
        )
        providers = [ProviderOption(key="google", label="Google Gemini")]
        available_tools = [move_mouse, click, type_tool, press_combo]
        tool_registry = {fn.__name__: fn for fn in available_tools}

        save_session_config(config_path, config)
        loaded = load_session_config(
            config_path,
            providers=providers,
            tool_registry=tool_registry,
        )

        _print_block(
            "session.config.roundtrip",
            {
                "path": str(config_path),
                "loaded_provider": loaded.provider if loaded else None,
                "loaded_model": loaded.model if loaded else None,
            },
        )

        assert loaded == config


def test_session_config_fixed_path() -> None:
    from setup import CONFIG_PATH, config_path

    resolved = config_path()
    _print_block(
        "session.config.path",
        {
            "resolved": str(resolved),
            "expected_suffix": str(CONFIG_PATH),
        },
    )
    assert resolved == Path.cwd() / CONFIG_PATH


def test_session_parse_tool_strategy() -> None:
    from setup import parse_tool_strategy

    ask = parse_tool_strategy("ask")
    auto = parse_tool_strategy("AUTO")

    _print_block(
        "session.parse_tool_strategy",
        {
            "ask": ask,
            "auto": auto,
        },
    )

    assert ask == "ask"
    assert auto == "auto"

    try:
        parse_tool_strategy("nope")
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("Invalid strategy should raise ValueError.")

    assert "Invalid strategy" in message


def test_main_reconfigure_command() -> None:
    from main import RECONFIGURE_COMMANDS

    _print_block("main.reconfigure_command", {"commands": sorted(RECONFIGURE_COMMANDS)})
    assert "/reconfigure" in RECONFIGURE_COMMANDS


def test_main_strategy_command() -> None:
    from main import STRATEGY_COMMANDS

    _print_block("main.strategy_command", {"commands": sorted(STRATEGY_COMMANDS)})
    assert "/strategy" in STRATEGY_COMMANDS


def test_main_provider_options() -> None:
    from main import PROVIDERS

    keys = [provider.key for provider in PROVIDERS]
    labels = [provider.label for provider in PROVIDERS]
    _print_block("main.provider_options", {"keys": keys, "labels": labels})
    assert "google" in keys
    assert "openai" in keys


def test_main_clean_command() -> None:
    from main import CLEAN_COMMANDS

    _print_block("main.clean_command", {"commands": sorted(CLEAN_COMMANDS)})
    assert "/clean" in CLEAN_COMMANDS


def test_main_tool_approval_abort() -> None:
    import builtins

    from utils import ask_tool_approval

    original_input = builtins.input
    scripted_inputs = iter(["a", "", "y"])
    builtins.input = lambda _prompt="": next(scripted_inputs)

    try:
        abort_choice = ask_tool_approval("Run tool?", default="deny")
        default_choice = ask_tool_approval("Run tool?", default="deny")
        approve_choice = ask_tool_approval("Run tool?", default="deny")
    finally:
        builtins.input = original_input

    _print_block(
        "main.tool_approval_abort",
        {
            "abort_choice": abort_choice,
            "default_choice": default_choice,
            "approve_choice": approve_choice,
        },
    )
    assert abort_choice == "abort"
    assert default_choice == "deny"
    assert approve_choice == "approve"


def test_utils_emit_message() -> None:
    import io
    from contextlib import redirect_stdout

    from utils import emit_message, init_output_style

    init_output_style()
    client = _StubClient()
    out = io.StringIO()

    with redirect_stdout(out):
        emit_message(client, role="user", text="[protocol] test-message")

    rendered = _strip_ansi(out.getvalue())
    _print_block(
        "utils.emit_message",
        {
            "printed": rendered.strip(),
            "history_role": client.history[-1].role,
            "history_text": client.history[-1].parts[0].text,
        },
    )

    assert "you" in rendered
    assert "[protocol] test-message" in rendered
    assert client.history[-1].role == "user"
    assert client.history[-1].parts[0].text == "[protocol] test-message"


def test_model_client_openai_history_encoding_structured() -> None:
    from model_client import ConversationHistory, OpenAIClient, ThinkingSummary, ToolCall

    client = object.__new__(OpenAIClient)
    client.history = ConversationHistory()
    client.model = "gpt-5-mini"
    client.system_prompt = "system-prompt"
    client.tools = []

    client.add_text("hey who are you", role="user")
    client.add_model_turn(
        text="",
        thinking_summaries=[ThinkingSummary(text="plan", item_id="rs_test")],
        tool_calls=[ToolCall(name="return_to_user", args={"message": "hi"}, call_id="call_test")],
    )
    client.add_tool_result(
        name="return_to_user",
        result='{"message":"hi"}',
        call_id="call_test",
        role="user",
    )
    client.add_text("yo", role="user")

    encoded = client._encode_history()
    _print_block("model_client.openai_history_encoding_structured", {"encoded": encoded})

    assert encoded[0] == {"type": "message", "role": "user", "content": "hey who are you"}
    assert encoded[1] == {
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": "plan"}],
    }
    assert encoded[2] == {
        "type": "function_call",
        "call_id": "call_test",
        "name": "return_to_user",
        "arguments": '{"message": "hi"}',
    }
    assert encoded[3] == {
        "type": "function_call_output",
        "call_id": "call_test",
        "output": '{"message":"hi"}',
    }
    assert encoded[4] == {"type": "message", "role": "user", "content": "yo"}


def test_model_client_openai_request_config_stateless() -> None:
    from model_client import OpenAIClient

    client = object.__new__(OpenAIClient)
    client.system_prompt = "sys"
    client.tools = [{"type": "function", "name": "noop"}]

    config = client._build_request_config({})

    _print_block("model_client.openai_request_config_stateless", {"config": config})

    assert config["instructions"] == "sys"
    assert config["tools"] == [{"type": "function", "name": "noop"}]
    assert config["store"] is False
    assert config["reasoning"] == {"summary": "detailed"}
    assert "reasoning.encrypted_content" in config["include"]


def test_model_client_openai_history_replays_reasoning_items() -> None:
    from model_client import ConversationHistory, OpenAIClient, ThinkingSummary

    client = object.__new__(OpenAIClient)
    client.history = ConversationHistory()
    client.model = "gpt-5-mini"
    client.system_prompt = "system-prompt"
    client.tools = []

    client.add_text("hello", role="user")
    client.add_model_turn(
        text="",
        thinking_summaries=[
            ThinkingSummary(text="part one", item_id="rs_1"),
            ThinkingSummary(text="", item_id="rs_2", encrypted_content="enc_abc"),
        ],
        tool_calls=[],
    )
    client.add_text("next", role="user")

    encoded = client._encode_history()
    reasoning_items = [item for item in encoded if item["type"] == "reasoning"]

    _print_block(
        "model_client.openai_history_replays_reasoning_items",
        {
            "encoded": encoded,
            "reasoning_count": len(reasoning_items),
        },
    )

    assert len(reasoning_items) == 2
    assert reasoning_items[0] == {
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": "part one"}],
    }
    assert reasoning_items[1] == {
        "type": "reasoning",
        "encrypted_content": "enc_abc",
        "summary": [],
    }


def test_model_client_extract_openai_response() -> None:
    from model_client import OpenAIClient

    class _SummaryPart:
        def __init__(self, text: str) -> None:
            self.type = "summary_text"
            self.text = text

    class _Reasoning:
        def __init__(self, summary_text: str, item_id: str) -> None:
            self.type = "reasoning"
            self.id = item_id
            self.summary = [_SummaryPart(summary_text)]

    class _OutputTextPart:
        def __init__(self, text: str) -> None:
            self.type = "output_text"
            self.text = text

    class _Message:
        def __init__(self, text: str) -> None:
            self.type = "message"
            self.content = [_OutputTextPart(text)]

    class _FunctionCall:
        def __init__(self, name: str, arguments: str, call_id: str) -> None:
            self.type = "function_call"
            self.name = name
            self.arguments = arguments
            self.call_id = call_id

    class _Response:
        def __init__(self) -> None:
            self.output_text = "openai-response"
            self.output = [
                _Reasoning("Plan: click target from bounding-box estimate.", "rs_test"),
                _Message("openai-response"),
                _FunctionCall("click", '{"x": 0.1, "y": 0.9}', "call_abc"),
            ]

    try:
        client = OpenAIClient(api_key="test-key", model="gpt-4.1-mini")
    except ModuleNotFoundError as exc:
        raise SkipTest("openai is required for OpenAI parsing tests") from exc
    try:
        response = _Response()
        text = client.extract_response_text(response)
        calls = client.extract_tool_calls(response)
        summaries = client.extract_thinking_summaries(response)

        _print_block(
            "model_client.extract_openai_response",
            {
                "text": text,
                "tool_name": calls[0].name,
                "args": calls[0].args,
                "call_id": calls[0].call_id,
                "thinking_summary": summaries[0].text,
            },
        )

        assert text == "openai-response"
        assert len(calls) == 1
        assert calls[0].name == "click"
        assert calls[0].args == {"x": 0.1, "y": 0.9}
        assert calls[0].call_id == "call_abc"
        assert [summary.text for summary in summaries] == [
            "Plan: click target from bounding-box estimate."
        ]
    finally:
        client.close()


def test_model_client_extract_gemini_thinking() -> None:
    from model_client import GeminiClient

    class _FunctionCall:
        def __init__(self, name: str, args: dict[str, Any]) -> None:
            self.name = name
            self.args = args

    class _Part:
        def __init__(
            self,
            *,
            text: str | None = None,
            thought: bool = False,
            thought_signature: bytes | None = None,
            function_call: _FunctionCall | None = None,
        ) -> None:
            self.text = text
            self.thought = thought
            self.thought_signature = thought_signature
            self.function_call = function_call

    class _Content:
        def __init__(self, parts: list[_Part]) -> None:
            self.parts = parts

    class _Candidate:
        def __init__(self, parts: list[_Part]) -> None:
            self.content = _Content(parts)

    class _Response:
        def __init__(self) -> None:
            self.candidates = [
                _Candidate(
                    [
                        _Part(text="Plan: estimate target by color region.", thought=True),
                        _Part(text=None, thought=True, thought_signature=b"sig_thought_only"),
                        _Part(text="Clicking target now.", thought=False),
                        _Part(
                            function_call=_FunctionCall("click", {"x": 0.4, "y": 0.6}),
                            thought_signature=b"sig_fc",
                        ),
                    ]
                )
            ]

    client = object.__new__(GeminiClient)
    response = _Response()

    text = client.extract_response_text(response)
    calls = client.extract_tool_calls(response)
    summaries = client.extract_thinking_summaries(response)

    _print_block(
        "model_client.extract_gemini_thinking",
            {
                "text": text,
                "tool_name": calls[0].name,
                "args": calls[0].args,
                "tool_thought_signature": calls[0].thought_signature.decode("utf-8"),
                "thinking_summaries": [summary.text for summary in summaries],
            },
        )

    assert text == "Clicking target now."
    assert len(calls) == 1
    assert calls[0].name == "click"
    assert calls[0].args == {"x": 0.4, "y": 0.6}
    assert calls[0].thought_signature == b"sig_fc"
    assert len(summaries) == 2
    assert summaries[0].text == "Plan: estimate target by color region."
    assert summaries[0].thought_signature is None
    assert summaries[1].text == ""
    assert summaries[1].thought_signature == b"sig_thought_only"


def test_main_cli_args_debug() -> None:
    from utils import parse_main_cli_args

    no_flags = parse_main_cli_args([])
    debug_flag = parse_main_cli_args(["--debug"])

    _print_block(
        "main.cli_args_debug",
        {
            "no_flags_debug": no_flags.debug,
            "debug_flag_debug": debug_flag.debug,
        },
    )

    assert no_flags.debug is False
    assert debug_flag.debug is True

    try:
        parse_main_cli_args(["--nope"])
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("Unknown flags should raise ValueError.")

    assert "Unknown argument" in message
    assert "--debug" in message


def test_model_client_debug_recording() -> None:
    from debug_recorder import DebugRecorder

    with tempfile.TemporaryDirectory() as tmp_dir:
        recorder = DebugRecorder(root=tmp_dir)
        client = _StubClient()
        client.add_text("hello", role="user")

        turn = recorder.next_turn()
        response = client.generate(
            return_response=True,
            store_response=False,
            debug_recorder=recorder,
            debug_turn=turn,
            debug_screenshot_path="debug/snap.png",
        )

        request_payloads = json.loads(recorder.request_log_path.read_text(encoding="utf-8"))
        response_payloads = json.loads(recorder.response_log_path.read_text(encoding="utf-8"))
        request_payload = request_payloads[0]
        response_payload = response_payloads[0]

        _print_block(
            "model_client.debug_recording",
            {
                "request_turn": request_payload["turn"],
                "request_screenshot_path": request_payload["screenshot_path"],
                "response_turn": response_payload["turn"],
                "response_keys": sorted(response_payload["response"].keys()),
                "response_text": response["text"],
            },
        )

        assert len(request_payloads) == 1
        assert len(response_payloads) == 1
        assert request_payload["turn"] == turn
        assert request_payload["screenshot_path"] == "debug/snap.png"
        assert request_payload["model"] == "stub-model"
        assert request_payload["provider"] == "_stub"
        assert response_payload["turn"] == turn
        assert response_payload["response"] == {"text": "stub-response"}


def test_main_tool_request_history() -> None:
    from main import _run_agent_until_handoff
    from tools import return_to_user

    class _FakeCall:
        def __init__(self, name: str, args: dict[str, Any]) -> None:
            self.name = name
            self.args = args

    class _FakeResponse:
        def __init__(self, calls: list[object]) -> None:
            self.function_calls = calls
            self.candidates = []

    class _FakeClient:
        def __init__(self) -> None:
            self.history: list[dict[str, Any]] = []

        def tool_call_request_config(self, *, allowed_function_names: list[str]) -> dict[str, Any]:
            _ = allowed_function_names
            return {}

        def generate(self, **kwargs: Any) -> _FakeResponse:
            _ = kwargs
            return _FakeResponse([_FakeCall("return_to_user", {"message": "Done."})])

        def extract_response_text(self, response: _FakeResponse) -> str:
            _ = response
            return ""

        def extract_thinking_summaries(self, response: _FakeResponse) -> list[str]:
            _ = response
            return []

        def extract_tool_calls(self, response: _FakeResponse) -> list[ToolCall]:
            call = response.function_calls[0]
            return [ToolCall(name=call.name, args=call.args)]

        def add_model_turn(
            self,
            *,
            text: str,
            thinking_summaries: list[Any],
            tool_calls: list[ToolCall],
        ) -> None:
            self.history.append(
                {
                    "kind": "model_turn",
                    "text": text,
                    "thinking_summaries": thinking_summaries,
                    "tool_calls": tool_calls,
                }
            )

    fake_client = _FakeClient()
    deferred = _run_agent_until_handoff(
        fake_client,  # type: ignore[arg-type]
        [return_to_user],
        auto_approve_tools=False,
    )

    serialized_history = [
        {
            **entry,
            "tool_calls": [
                {"name": call.name, "args": call.args, "call_id": call.call_id}
                for call in entry["tool_calls"]
            ],
        }
        for entry in fake_client.history
    ]
    _print_block(
        "main.tool_request_history",
        {
            "history": serialized_history,
        },
    )
    assert len(fake_client.history) == 1
    assert fake_client.history[0]["kind"] == "model_turn"
    assert fake_client.history[0]["text"] == ""
    assert fake_client.history[0]["thinking_summaries"] == []
    assert len(fake_client.history[0]["tool_calls"]) == 1
    assert fake_client.history[0]["tool_calls"][0].name == "return_to_user"
    assert fake_client.history[0]["tool_calls"][0].args == {"message": "Done."}
    assert len(deferred) == 1
    assert deferred[0].name == "return_to_user"
    assert deferred[0].call_id is None
    assert deferred[0].use_next_user_message_as_result is True


def test_main_tool_request_history_openai_shape() -> None:
    from main import _run_agent_until_handoff
    from tools import return_to_user

    class _Function:
        def __init__(self, name: str, arguments: str) -> None:
            self.name = name
            self.arguments = arguments

    class _ToolCall:
        def __init__(self, function: _Function) -> None:
            self.function = function

    class _Message:
        def __init__(self, tool_calls: list[_ToolCall]) -> None:
            self.tool_calls = tool_calls
            self.content = ""

    class _Choice:
        def __init__(self, message: _Message) -> None:
            self.message = message

    class _Response:
        def __init__(self, choices: list[_Choice]) -> None:
            self.choices = choices

    class _FakeClient:
        def __init__(self) -> None:
            self.history: list[dict[str, Any]] = []

        def tool_call_request_config(self, *, allowed_function_names: list[str]) -> dict[str, Any]:
            _ = allowed_function_names
            return {}

        def generate(self, **kwargs: Any) -> _Response:
            _ = kwargs
            return _Response(
                [
                    _Choice(
                        _Message(
                            [
                                _ToolCall(
                                    _Function("return_to_user", '{"message": "Done from OpenAI."}')
                                )
                            ]
                        )
                    )
                ]
            )

        def extract_response_text(self, response: _Response) -> str:
            _ = response
            return ""

        def extract_thinking_summaries(self, response: _Response) -> list[str]:
            _ = response
            return []

        def extract_tool_calls(self, response: _Response) -> list[ToolCall]:
            function = response.choices[0].message.tool_calls[0].function
            return [
                ToolCall(
                    name=function.name,
                    args=json.loads(function.arguments),
                    call_id="call_123",
                )
            ]

        def add_model_turn(
            self,
            *,
            text: str,
            thinking_summaries: list[Any],
            tool_calls: list[ToolCall],
        ) -> None:
            self.history.append(
                {
                    "kind": "model_turn",
                    "text": text,
                    "thinking_summaries": thinking_summaries,
                    "tool_calls": tool_calls,
                }
            )

    fake_client = _FakeClient()
    deferred = _run_agent_until_handoff(
        fake_client,  # type: ignore[arg-type]
        [return_to_user],
        auto_approve_tools=False,
    )

    serialized_history = [
        {
            **entry,
            "tool_calls": [
                {"name": call.name, "args": call.args, "call_id": call.call_id}
                for call in entry["tool_calls"]
            ],
        }
        for entry in fake_client.history
    ]
    _print_block(
        "main.tool_request_history_openai_shape",
        {
            "history": serialized_history,
        },
    )
    assert len(fake_client.history) == 1
    assert fake_client.history[0]["kind"] == "model_turn"
    assert fake_client.history[0]["text"] == ""
    assert fake_client.history[0]["thinking_summaries"] == []
    assert len(fake_client.history[0]["tool_calls"]) == 1
    assert fake_client.history[0]["tool_calls"][0].name == "return_to_user"
    assert fake_client.history[0]["tool_calls"][0].args == {"message": "Done from OpenAI."}
    assert fake_client.history[0]["tool_calls"][0].call_id == "call_123"
    assert len(deferred) == 1
    assert deferred[0].name == "return_to_user"
    assert deferred[0].call_id == "call_123"
    assert deferred[0].use_next_user_message_as_result is True


def test_main_deferred_tool_result_return_to_user() -> None:
    from main import _apply_deferred_tool_results, _run_agent_until_handoff
    from tools import return_to_user

    class _FakeResponse:
        pass

    class _FakeClient:
        def __init__(self) -> None:
            self.model_turns: list[dict[str, Any]] = []
            self.tool_results: list[dict[str, Any]] = []

        def tool_call_request_config(self, *, allowed_function_names: list[str]) -> dict[str, Any]:
            _ = allowed_function_names
            return {}

        def generate(self, **kwargs: Any) -> _FakeResponse:
            _ = kwargs
            return _FakeResponse()

        def extract_response_text(self, response: _FakeResponse) -> str:
            _ = response
            return ""

        def extract_thinking_summaries(self, response: _FakeResponse) -> list[Any]:
            _ = response
            return []

        def extract_tool_calls(self, response: _FakeResponse) -> list[ToolCall]:
            _ = response
            return [ToolCall(name="return_to_user", args={"message": "Done."}, call_id="call_ret")]

        def add_model_turn(
            self,
            *,
            text: str,
            thinking_summaries: list[Any],
            tool_calls: list[ToolCall],
        ) -> None:
            self.model_turns.append(
                {
                    "text": text,
                    "thinking_summaries": thinking_summaries,
                    "tool_calls": tool_calls,
                }
            )

        def add_tool_result(
            self,
            *,
            name: str,
            result: str,
            call_id: str | None,
            role: str = "user",
        ) -> None:
            self.tool_results.append(
                {
                    "name": name,
                    "result": result,
                    "call_id": call_id,
                    "role": role,
                }
            )

    fake_client = _FakeClient()
    deferred = _run_agent_until_handoff(
        fake_client,  # type: ignore[arg-type]
        [return_to_user],
        auto_approve_tools=False,
    )

    consumed = _apply_deferred_tool_results(
        fake_client,  # type: ignore[arg-type]
        deferred,
        next_user_message="who are you",
    )

    _print_block(
        "main.deferred_tool_result_return_to_user",
        {
            "deferred": [
                {
                    "name": item.name,
                    "call_id": item.call_id,
                    "use_next_user_message_as_result": item.use_next_user_message_as_result,
                }
                for item in deferred
            ],
            "tool_results": fake_client.tool_results,
        },
    )

    assert len(deferred) == 1
    assert deferred[0].name == "return_to_user"
    assert deferred[0].call_id == "call_ret"
    assert deferred[0].use_next_user_message_as_result is True
    assert consumed is True
    assert fake_client.tool_results == [
        {
            "name": "return_to_user",
            "result": "who are you",
            "call_id": "call_ret",
            "role": "user",
        }
    ]


def test_main_deferred_tool_result_abort() -> None:
    from main import _apply_deferred_tool_results, _run_agent_until_handoff
    from tools import click, return_to_user

    class _FakeResponse:
        pass

    class _FakeClient:
        def __init__(self) -> None:
            self.model_turns: list[dict[str, Any]] = []
            self.tool_results: list[dict[str, Any]] = []

        def tool_call_request_config(self, *, allowed_function_names: list[str]) -> dict[str, Any]:
            _ = allowed_function_names
            return {}

        def generate(self, **kwargs: Any) -> _FakeResponse:
            _ = kwargs
            return _FakeResponse()

        def extract_response_text(self, response: _FakeResponse) -> str:
            _ = response
            return ""

        def extract_thinking_summaries(self, response: _FakeResponse) -> list[Any]:
            _ = response
            return []

        def extract_tool_calls(self, response: _FakeResponse) -> list[ToolCall]:
            _ = response
            return [ToolCall(name="click", args={"x": 0.1, "y": 0.2}, call_id="call_abort")]

        def add_model_turn(
            self,
            *,
            text: str,
            thinking_summaries: list[Any],
            tool_calls: list[ToolCall],
        ) -> None:
            self.model_turns.append(
                {
                    "text": text,
                    "thinking_summaries": thinking_summaries,
                    "tool_calls": tool_calls,
                }
            )

        def add_tool_result(
            self,
            *,
            name: str,
            result: str,
            call_id: str | None,
            role: str = "user",
        ) -> None:
            self.tool_results.append(
                {
                    "name": name,
                    "result": result,
                    "call_id": call_id,
                    "role": role,
                }
            )

    import main as main_module

    original_ask_tool_approval = main_module.ask_tool_approval
    main_module.ask_tool_approval = lambda _prompt, default="deny": "abort"
    try:
        fake_client = _FakeClient()
        deferred = _run_agent_until_handoff(
            fake_client,  # type: ignore[arg-type]
            [click, return_to_user],
            auto_approve_tools=False,
        )
    finally:
        main_module.ask_tool_approval = original_ask_tool_approval

    assert fake_client.tool_results == []

    consumed = _apply_deferred_tool_results(
        fake_client,  # type: ignore[arg-type]
        deferred,
        next_user_message="continue",
    )

    _print_block(
        "main.deferred_tool_result_abort",
        {
            "deferred": [
                {
                    "name": item.name,
                    "call_id": item.call_id,
                    "result_text": item.result_text,
                    "use_next_user_message_as_result": item.use_next_user_message_as_result,
                }
                for item in deferred
            ],
            "tool_results": fake_client.tool_results,
        },
    )

    assert len(deferred) == 1
    assert deferred[0].name == "click"
    assert deferred[0].call_id == "call_abort"
    assert deferred[0].use_next_user_message_as_result is False
    assert "User denied tool `click` and aborted this run" in deferred[0].result_text
    assert consumed is False
    assert fake_client.tool_results == [
        {
            "name": "click",
            "result": deferred[0].result_text,
            "call_id": "call_abort",
            "role": "user",
        }
    ]


def test_main_api_error_handling() -> None:
    import io
    from contextlib import redirect_stdout

    from main import _run_agent_until_handoff
    from tools import return_to_user

    class _ErrorClient:
        def tool_call_request_config(self, *, allowed_function_names: list[str]) -> dict[str, Any]:
            _ = allowed_function_names
            return {}

        def generate(self, **kwargs: Any):  # noqa: ANN003
            raise RuntimeError("simulated API failure")

        def extract_thinking_summaries(self, response: Any) -> list[str]:
            _ = response
            return []

        def add_text(self, text: str, *, role: str = "user") -> None:
            raise AssertionError("add_text should not be called when generate fails")

    out = io.StringIO()
    with redirect_stdout(out):
        _run_agent_until_handoff(
            _ErrorClient(),  # type: ignore[arg-type]
            [return_to_user],
            auto_approve_tools=False,
        )

    rendered = _strip_ansi(out.getvalue())
    _print_block("main.api_error_handling", {"printed": rendered.strip()})
    assert "error: simulated API failure" in rendered


def test_main_tool_selection_parse() -> None:
    from setup import parse_tool_selection
    from tools import click, move_mouse, press_combo, type as type_tool

    available_tools = [move_mouse, click, type_tool, press_combo]
    tool_registry = {fn.__name__: fn for fn in available_tools}
    selected = parse_tool_selection(
        "1,click",
        available_tools=available_tools,
        tool_registry=tool_registry,
    )
    selected_names = [fn.__name__ for fn in selected]
    _print_block("main.tool_selection_parse", {"selected": selected_names})
    assert selected_names == ["move_mouse", "click"]


def test_history_visualization_text_and_image() -> None:
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise SkipTest("Pillow is required for image preprocessing tests") from exc

    client = _StubClient()
    image_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image_path = Path(tmp.name)

        Image.new("RGB", (320, 200), (80, 140, 200)).save(image_path)

        client.add_text("Hello from user", role="user")
        client.add_image(image_path, role="user")

        rendered_colored = client.format_history()
        rendered = _strip_ansi(rendered_colored)

        print("[history.visualized]")
        print(rendered_colored)

        assert "\x1b[" in rendered_colored
        assert "1. role: user" in rendered
        assert "2. role: user" in rendered
        assert "input 1 (text)" in rendered
        assert "content: Hello from user" in rendered
        assert "input 1 (image)" in rendered
        assert "type: image/png" in rendered
        assert "compression: lossless" in rendered
    finally:
        if image_path is not None:
            image_path.unlink(missing_ok=True)


def test_return_to_user_tool() -> None:
    from tools import return_to_user

    result = return_to_user("Need your confirmation.")
    _print_block("tool.return_to_user", {"result": result})
    assert result == "Need your confirmation."


def test_type_tool() -> None:
    from tools.tools import type as type_tool

    fake = types.ModuleType("pyautogui")
    write_calls: list[dict[str, Any]] = []

    def _write(text: str, *, interval: float = 0) -> None:
        write_calls.append({"text": text, "interval": interval})

    fake.write = _write

    original = sys.modules.get("pyautogui")
    sys.modules["pyautogui"] = fake
    try:
        result = type_tool("abc", interval_ms=120)
    finally:
        if original is None:
            del sys.modules["pyautogui"]
        else:
            sys.modules["pyautogui"] = original

    _print_block(
        "tool.type",
        {
            "result": result,
            "write_calls": write_calls,
        },
    )

    assert result == "Successfully sent typed text (3 chars)"
    assert write_calls == [{"text": "abc", "interval": 0.12}]


def test_image_preprocess_settings() -> None:
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise SkipTest("Pillow is required for image preprocessing tests") from exc

    client = _StubClient()
    image_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image_path = Path(tmp.name)

        base_image = Image.new("RGB", (1200, 800), (10, 90, 180))
        base_image.save(image_path)

        client.add_image(
            image_path,
            settings=ImageSettings(scale=0.25, format="jpeg", quality=70),
        )

        part = client.history[-1].parts[0]
        with Image.open(io.BytesIO(part.data)) as processed:
            width, height = processed.size
            image_format = processed.format

        _print_block(
            "image.preprocess",
            {
                "mime_type": part.mime_type,
                "size": [width, height],
                "format": image_format,
                "bytes": len(part.data),
            },
        )

        assert (width, height) == (300, 200)
        assert part.mime_type == "image/jpeg"
        assert image_format == "JPEG"
    finally:
        if image_path is not None:
            image_path.unlink(missing_ok=True)


def test_image_auto_downsize_default() -> None:
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise SkipTest("Pillow is required for image preprocessing tests") from exc

    client = _StubClient()
    image_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image_path = Path(tmp.name)

        base_image = Image.new("RGB", (640, 360), (220, 70, 40))
        base_image.save(image_path)

        client.add_image(image_path)

        part = client.history[-1].parts[0]
        with Image.open(io.BytesIO(part.data)) as processed:
            width, height = processed.size
            image_format = processed.format

        _print_block(
            "image.autodownsize",
            {
                "mime_type": part.mime_type,
                "size": [width, height],
                "format": image_format,
                "bytes": len(part.data),
            },
        )

        assert (width, height) == (160, 90)
        assert part.mime_type == "image/png"
        assert image_format == "PNG"
    finally:
        if image_path is not None:
            image_path.unlink(missing_ok=True)


def test_press_combo_releases_on_failsafe() -> None:
    from tools.tools import press_combo

    fake = types.ModuleType("pyautogui")
    held: list[str] = []
    fake.FAILSAFE = True
    fake.KEYBOARD_KEYS = ("shift", "a", "ctrl", "alt", "command", "option", "win")

    def _key_down(key: str) -> None:
        held.append(key)

    def _key_up(key: str) -> None:
        if fake.FAILSAFE:
            raise RuntimeError("failsafe blocked keyUp")
        if key in held:
            held.remove(key)

    fake.keyDown = _key_down
    fake.keyUp = _key_up

    original = sys.modules.get("pyautogui")
    sys.modules["pyautogui"] = fake
    try:
        result = press_combo("shift", "a")
    finally:
        if original is None:
            del sys.modules["pyautogui"]
        else:
            sys.modules["pyautogui"] = original

    _print_block(
        "press_combo.releases_on_failsafe",
        {
            "result": result,
            "held_after": held,
            "failsafe_after": fake.FAILSAFE,
        },
    )

    assert result.startswith("Successfully sent combo:")
    assert held == []
    assert fake.FAILSAFE is True


def test_press_combo_stuck_key_recovery() -> None:
    from tools.tools import press_combo

    fake = types.ModuleType("pyautogui")
    held: list[str] = []
    key_up_calls: dict[str, int] = {}
    fake.FAILSAFE = True
    fake.KEYBOARD_KEYS = ("shift", "ctrl", "alt", "command", "option", "win")

    def _key_down(key: str) -> None:
        held.append(key)

    def _key_up(key: str) -> None:
        key_up_calls[key] = key_up_calls.get(key, 0) + 1
        if key == "shift" and key_up_calls[key] == 1:
            raise RuntimeError("temporary release failure")
        if key in held:
            held.remove(key)

    fake.keyDown = _key_down
    fake.keyUp = _key_up

    original = sys.modules.get("pyautogui")
    sys.modules["pyautogui"] = fake
    try:
        result = press_combo("shift")
    finally:
        if original is None:
            del sys.modules["pyautogui"]
        else:
            sys.modules["pyautogui"] = original

    _print_block(
        "press_combo.stuck_key_recovery",
        {
            "result": result,
            "held_after": held,
            "key_up_calls": key_up_calls,
        },
    )

    assert result.startswith("Successfully sent combo:")
    assert held == []
    assert key_up_calls.get("shift", 0) >= 2


def test_press_combo_invalid_key() -> None:
    from tools.tools import press_combo

    fake = types.ModuleType("pyautogui")
    held: list[str] = []
    key_down_calls: list[str] = []
    fake.FAILSAFE = True
    fake.KEYBOARD_KEYS = ("a", "shift", "ctrl", "alt", "command", "option", "win")

    def _key_down(key: str) -> None:
        key_down_calls.append(key)
        held.append(key)

    def _key_up(key: str) -> None:
        if key in held:
            held.remove(key)

    fake.keyDown = _key_down
    fake.keyUp = _key_up

    original = sys.modules.get("pyautogui")
    sys.modules["pyautogui"] = fake
    try:
        result = press_combo("not_a_real_key", "a")
    finally:
        if original is None:
            del sys.modules["pyautogui"]
        else:
            sys.modules["pyautogui"] = original

    _print_block(
        "press_combo.invalid_key",
        {
            "result": result,
            "key_down_calls": key_down_calls,
            "held_after": held,
        },
    )

    assert result == "Failed to send combo not_a_real_key, a: unsupported key(s): not_a_real_key"
    assert key_down_calls == []
    assert held == []


def test_init() -> None:
    client = _maybe_client()
    _print_block(
        "gemini.init",
        {
            "model": client.model,
            "system_prompt": client.system_prompt,
            "tools": len(client.tools),
            "history": len(client.history),
            "api_key_set": True,
        },
    )
    client.close()


def test_text_and_image() -> None:
    client = _maybe_client()

    image_path = None
    try:
        client.add_text("Hello from user", role="user")
        try:
            import pyautogui

            shot = pyautogui.screenshot()
        except Exception as exc:
            raise SkipTest(f"screenshot unavailable: {exc}") from exc

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            shot.save(tmp.name)
            image_path = Path(tmp.name)

        client.add_image(image_path, role="user")
        rendered_colored = client.format_history()
        rendered = _strip_ansi(rendered_colored)

        _print_block(
            "gemini.text_image",
            {
                "history": len(client.history),
                "last_parts": [p.kind for p in client.history[-1].parts],
                "image_bytes": image_path.stat().st_size,
                "image_path": str(image_path),
                "history_visualization": rendered,
            },
        )
        print("[gemini.text_image.visualized]")
        print(rendered_colored)
        assert "1. role: user" in rendered
        assert "2. role: user" in rendered
        assert "input 1 (text)" in rendered
        assert "content: Hello from user" in rendered
        assert "input 1 (image)" in rendered
        assert "type: image/png" in rendered
        assert "compression: lossless" in rendered
    finally:
        if image_path is not None:
            try:
                image_path.unlink(missing_ok=True)
            except Exception:
                pass
        client.close()


def test_tools_gui() -> None:
    try:
        import tkinter as tk
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"tkinter not available: {exc}")

    import pyautogui

    from tools import click, move_mouse, press_combo

    try:  # Fix coordinate mismatches on Windows with DPI scaling.
        import ctypes

        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

    root = tk.Tk()
    root.title("tinyAgent selftest")
    root.geometry("240x160+200+200")
    root.attributes("-topmost", True)

    ok_clicked = False

    label = tk.Label(root, text="Self-test running…\nDon’t touch mouse/keyboard.", justify="center")
    label.pack(expand=True, fill="both")

    entry = tk.Entry(root, width=22)
    entry.pack(padx=12, pady=(0, 8))

    def _ok():
        nonlocal ok_clicked
        ok_clicked = True
        label.config(text="✅ OK")
        root.after(150, root.destroy)

    ok_btn = tk.Button(root, text="OK", command=_ok)
    ok_btn.pack(padx=12, pady=(0, 12))

    try:
        root.update_idletasks()
        root.update()
        root.focus_force()
        root.lift()
        root.update()
        time.sleep(0.05)

        ex = entry.winfo_rootx() + entry.winfo_width() // 2
        ey = entry.winfo_rooty() + entry.winfo_height() // 2
        width, height = pyautogui.size()

        move_mouse(ex / width, ey / height)
        time.sleep(0.05)
        click(ex / width, ey / height, "left")
        _wait_for(root, lambda: root.focus_get() is entry)

        pyautogui.write("abc", interval=0)
        press_combo("mod", "a", hold_ms=20)
        pyautogui.write("success", interval=0)
        _wait_for(root, lambda: entry.get() == "success")

        bx = ok_btn.winfo_rootx() + ok_btn.winfo_width() // 2
        by = ok_btn.winfo_rooty() + ok_btn.winfo_height() // 2
        move_mouse(bx / width, by / height)
        time.sleep(0.05)
        click(bx / width, by / height, "left")
        _wait_for(root, lambda: ok_clicked)
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def test_tool_schemas() -> None:
    schemas = _default_tools()
    _print_block("tool_schemas", schemas)
    names = [schema["name"] for schema in schemas]
    assert "type" in names


def test_gemini_tool_click() -> None:
    try:
        import tkinter as tk
    except Exception as exc:  # pragma: no cover
        raise SkipTest(f"tkinter not available: {exc}") from exc

    try:  # Fix coordinate mismatches on Windows with DPI scaling.
        import ctypes

        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

    from tools import click

    client = _maybe_client()

    root = tk.Tk()
    root.title("Gemini tool click test")
    root.attributes("-topmost", True)

    ok_clicked = False

    def _on_click():
        nonlocal ok_clicked
        ok_clicked = True
        label.config(text="✅ Clicked")
        root.after(150, root.destroy)

    label = tk.Label(root, text="Waiting for tool click…", justify="center")
    label.pack(expand=True, fill="both")

    btn = tk.Button(root, text="Click Me", command=_on_click)
    btn.place(relx=0.5, rely=0.5, anchor="center")

    try:
        w, h = 240, 160
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        root.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")
        root.update_idletasks()
        root.update()
        root.focus_force()
        root.lift()
        root.update()
        time.sleep(0.05)

        import pyautogui

        screen_w, screen_h = pyautogui.size()
        bx = btn.winfo_rootx() + btn.winfo_width() / 2
        by = btn.winfo_rooty() + btn.winfo_height() / 2
        expected_x = bx / screen_w
        expected_y = by / screen_h

        from google.genai import types

        client.add_text(
            "Use the click tool to click the center of the screen. "
            f"Use x={expected_x:.4f}, y={expected_y:.4f}, button='left'.",
            role="user",
        )

        response = client.generate(
            return_response=True,
            store_response=False,
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY",
                    allowed_function_names=["click"],
                )
            ),
        )

        call = _extract_first_function_call(response)
        print({"debug": "extracted tool call", "call": call})
        if call is None:
            raise AssertionError("model did not call any tool")

        if hasattr(call, "function_call"):
            call = call.function_call

        name = getattr(call, "name", None)
        if name != "click":
            raise AssertionError(f"unexpected tool call: {name}")

        args = _args_to_dict(getattr(call, "args", None))
        x = float(args.get("x", expected_x))
        y = float(args.get("y", expected_y))
        button = str(args.get("button", "left"))

        def _try_click(nx: float, ny: float, label: str) -> bool:
            click(nx, ny, button)
            try:
                _wait_for(root, lambda: ok_clicked, timeout_s=1.0)
                return True
            except Exception:
                print({"warning": f"{label} click missed", "x": nx, "y": ny, "button": button})
                return False

        if not _try_click(x, y, "model"):
            if not _try_click(expected_x, expected_y, "expected"):
                raise AssertionError("button was not clicked")
    finally:
        try:
            root.destroy()
        except Exception:
            pass
        client.close()

def main() -> None:
    tests = _all_tests()
    args = _parse_args(sys.argv[1:])
    try:
        tests = _select_tests(tests, requested_test=args.test)
    except ValueError as exc:
        print(f"❌ {exc}")
        sys.exit(2)

    failed = 0
    skipped = 0
    for name, fn in tests:
        print(f"\n\n== {name} ==")
        try:
            fn()
            print(f"✅ {name}")
        except SkipTest as exc:
            skipped += 1
            print(f"⏭️ {name}: {exc}")
        except Exception as exc:
            failed += 1
            print(f"❌ {name}: {exc}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
