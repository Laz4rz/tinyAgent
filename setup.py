from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from models import known_models
from tool_schema import make_tool_schema


ToolFn = Callable[..., Any]
ToolStrategy = str
CONFIG_PATH = Path(".tinyagent.config.json")
VALID_TOOL_STRATEGIES = {"ask", "auto"}


@dataclass(frozen=True)
class ProviderOption:
    key: str
    label: str


@dataclass(frozen=True)
class SessionConfig:
    provider: str
    model: str
    tools: list[str]
    tool_strategy: ToolStrategy


def config_path(config_name: Path = CONFIG_PATH) -> Path:
    return Path.cwd() / config_name


def _session_config_from_dict(
    payload: dict[str, Any],
    *,
    providers: list[ProviderOption],
    tool_registry: dict[str, ToolFn],
) -> SessionConfig:
    provider = payload.get("provider")
    model = payload.get("model")
    tools = payload.get("tools")
    tool_strategy = payload.get("tool_strategy")

    if not isinstance(provider, str) or not provider:
        raise ValueError("config `provider` must be a non-empty string")
    if provider not in {option.key for option in providers}:
        raise ValueError(f"config provider `{provider}` is not supported")

    if not isinstance(model, str) or not model:
        raise ValueError("config `model` must be a non-empty string")

    if not isinstance(tools, list) or not tools:
        raise ValueError("config `tools` must be a non-empty list")
    if not all(isinstance(tool_name, str) and tool_name for tool_name in tools):
        raise ValueError("config `tools` must contain non-empty strings")
    if any(tool_name not in tool_registry for tool_name in tools):
        known = ", ".join(sorted(tool_registry))
        raise ValueError(f"config `tools` contains unknown tool. Known tools: {known}")

    if not isinstance(tool_strategy, str) or tool_strategy not in VALID_TOOL_STRATEGIES:
        raise ValueError("config `tool_strategy` must be `ask` or `auto`")

    return SessionConfig(
        provider=provider,
        model=model,
        tools=tools,
        tool_strategy=tool_strategy,
    )


def load_session_config(
    path: Path,
    *,
    providers: list[ProviderOption],
    tool_registry: dict[str, ToolFn],
) -> SessionConfig | None:
    if not path.exists():
        return None

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("config root must be an object")
    return _session_config_from_dict(payload, providers=providers, tool_registry=tool_registry)


def save_session_config(path: Path, config: SessionConfig) -> None:
    payload = {
        "provider": config.provider,
        "model": config.model,
        "tools": config.tools,
        "tool_strategy": config.tool_strategy,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def print_tools(available_tools: list[ToolFn], *, handoff_tool_name: str) -> None:
    for index, fn in enumerate(available_tools, start=1):
        schema = make_tool_schema(fn)
        description = schema["description"] or "No description."
        print(f"{index}. {fn.__name__}: {description}")
    print(f"* {handoff_tool_name}: always enabled as handoff tool.")


def print_models(models: list[str], *, title: str) -> None:
    _section(title)
    for index, model_name in enumerate(models, start=1):
        print(f"{index}. {model_name}")


def parse_tool_selection(
    raw: str,
    *,
    available_tools: list[ToolFn],
    tool_registry: dict[str, ToolFn],
) -> list[ToolFn]:
    value = raw.strip().lower()
    if value in {"", "all", "*"}:
        return list(available_tools)

    names: list[str] = []
    for token in raw.split(","):
        item = token.strip()
        if not item:
            continue
        if item.isdigit():
            index = int(item)
            if index < 1 or index > len(available_tools):
                raise ValueError(f"tool index out of range: {item}")
            name = available_tools[index - 1].__name__
        else:
            name = item

        if name not in tool_registry:
            known = ", ".join(tool_registry)
            raise ValueError(f"unknown tool '{name}'. Known tools: {known}")
        if name not in names:
            names.append(name)

    if not names:
        raise ValueError("no tools selected")
    return [tool_registry[name] for name in names]


def run_setup_wizard(
    *,
    providers: list[ProviderOption],
    available_tools: list[ToolFn],
    tool_registry: dict[str, ToolFn],
    handoff_tool_name: str,
) -> SessionConfig:
    _section("Setup")
    provider = _prompt_provider(providers=providers)
    model = _prompt_model(provider=provider)
    tools = _prompt_tools(
        available_tools=available_tools,
        tool_registry=tool_registry,
        handoff_tool_name=handoff_tool_name,
    )
    tool_strategy = _prompt_tool_strategy()

    return SessionConfig(
        provider=provider,
        model=model,
        tools=[fn.__name__ for fn in tools],
        tool_strategy=tool_strategy,
    )


def ensure_session_config(
    *,
    config_file: Path,
    providers: list[ProviderOption],
    available_tools: list[ToolFn],
    tool_registry: dict[str, ToolFn],
    handoff_tool_name: str,
) -> SessionConfig:
    try:
        saved = load_session_config(
            config_file,
            providers=providers,
            tool_registry=tool_registry,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"warning: invalid config at {config_file}: {exc}")
        print("warning: config will be recreated.")
        saved = None

    if saved is not None:
        return saved

    print(f"No saved session config found. Creating: {config_file}")
    config = run_setup_wizard(
        providers=providers,
        available_tools=available_tools,
        tool_registry=tool_registry,
        handoff_tool_name=handoff_tool_name,
    )
    save_session_config(config_file, config)
    print(f"Saved session config to {config_file}")
    return config


def reconfigure_session(
    *,
    config_file: Path,
    providers: list[ProviderOption],
    available_tools: list[ToolFn],
    tool_registry: dict[str, ToolFn],
    handoff_tool_name: str,
) -> SessionConfig:
    print(f"Reconfiguring session settings in {config_file}")
    config = run_setup_wizard(
        providers=providers,
        available_tools=available_tools,
        tool_registry=tool_registry,
        handoff_tool_name=handoff_tool_name,
    )
    save_session_config(config_file, config)
    print(f"Saved session config to {config_file}")
    return config


def prompt_provider(*, providers: list[ProviderOption]) -> str:
    return _prompt_provider(providers=providers)


def prompt_tool_strategy() -> ToolStrategy:
    return _prompt_tool_strategy()


def parse_tool_strategy(raw: str) -> ToolStrategy:
    value = raw.strip().lower()
    if value in VALID_TOOL_STRATEGIES:
        return value
    raise ValueError("Invalid strategy. Use `ask` or `auto`.")


def _section(title: str) -> None:
    print(f"\n=== {title} ===")


def _can_use_arrow_picker() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _read_picker_key_posix() -> str:
    data = sys.stdin.buffer.read(1)
    if data in {b"\r", b"\n"}:
        return "enter"
    if data == b" ":
        return "space"
    if data == b"\x03":
        raise KeyboardInterrupt
    if data == b"\x1b":
        next_bytes = sys.stdin.buffer.read(2)
        if next_bytes == b"[A":
            return "up"
        if next_bytes == b"[B":
            return "down"
        return "escape"
    if data in {b"k", b"K"}:
        return "up"
    if data in {b"j", b"J"}:
        return "down"
    return "other"


def _read_picker_key_windows() -> str:
    import msvcrt

    ch = msvcrt.getwch()
    if ch in {"\r", "\n"}:
        return "enter"
    if ch == " ":
        return "space"
    if ch == "\x03":
        raise KeyboardInterrupt
    if ch in {"\x00", "\xe0"}:
        ext = msvcrt.getwch()
        if ext == "H":
            return "up"
        if ext == "P":
            return "down"
        return "other"
    if ch in {"k", "K"}:
        return "up"
    if ch in {"j", "J"}:
        return "down"
    return "other"


def _render_picker(
    title: str,
    options: list[str],
    *,
    cursor: int,
    checked: set[int] | None,
    hint: str,
    error: str | None = None,
) -> None:
    print("\x1b[2J\x1b[H", end="")
    print(title)
    print()
    for index, option in enumerate(options):
        cursor_marker = ">" if index == cursor else " "
        if checked is None:
            print(f"{cursor_marker} {option}")
        else:
            box = "x" if index in checked else " "
            print(f"{cursor_marker} [{box}] {option}")
    print()
    print(hint)
    if error:
        print(error)
    sys.stdout.flush()


def _picker_single_select(title: str, options: list[str], *, hint: str) -> int:
    if not options:
        raise ValueError("single-select picker requires at least one option")

    if os.name == "nt":
        cursor = 0
        while True:
            _render_picker(title, options, cursor=cursor, checked=None, hint=hint)
            key = _read_picker_key_windows()
            if key == "up":
                cursor = (cursor - 1) % len(options)
                continue
            if key == "down":
                cursor = (cursor + 1) % len(options)
                continue
            if key == "enter":
                print()
                return cursor
    else:
        import termios
        import tty

        cursor = 0
        fd = sys.stdin.fileno()
        old_mode = termios.tcgetattr(fd)

        try:
            tty.setraw(fd)
            while True:
                _render_picker(title, options, cursor=cursor, checked=None, hint=hint)
                key = _read_picker_key_posix()
                if key == "up":
                    cursor = (cursor - 1) % len(options)
                    continue
                if key == "down":
                    cursor = (cursor + 1) % len(options)
                    continue
                if key == "enter":
                    print()
                    return cursor
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_mode)


def _picker_multi_select(
    title: str,
    options: list[str],
    *,
    hint: str,
    prechecked: set[int] | None = None,
) -> list[int]:
    if not options:
        raise ValueError("multi-select picker requires at least one option")

    if os.name == "nt":
        cursor = 0
        checked: set[int] = set(prechecked or set())
        error: str | None = None

        while True:
            _render_picker(
                title,
                options,
                cursor=cursor,
                checked=checked,
                hint=hint,
                error=error,
            )
            key = _read_picker_key_windows()
            error = None
            if key == "up":
                cursor = (cursor - 1) % len(options)
                continue
            if key == "down":
                cursor = (cursor + 1) % len(options)
                continue
            if key == "space":
                if cursor in checked:
                    checked.remove(cursor)
                else:
                    checked.add(cursor)
                continue
            if key == "enter":
                if not checked:
                    error = "Select at least one item."
                    continue
                print()
                return sorted(checked)
    else:
        import termios
        import tty

        cursor = 0
        checked: set[int] = set(prechecked or set())
        error: str | None = None

        fd = sys.stdin.fileno()
        old_mode = termios.tcgetattr(fd)

        try:
            tty.setraw(fd)
            while True:
                _render_picker(
                    title,
                    options,
                    cursor=cursor,
                    checked=checked,
                    hint=hint,
                    error=error,
                )
                key = _read_picker_key_posix()
                error = None
                if key == "up":
                    cursor = (cursor - 1) % len(options)
                    continue
                if key == "down":
                    cursor = (cursor + 1) % len(options)
                    continue
                if key == "space":
                    if cursor in checked:
                        checked.remove(cursor)
                    else:
                        checked.add(cursor)
                    continue
                if key == "enter":
                    if not checked:
                        error = "Select at least one item."
                        continue
                    print()
                    return sorted(checked)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_mode)


def _prompt_provider(*, providers: list[ProviderOption]) -> str:
    if _can_use_arrow_picker():
        options = [option.key for option in providers]
        choice = _picker_single_select(
            "Provider",
            options,
            hint="Use ↑/↓ to move, Enter to select.",
        )
        return options[choice]

    _section("Provider")
    for index, option in enumerate(providers, start=1):
        print(f"{index}. {option.key}")

    while True:
        raw = input("Provider: ").strip().lower()
        if raw == "":
            print("Provider is required.")
            continue
        if raw.isdigit():
            index = int(raw)
            if 1 <= index <= len(providers):
                return providers[index - 1].key
        if raw in {option.key for option in providers}:
            return raw
        print("Invalid provider selection.")


def _prompt_model(*, provider: str) -> str:
    models = known_models(provider)
    if _can_use_arrow_picker():
        options = models + ["<type custom model id>"]
        choice = _picker_single_select(
            "Model",
            options,
            hint="Use ↑/↓ to move, Enter to select.",
        )
        if choice < len(models):
            return models[choice]
        while True:
            raw = input("Custom model id: ").strip()
            if raw:
                return raw
            print("Model is required.")

    print_models(models, title="Model")
    print("Type a model id or choose a number from the list.")

    while True:
        raw = input("Model: ").strip()
        if raw == "":
            print("Model is required.")
            continue
        if raw.isdigit():
            index = int(raw)
            if 1 <= index <= len(models):
                return models[index - 1]
            print("Model index is out of range.")
            continue
        return raw


def _prompt_tools(
    *,
    available_tools: list[ToolFn],
    tool_registry: dict[str, ToolFn],
    handoff_tool_name: str,
) -> list[ToolFn]:
    if _can_use_arrow_picker():
        labels = [fn.__name__ for fn in available_tools]
        selected_indexes = _picker_multi_select(
            "Tools",
            labels,
            hint="Use ↑/↓ to move, Space to toggle, Enter to confirm (all selected by default).",
            prechecked=set(range(len(labels))),
        )
        return [available_tools[index] for index in selected_indexes]

    _section("Tools")
    print_tools(available_tools, handoff_tool_name=handoff_tool_name)
    print("Enter comma-separated tool names or indexes, or `all`.")
    print("Press Enter to accept all tools.")

    while True:
        raw = input("Tools [all]: ").strip()
        if raw == "":
            raw = "all"
        try:
            return parse_tool_selection(
                raw,
                available_tools=available_tools,
                tool_registry=tool_registry,
            )
        except ValueError as exc:
            print(f"Invalid tool selection: {exc}")


def _prompt_tool_strategy() -> ToolStrategy:
    if _can_use_arrow_picker():
        options = [
            "ask  - ask for approval on every tool call",
            "auto - run tool calls without approval prompts",
        ]
        choice = _picker_single_select(
            "Tool Strategy",
            options,
            hint="Use ↑/↓ to move, Enter to select.",
        )
        return "ask" if choice == 0 else "auto"

    _section("Tool Strategy")
    print("ask  = ask for approval on every tool call")
    print("auto = run tool calls without approval prompts")

    while True:
        raw = input("Strategy (ask/auto): ").strip().lower()
        if raw == "":
            print("Tool strategy is required.")
            continue
        try:
            return parse_tool_strategy(raw)
        except ValueError as exc:
            print(str(exc))
