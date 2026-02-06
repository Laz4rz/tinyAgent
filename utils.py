from __future__ import annotations

import json
import os
import sys
from typing import Any, Callable, Literal, Protocol

from tool_schema import validate_and_call


ToolFn = Callable[..., Any]
ToolApproval = Literal["approve", "deny", "abort"]
RoleName = Literal["user", "model"]


class TextSink(Protocol):
    def add_text(self, text: str, *, role: str = "user") -> None:
        ...

ANSI_RESET = "\033[0m"
ROLE_COLORS: dict[RoleName, str] = {
    "user": "\033[38;5;39m",
    "model": "\033[38;5;82m",
}
STYLE_SECTION = "\033[1;36m"
STYLE_INFO = "\033[38;5;111m"
STYLE_SUCCESS = "\033[38;5;82m"
STYLE_WARNING = "\033[38;5;214m"
STYLE_ERROR = "\033[38;5;203m"
COLOR_ENABLED = False


def init_output_style() -> None:
    global COLOR_ENABLED

    _enable_windows_ansi()
    COLOR_ENABLED = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _enable_windows_ansi() -> None:
    if os.name != "nt":
        return

    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        output_handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint()
        if kernel32.GetConsoleMode(output_handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(output_handle, mode.value | 0x0004)
    except Exception:
        return


def _style(text: str, color_code: str) -> str:
    if not COLOR_ENABLED:
        return text
    return f"{color_code}{text}{ANSI_RESET}"


def role_label(role: RoleName) -> str:
    return _style(f"{role}>", ROLE_COLORS[role])


def print_section(title: str) -> None:
    print(f"\n{_style(f'=== {title} ===', STYLE_SECTION)}")


def print_role(role: RoleName, text: str) -> None:
    print(f"\n{role_label(role)} {text}")


def emit_message(target: TextSink, *, role: RoleName, text: str) -> None:
    print_role(role, text)
    target.add_text(text, role=role)


def print_info(text: str) -> None:
    print(_style(text, STYLE_INFO))


def print_success(text: str) -> None:
    print(_style(text, STYLE_SUCCESS))


def print_warning(text: str) -> None:
    print(_style(text, STYLE_WARNING))


def print_error(text: str) -> None:
    print(_style(text, STYLE_ERROR))


def user_prompt() -> str:
    return f"\n{role_label('user')} "


def print_session_help() -> None:
    print_info("\nSession commands:")
    print_info("  /help    Show commands")
    print_info("  /status  Show current configuration")
    print_info("  /reconfigure  Re-run setup and save config")
    print_info("  /tools   Show enabled tools")
    print_info("  /history Show local conversation history")
    print_info("  /clean   Clear local conversation history")
    print_info("  /exit    Quit")


def print_session_status(*, provider_label: str, model: str, tools: list[str], auto_approve_tools: bool) -> None:
    print_info(f"Provider: {provider_label}")
    print_info(f"Model: {model}")
    print_info(f"Tools: {', '.join(tools)}")
    print_info(f"Tool approval: {'auto-approve' if auto_approve_tools else 'ask every time'}")


def ask_tool_approval(prompt: str, *, default: ToolApproval = "deny") -> ToolApproval:
    if default not in {"approve", "deny", "abort"}:
        raise ValueError("default tool approval must be approve, deny, or abort")

    if default == "approve":
        suffix = "[Y/n/a]"
    elif default == "deny":
        suffix = "[y/N/a]"
    else:
        suffix = "[y/n/A]"

    while True:
        raw = input(f"{prompt} {suffix} ").strip().lower()
        if raw == "":
            return default
        if raw in {"y", "yes"}:
            return "approve"
        if raw in {"n", "no"}:
            return "deny"
        if raw in {"a", "abort"}:
            return "abort"
        print_warning("Please answer y, n, or a.")


def run_tool(tool_fn: ToolFn, args: dict[str, Any]) -> str:
    try:
        result = validate_and_call(tool_fn, args)
    except Exception as exc:
        return f"Tool `{tool_fn.__name__}` failed: {exc}"
    return str(result)


def format_tool_request(name: str, args: dict[str, Any]) -> str:
    return f"[tool-request] {name} {json.dumps(args)}"
