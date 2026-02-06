from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Callable

from api_key import load_api_key
from model_client import BaseModelClient, build_model_client
from setup import (
    CONFIG_PATH,
    ProviderOption,
    SessionConfig,
    config_path,
    ensure_session_config,
    parse_tool_selection,
    reconfigure_session,
)
from tools import click, move_mouse, press_combo, return_to_user
from utils import (
    ask_tool_approval,
    emit_message,
    format_tool_request,
    init_output_style,
    print_error,
    print_info,
    print_role,
    print_section,
    print_session_help,
    print_session_status,
    print_success,
    print_warning,
    run_tool,
    user_prompt,
)


ToolFn = Callable[..., Any]

AGENT_SYSTEM_PROMPT = (
    "You are a computer-use agent.\n"
    "You must use available tools for actions.\n"
    "After each tool result, continue working until done.\n"
    "When you finish, need clarification, or cannot proceed, call return_to_user(message=...).\n"
    "Do not stop by plain text alone: use return_to_user to hand control back."
)

EXIT_COMMANDS = {"/exit"}
HELP_COMMANDS = {"/help"}
HISTORY_COMMANDS = {"/history"}
RECONFIGURE_COMMANDS = {"/reconfigure"}
CLEAN_COMMANDS = {"/clean"}
SHARED_SECRET_PATH = ".secret"
API_ENV_BY_PROVIDER = {
    "google": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
}

PROVIDERS = [
    ProviderOption(key="google", label="Google Gemini"),
    ProviderOption(key="openai", label="OpenAI"),
]
AVAILABLE_TOOLS: list[ToolFn] = [move_mouse, click, press_combo]
TOOL_REGISTRY: dict[str, ToolFn] = {fn.__name__: fn for fn in AVAILABLE_TOOLS}


@dataclass
class RuntimeSession:
    client: BaseModelClient
    provider: ProviderOption
    model: str
    agent_tools: list[ToolFn]
    auto_approve_tools: bool


def _provider_option(provider_key: str) -> ProviderOption:
    for option in PROVIDERS:
        if option.key == provider_key:
            return option
    raise ValueError(f"Unsupported provider: {provider_key}")


def _select_tools(raw: str) -> list[ToolFn]:
    return parse_tool_selection(
        raw,
        available_tools=AVAILABLE_TOOLS,
        tool_registry=TOOL_REGISTRY,
    )


def _build_runtime_session(api_key: str, config: SessionConfig) -> RuntimeSession:
    print_info(f"Initializing model client ({config.provider}/{config.model})...")
    provider = _provider_option(config.provider)
    selected_tools = _select_tools(",".join(config.tools))
    agent_tools = selected_tools + [return_to_user]

    client = build_model_client(
        provider=config.provider,
        api_key=api_key,
        model=config.model,
        system_prompt=AGENT_SYSTEM_PROMPT,
    )
    client.set_tool_functions(agent_tools)

    print_success("Model client ready.")

    return RuntimeSession(
        client=client,
        provider=provider,
        model=config.model,
        agent_tools=agent_tools,
        auto_approve_tools=config.tool_strategy == "auto",
    )


def _load_provider_api_key(provider_key: str) -> str:
    try:
        env_var = API_ENV_BY_PROVIDER[provider_key]
    except KeyError as exc:
        raise ValueError(f"Unsupported provider for API key lookup: {provider_key}") from exc
    return load_api_key(
        env_var=env_var,
        secret_path=SHARED_SECRET_PATH,
        prompt=True,
    )


def _run_agent_until_handoff(
    client: BaseModelClient,
    tool_functions: list[ToolFn],
    *,
    auto_approve_tools: bool = False,
) -> None:
    tool_map = {fn.__name__: fn for fn in tool_functions}
    allowed_function_names = list(tool_map.keys())
    missing_function_call_rounds = 0

    while True:
        request_config = client.tool_call_request_config(
            allowed_function_names=allowed_function_names,
        )
        try:
            response = client.generate(
                return_response=True,
                store_response=False,
                **request_config,
            )
        except Exception as exc:
            print_error(f"[error] {exc}")
            return

        model_text = client.extract_response_text(response)
        if model_text:
            emit_message(client, role="model", text=model_text)

        try:
            function_calls = client.extract_tool_calls(response)
        except ValueError as exc:
            emit_message(client, role="user", text=f"[protocol] invalid tool-call payload: {exc}")
            continue
        if not function_calls:
            missing_function_call_rounds += 1
            if missing_function_call_rounds > 3:
                failure = (
                    "[protocol] Model repeatedly skipped tool calls "
                    "(including return_to_user). Stopping run."
                )
                emit_message(client, role="user", text=failure)
                raise RuntimeError("Model repeatedly skipped tool calls (including return_to_user).")

            reminder = (
                "Protocol reminder: you must call a tool. "
                "If you are done or need user input, call return_to_user(message=...)."
            )
            tagged_reminder = f"[protocol] {reminder}"
            emit_message(client, role="user", text=tagged_reminder)
            continue

        missing_function_call_rounds = 0

        for tool_call in function_calls:
            name = tool_call.name
            args = tool_call.args
            emit_message(client, role="model", text=format_tool_request(name, args))

            if name == return_to_user.__name__:
                message = str(args.get("message", "")).strip()
                if message and message != model_text:
                    emit_message(client, role="model", text=message)
                if not message and not model_text:
                    print_role("model", "Returning control to user.")
                return

            tool_fn = tool_map.get(name)
            aborted_by_user = False
            if tool_fn is None:
                tool_result = f"Tool `{name}` is not available."
            else:
                approval = "approve" if auto_approve_tools else ask_tool_approval(
                    f"Run tool `{name}`?",
                    default="deny",
                )

                if approval == "approve":
                    tool_result = run_tool(tool_fn, args)
                elif approval == "deny":
                    tool_result = f"User denied tool `{name}`."
                else:
                    aborted_by_user = True
                    tool_result = (
                        f"User denied tool `{name}` and aborted this run. "
                        "The user will provide another instruction."
                    )

            tagged_result = f"[tool-response] {name} result: {tool_result}"
            emit_message(client, role="user", text=tagged_result)

            if aborted_by_user:
                print_role("user", "[control] Execution paused. Waiting for user instruction.")
                return


def main() -> None:
    init_output_style()
    config_file = config_path(CONFIG_PATH)
    resolved_config = ensure_session_config(
        config_file=config_file,
        providers=PROVIDERS,
        available_tools=AVAILABLE_TOOLS,
        tool_registry=TOOL_REGISTRY,
        handoff_tool_name=return_to_user.__name__,
    )

    api_key = _load_provider_api_key(resolved_config.provider)
    runtime = _build_runtime_session(api_key, resolved_config)

    print_section("Session Ready")
    print_session_status(
        provider_label=runtime.provider.label,
        model=runtime.model,
        tools=[fn.__name__ for fn in runtime.agent_tools],
        auto_approve_tools=runtime.auto_approve_tools,
    )
    print_session_help()

    try:
        while True:
            user_input = input(user_prompt()).strip()
            if user_input in EXIT_COMMANDS:
                print_info("Exiting.")
                return
            if user_input in HELP_COMMANDS:
                print_session_help()
                continue
            if user_input in RECONFIGURE_COMMANDS:
                new_config = reconfigure_session(
                    config_file=config_file,
                    providers=PROVIDERS,
                    available_tools=AVAILABLE_TOOLS,
                    tool_registry=TOOL_REGISTRY,
                    handoff_tool_name=return_to_user.__name__,
                )
                runtime.client.close()
                api_key = _load_provider_api_key(new_config.provider)
                runtime = _build_runtime_session(api_key, new_config)
                print_section("Session Reconfigured")
                print_session_status(
                    provider_label=runtime.provider.label,
                    model=runtime.model,
                    tools=[fn.__name__ for fn in runtime.agent_tools],
                    auto_approve_tools=runtime.auto_approve_tools,
                )
                continue
            if user_input in CLEAN_COMMANDS:
                runtime.client.clear_history()
                print_success("Conversation history cleared.")
                continue
            if user_input == "/status":
                print_session_status(
                    provider_label=runtime.provider.label,
                    model=runtime.model,
                    tools=[fn.__name__ for fn in runtime.agent_tools],
                    auto_approve_tools=runtime.auto_approve_tools,
                )
                continue
            if user_input == "/tools":
                print_info("\n[tools]")
                print_info(", ".join(fn.__name__ for fn in runtime.agent_tools))
                continue
            if user_input in HISTORY_COMMANDS:
                print_info("\n[history]")
                print(runtime.client.format_history())
                continue
            if not user_input:
                continue

            runtime.client.add_text(user_input, role="user")
            _run_agent_until_handoff(
                runtime.client,
                runtime.agent_tools,
                auto_approve_tools=runtime.auto_approve_tools,
            )
    finally:
        runtime.client.close()


if __name__ == "__main__":
    try:
        main()
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        print_error(f"error: missing dependency '{missing}'. Install project dependencies first.")
        sys.exit(2)
    except ValueError as exc:
        print_error(f"error: {exc}")
        sys.exit(2)
    except KeyboardInterrupt:
        print_warning("\nInterrupted.")
        sys.exit(130)
