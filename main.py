from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Callable

from api_key import load_api_key
from debug_recorder import DebugRecorder
from model_client import BaseModelClient, build_model_client
from setup import (
    CONFIG_PATH,
    SYSTEM_PROMPT_PATH,
    ProviderOption,
    SessionConfig,
    config_path,
    ensure_session_config,
    load_system_prompt,
    parse_tool_strategy,
    parse_tool_selection,
    prompt_tool_strategy,
    reconfigure_session,
    save_session_config,
)
from tools import click, move_mouse, press_combo, return_to_user, type as type_text
from utils import (
    ask_tool_approval,
    emit_message,
    format_control_note,
    format_protocol_note,
    format_tool_request,
    format_tool_result,
    format_thinking_summary,
    init_output_style,
    print_error,
    print_info,
    print_role,
    print_section,
    print_session_help,
    print_session_status,
    print_model_waiting,
    parse_main_cli_args,
    print_success,
    print_warning,
    run_tool,
    user_prompt,
)


ToolFn = Callable[..., Any]

EXIT_COMMANDS = {"/exit"}
HELP_COMMANDS = {"/help"}
HISTORY_COMMANDS = {"/history"}
RECONFIGURE_COMMANDS = {"/reconfigure"}
CLEAN_COMMANDS = {"/clean"}
STRATEGY_COMMANDS = {"/strategy"}
PROMPT_COMMANDS = {"/prompt"}
SHARED_SECRET_PATH = ".secret"
API_ENV_BY_PROVIDER = {
    "google": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
}

PROVIDERS = [
    ProviderOption(key="google", label="Google Gemini"),
    ProviderOption(key="openai", label="OpenAI"),
]
AVAILABLE_TOOLS: list[ToolFn] = [move_mouse, click, type_text, press_combo]
TOOL_REGISTRY: dict[str, ToolFn] = {fn.__name__: fn for fn in AVAILABLE_TOOLS}


@dataclass
class RuntimeSession:
    client: BaseModelClient
    provider: ProviderOption
    model: str
    agent_tools: list[ToolFn]
    auto_approve_tools: bool


@dataclass(frozen=True)
class DeferredToolResult:
    name: str
    call_id: str | None
    result_text: str
    use_next_user_message_as_result: bool = False


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


def _build_runtime_session(
    api_key: str,
    config: SessionConfig,
    *,
    system_prompt: str,
) -> RuntimeSession:
    print_info(f"Initializing model client ({config.provider}/{config.model})...")
    provider = _provider_option(config.provider)
    selected_tools = _select_tools(",".join(config.tools))
    agent_tools = selected_tools + [return_to_user]

    client = build_model_client(
        provider=config.provider,
        api_key=api_key,
        model=config.model,
        system_prompt=system_prompt,
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
    debug_recorder: DebugRecorder | None = None,
) -> list[DeferredToolResult]:
    tool_map = {fn.__name__: fn for fn in tool_functions}
    allowed_function_names = list(tool_map.keys())
    missing_function_call_rounds = 0

    while True:
        debug_turn: int | None = None
        debug_screenshot_path: str | None = None
        if debug_recorder is not None:
            debug_turn = debug_recorder.next_turn()
            debug_screenshot_path = debug_recorder.capture_screenshot(turn=debug_turn)

        request_config = client.tool_call_request_config(
            allowed_function_names=allowed_function_names,
        )
        print_model_waiting()
        try:
            response = client.generate(
                return_response=True,
                store_response=False,
                debug_recorder=debug_recorder,
                debug_turn=debug_turn,
                debug_screenshot_path=debug_screenshot_path,
                **request_config,
            )
        except Exception as exc:
            print_error(f"error: {exc}")
            return []

        try:
            function_calls = client.extract_tool_calls(response)
        except ValueError as exc:
            emit_message(
                client,
                role="user",
                text=format_protocol_note(f"invalid tool-call payload: {exc}"),
            )
            continue

        thinking_summaries = client.extract_thinking_summaries(response)
        model_text = client.extract_response_text(response)

        for summary in thinking_summaries:
            if summary.text:
                print_role("model", format_thinking_summary(summary.text))
        if model_text:
            print_role("model", model_text)
        for tool_call in function_calls:
            print_role("model", format_tool_request(tool_call.name, tool_call.args))

        client.add_model_turn(
            text=model_text,
            thinking_summaries=thinking_summaries,
            tool_calls=function_calls,
        )

        if not function_calls:
            missing_function_call_rounds += 1
            if missing_function_call_rounds > 3:
                failure = (
                    "Model repeatedly skipped tool calls "
                    "(including return_to_user). Stopping run."
                )
                emit_message(client, role="user", text=format_protocol_note(failure))
                raise RuntimeError("Model repeatedly skipped tool calls (including return_to_user).")

            reminder = (
                "Protocol reminder: you must call a tool. "
                "If you are done or need user input, call return_to_user(message=...)."
            )
            emit_message(client, role="user", text=format_protocol_note(reminder))
            continue

        missing_function_call_rounds = 0

        for tool_call in function_calls:
            name = tool_call.name
            args = tool_call.args

            if name == return_to_user.__name__:
                message = str(args.get("message", "")).strip()
                if message and message != model_text:
                    print_role("model", message)
                if not message and not model_text:
                    print_role("model", "Returning control to user.")
                return [
                    DeferredToolResult(
                        name=name,
                        call_id=tool_call.call_id,
                        result_text="",
                        use_next_user_message_as_result=True,
                    )
                ]

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

            if not aborted_by_user:
                print_role("user", format_tool_result(name, tool_result))
                client.add_tool_result(
                    name=name,
                    result=tool_result,
                    call_id=tool_call.call_id,
                    role="user",
                )

            if aborted_by_user:
                print_role("user", format_control_note("Execution paused. Waiting for user instruction."))
                return [
                    DeferredToolResult(
                        name=name,
                        call_id=tool_call.call_id,
                        result_text=tool_result,
                        use_next_user_message_as_result=False,
                    )
                ]


def _apply_deferred_tool_results(
    client: BaseModelClient,
    deferred_results: list[DeferredToolResult],
    *,
    next_user_message: str,
) -> bool:
    consumed_next_user_message = False
    for deferred in deferred_results:
        if deferred.use_next_user_message_as_result:
            result_text = next_user_message
            consumed_next_user_message = True
        else:
            result_text = deferred.result_text
        print_role("user", format_tool_result(deferred.name, result_text))
        client.add_tool_result(
            name=deferred.name,
            result=result_text,
            call_id=deferred.call_id,
            role="user",
        )
    return consumed_next_user_message


def main() -> None:
    init_output_style()
    cli_args = parse_main_cli_args(sys.argv[1:])
    debug_recorder: DebugRecorder | None = None
    if cli_args.debug:
        debug_recorder = DebugRecorder()
        print_info(f"Debug mode enabled. Writing artifacts to: {debug_recorder.run_dir}")

    config_file = config_path(CONFIG_PATH)
    system_prompt_file = config_path(SYSTEM_PROMPT_PATH)
    agent_system_prompt = load_system_prompt(system_prompt_file)
    resolved_config = ensure_session_config(
        config_file=config_file,
        providers=PROVIDERS,
        available_tools=AVAILABLE_TOOLS,
        tool_registry=TOOL_REGISTRY,
        handoff_tool_name=return_to_user.__name__,
    )
    session_config = resolved_config

    api_key = _load_provider_api_key(session_config.provider)
    runtime = _build_runtime_session(
        api_key,
        session_config,
        system_prompt=agent_system_prompt,
    )
    deferred_tool_results: list[DeferredToolResult] = []

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
                runtime = _build_runtime_session(
                    api_key,
                    new_config,
                    system_prompt=agent_system_prompt,
                )
                session_config = new_config
                deferred_tool_results = []
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
                deferred_tool_results = []
                print_success("Conversation history cleared.")
                continue
            if user_input in STRATEGY_COMMANDS or user_input.startswith("/strategy "):
                if user_input in STRATEGY_COMMANDS:
                    new_strategy = prompt_tool_strategy()
                else:
                    raw_strategy = user_input.split(maxsplit=1)[1]
                    try:
                        new_strategy = parse_tool_strategy(raw_strategy)
                    except ValueError as exc:
                        print_warning(str(exc))
                        continue

                if new_strategy == session_config.tool_strategy:
                    print_info(f"Tool strategy already set to `{new_strategy}`.")
                    continue

                session_config = SessionConfig(
                    provider=session_config.provider,
                    model=session_config.model,
                    tools=list(session_config.tools),
                    tool_strategy=new_strategy,
                )
                save_session_config(config_file, session_config)
                runtime.auto_approve_tools = new_strategy == "auto"
                print_success(f"Tool strategy set to `{new_strategy}` and saved.")
                continue
            if user_input in PROMPT_COMMANDS:
                print_section("System Prompt")
                print_role("model", agent_system_prompt)
                print_info(
                    f"Edit `{system_prompt_file}` to change it. Restart `main.py` to load updates."
                )
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
                print_section("Tools")
                print_info(", ".join(fn.__name__ for fn in runtime.agent_tools))
                continue
            if user_input in HISTORY_COMMANDS:
                print_section("History")
                print(runtime.client.format_history())
                continue
            if not user_input:
                continue

            if deferred_tool_results:
                consumed_user_input = _apply_deferred_tool_results(
                    runtime.client,
                    deferred_tool_results,
                    next_user_message=user_input,
                )
                deferred_tool_results = []
                if consumed_user_input:
                    deferred_tool_results = _run_agent_until_handoff(
                        runtime.client,
                        runtime.agent_tools,
                        auto_approve_tools=runtime.auto_approve_tools,
                        debug_recorder=debug_recorder,
                    )
                    continue

            runtime.client.add_text(user_input, role="user")
            deferred_tool_results = _run_agent_until_handoff(
                runtime.client,
                runtime.agent_tools,
                auto_approve_tools=runtime.auto_approve_tools,
                debug_recorder=debug_recorder,
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
