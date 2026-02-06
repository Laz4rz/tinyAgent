# tinyAgent

## CLI Computer-Use Session

Quick start:

```bash
python3 main.py
```

Debug run (records raw requests/responses + per-turn screenshots):

```bash
uv run main.py --debug
```

First run behavior:
1. Looks for `.tinyagent.config.json` in the current working directory.
2. If missing, prompts for provider, model, tools, and tool strategy.
3. Saves those values to `.tinyagent.config.json`.
4. In interactive terminals, setup uses arrow-key selection and checkbox-style tool toggles.

Supported providers:
1. Google Gemini (`GEMINI_API_KEY`)
2. OpenAI (`OPENAI_API_KEY`)

API key storage:
1. Uses one shared `.secret` file in the working directory.
2. The file stores keys as JSON by env-var name (for example `GEMINI_API_KEY`, `OPENAI_API_KEY`).

Notes:
1. The model yields control only through `return_to_user`.
2. Setup picker controls: `↑/↓` move, `Space` toggle checkbox, `Enter` confirm.
3. Tool setup starts with all tools selected by default.
4. Windows is supported for arrow/checkbox setup in regular terminal hosts; if stdin/stdout is non-TTY, the app falls back to text prompts.
5. Tool approval prompt supports `y` (run), `n` (deny), and `a` (deny + abort current run so you can provide another instruction).
6. When the provider returns reasoning/thinking summaries, the CLI prints them as `model> [thinking-summary] ...` (display/debug only).
7. Tool calls/results are stored as structured history events (not bracketed assistant text), so provider payloads stay close to native API formats.
8. `return_to_user` and aborted tool calls defer their tool-result item until your next input, so the next request sends tool result plus your new user message together.

Session commands:
- `/help`
- `/reconfigure`
- `/status`
- `/tools`
- `/history`
- `/clean`
- `/exit`

Debug artifacts:
1. Stored under `debug/run-<timestamp>/`.
2. `history_requests.json`: raw encoded history + request config per model call.
3. `responses.json`: raw model response payload per model call.
4. `screenshots/turn_XXXX.png`: one screenshot captured each model turn.
