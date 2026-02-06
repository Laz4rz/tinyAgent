# tinyAgent

## CLI Computer-Use Session

Quick start:

```bash
python3 main.py
```

First run behavior:
1. Looks for `.tinyagent.config.json` in the current working directory.
2. If missing, prompts for provider, model, tools, and tool strategy.
3. Saves those values to `.tinyagent.config.json`.
4. In interactive terminals, setup uses arrow-key selection and checkbox-style tool toggles.

Notes:
1. The model yields control only through `return_to_user`.
2. Setup picker controls: `↑/↓` move, `Space` toggle checkbox, `Enter` confirm.
3. Tool setup starts with all tools selected by default.
4. Windows is supported for arrow/checkbox setup in regular terminal hosts; if stdin/stdout is non-TTY, the app falls back to text prompts.
5. Tool approval prompt supports `y` (run), `n` (deny), and `a` (deny + abort current run so you can provide another instruction).

Session commands:
- `/help`
- `/reconfigure`
- `/status`
- `/tools`
- `/history`
- `/clean`
- `/exit`
