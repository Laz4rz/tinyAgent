# tinyAgent Collaboration Notes

## What This File Is For
This file is a durable playbook for how to work on this project together.

It should capture:
1. Engineering style preferences.
2. Architectural intent and non-goals.
3. Decision rules for tradeoffs.
4. Practical defaults for future tasks.

It should not be a changelog (Git already provides that).

## Project Intent
tinyAgent is an educational codebase. Clarity and teachability are first-class goals.

Core product intent:
1. Keep the model-client architecture generalizable where it matters (provider abstraction).
2. Keep day-to-day code simple and explicit.
3. Prefer readable, direct flows over defensive complexity.

## Working Style Contract
Use these defaults for future work unless explicitly overridden.

1. Simplicity over defensive coding.
2. Fail fast for internal invariants.
3. Avoid broad exception swallowing.
4. Minimize magic/fallback behavior.
5. Keep abstractions small and justified.

### Specifically Avoid
1. `try/except Exception` around large internal flows.
2. `getattr(..., default)` in places where object shape is controlled by us.
3. Silent defaults like `or ""` / `or b""` that hide broken state.
4. Abstractions that are added "just in case" and not used.

### Specifically Prefer
1. Direct field access for stable internal structures.
2. Explicit errors when assumptions are violated.
3. Small, named hook points for true extension seams.
4. Code that a learner can read top-to-bottom without guessing hidden behavior.

## Architecture Direction

### Provider Model
`BaseModelClient` should own the common orchestration flow.

Provider classes should only implement provider-specific parts:
1. Request config shaping.
2. History/message encoding.
3. Provider API call.
4. Response text extraction.

Rule: if behavior is identical across providers, it belongs in base.
Rule: provider-specific callable-tool declaration building belongs in model-client code, not in CLI entrypoints.
Rule: when adding a provider, wire the full path together in one change (model constants, setup selection, API-key source, runtime client factory, and response/tool-call parsing) to avoid partial support.
Rule: provider response parsing (text/tool-call extraction and args decoding) belongs in provider model clients, not in generic CLI utils.

### History Model
History should stay easy to inspect and manipulate.

Current preference:
1. Keep history list-like and explicit.
2. No extra helper surface unless a real repeated need appears.
3. Readability of history operations is more important than API completeness.
4. For CLI history visualization, show text content directly and summarize images with metadata only (for example type and compression), never raw bytes.

### Image Handling
Default behavior should be practical and lightweight.

Current policy:
1. Auto-downsize images to 25% dimensions by default.
2. Allow explicit override via settings when needed.
3. Keep a provider hook for provider-specific image preprocessing paths.

### Tools Module Convention
All callable tools should live under a single package: `tools/`.

Current preference:
1. Treat `tools/` as the canonical location for all tool functions.
2. Keep callable tool functions in tool-facing modules (for example `tools/tools.py`).
3. Keep shared helper logic in dedicated helper modules (for example `tools/helpers.py`).
4. Keep tool interfaces simple and schema-friendly (clean signatures, clear types).
5. Use one coordinate convention across pointer tools; currently both `move_mouse` and `click` use normalized `x, y` in `[0, 1]`.
6. Keep an explicit `return_to_user` handoff tool so autonomous loops have a clear, inspectable stop condition.
7. Any keyboard tool that calls key-down events must guarantee key-up cleanup inside the same tool call, including failure/retry paths.
8. For low-level input tools (for example keyboard/mouse dispatch), success text should describe dispatched input, not guarantee the external app/OS effect happened.
9. Keep a dedicated text-entry tool (`type`) distinct from shortcut/chord tools (`press_combo`) so model intent stays explicit.

### Runtime Loop
For interactive computer-use sessions:
1. Ask for explicit user approval before executing each non-handoff tool call.
2. Let the model continue autonomously after tool results until it calls `return_to_user`.
3. Keep provider/model/tool selection steps explicit in the CLI for teachability.
4. Use a single explicit config path (`.tinyagent.config.json` in working directory) instead of path-discovery/fallback chains.
4a. Use a single shared secret file (`.secret`) for provider API keys, keyed by env var name.
5. On first run (no config), prompt explicitly for provider/model/tools/tool strategy and persist them; do not silently pick runtime defaults.
6. Prefer arrow-key interactive pickers (and checkbox-style multi-select for tools) when running in a real TTY, with text prompts as fallback.
7. Keep a runtime command to re-run setup and persist new config without restarting the process.
8. Ensure picker key handling works on both POSIX terminals and Windows consoles; keep non-TTY fallback prompts.
9. In setup, start with all tools selected by default so users can deselect instead of opt-in one by one.
10. Keep runtime output role-oriented and easy to scan: use clear role-prefixed lines (for example `agent › ...`, `you › ...`), and render tool/protocol/control/thinking/status notes as visually emphasized semantic tags.
11. Keep a `/clean` runtime command that clears local conversation history without restarting the process.
12. For Gemini responses in tool mode, extract text directly from response parts instead of relying on `response.text` warnings.
13. Keep setup/config wizard logic in a dedicated setup module so `main.py` stays focused on session orchestration and the interaction loop.
14. Tool approval prompts should support an explicit abort option that records a denial tool result and yields control back to the user for a new instruction.
15. Keep `main.py` as a lean interaction entrypoint; move argument parsing and utility/helper functions into dedicated modules.
16. Keep print/store parity in the runtime loop: any message appended to model history must be echoed to terminal output in the same step.
17. Store model tool-call intents and tool results as structured history entries (not bracketed plain-text logs) so encoded provider requests stay close to native API formats.
18. Treat provider API failures as external-boundary errors: catch around model request calls, print a clear red `error: ...` line, and return control to user without crashing the process.
19. Keep an explicit `--debug` CLI mode that records per-turn screenshots plus raw model request/response payloads under a timestamped `debug/` run directory.
20. Keep debug payload artifacts human-inspectable by writing pretty JSON files (indented arrays) instead of dense single-line logs.
21. When providers expose model reasoning/thinking summaries, print them as readable model notes (for example `thinking · ...`) for tool-action traceability without noisy bracket tags.
22. Before each model API call in the interactive loop, show a waiting indicator; in TTY mode keep it transient and clear it when the response arrives.
23. For tool-using turns, keep encoded history close to provider-native protocol items (for example OpenAI `reasoning`/`function_call`/`function_call_output`, Gemini `thought`/`function_call`/`function_response`) instead of flattening these into synthetic bracketed text messages.
24. For OpenAI stateless Responses mode (`store=false`), request `reasoning.encrypted_content` in `include` and replay reasoning/tool/message protocol items without reasoning-id references to avoid not-found errors.
25. For handoff pauses, use the next user input as the deferred `return_to_user` tool result and do not duplicate it as a second plain user text turn; for abort pauses, send deferred tool result plus the next user message.
26. For Gemini tool loops, preserve `thought_signature` across history replay, including signatures attached to `function_call` parts and signature-only thought parts, to stay aligned with Gemini function-calling protocol guidance.
27. Keep a dedicated runtime command to update and persist tool approval strategy (`ask`/`auto`) without full reconfiguration.
28. Keep the agent system prompt in a user-editable root file and expose a runtime command to print the loaded prompt with clear restart guidance.

## Testing Philosophy
1. Keep fast local tests for core logic (history, preprocessing, schema behavior).
2. Keep provider/network/UI tests separate and skippable when environment is missing.
3. Test what teaches the architecture, not every SDK edge-case.
4. When adding history display behavior, include `text_and_image` coverage so multimodal output remains easy to inspect.
5. Keep the local test runner easy to scope (for example a single-test selector) to support quick feedback loops during iteration.

## Code Review Heuristics
When polishing code, prioritize in this order:
1. Hidden complexity.
2. Unnecessary defensive logic.
3. Drift from architectural seams (base vs provider responsibility).
4. Readability for an educational audience.
5. Duplication that risks conceptual mismatch.

## Active Design Preferences
These are intentional project choices:
1. Generalize provider boundaries, not everything else.
2. Prefer explicit contracts over tolerant parsing.
3. Keep internal APIs small and concrete.
4. Add edge-case handling mainly at true external boundaries (SDK/network/OS).

## Provider Options Notes (Free/Open)
Keep this section as a short decision aid for adding test providers.

Current candidates to revisit first:
1. OpenRouter
2. Groq
3. Cloudflare Workers AI
4. Hugging Face Inference Providers

Selection bias for this project:
1. Prefer OpenAI-compatible APIs first (faster adapter development).
2. Prefer providers with stable free tier for educational experiments.
3. Prefer simpler auth/setup over max feature breadth.

Practical default:
1. Prototype with Groq or OpenRouter first.
2. Add one secondary provider only after the first adapter is clean.

Maintenance rule:
1. This section should store provider *direction* and *selection criteria*, not detailed pricing tables.
2. Re-check current free-tier limits before implementation decisions.

Model listing preference:
1. Keep a local constant catalog of common model IDs per provider for predictable CLI defaults.
2. Treat live model listing as optional runtime discovery, not the only source of model choices.
3. Avoid separate default-model constants when setup/config already requires explicit model selection.

## Near-Term Focus Areas
1. Simplify `tools/` exception strategy so failures are visible and educational.
2. Unify tool declaration/schema source of truth to reduce duplication.
3. Reduce defensive parsing patterns in model tool-call tests where SDK shape is stable.

## Collaboration Notes For Future Sessions
Before implementing, align quickly on:
1. Whether the change is a true extension seam or just local behavior.
2. Whether new complexity improves teachability.
3. Whether a fail-fast approach is acceptable for the call site.
4. As work progresses, update this file when new observations clarify what the user actually wants and what most directly supports the project goal.

If uncertain, bias toward the simpler implementation first, then iterate.

## Reusable Maintenance Prompt
Use this prompt whenever updating this file in a future session:

```text
Update `CODEBASE_POLISH_NOTES.md` as a long-lived collaboration/design playbook for tinyAgent.

Constraints:
1. Do not write a changelog or commit summary.
2. Keep only stable guidance that helps future work quality.
3. Capture coding style preferences, architecture intent, decision heuristics, and collaboration defaults.
4. Preserve or update project policies (for example: fail-fast internals, simple history API, default image downsize behavior).
5. Keep a short provider-options section with selection criteria and preferred starting options.
6. Remove stale or overly specific details that belong in git history instead.
7. While working, fold in concrete observations that improve understanding of user intent and improve alignment with project goals.

Output quality bar:
1. Clear, concise, and actionable.
2. Useful for someone starting a new task with zero chat context.
3. Focused on how we want to build, not on what we already changed.
```
