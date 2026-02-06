from __future__ import annotations

import base64
import json
from datetime import datetime
from pathlib import Path
from typing import Any


class DebugRecorder:
    def __init__(self, *, root: str | Path = "debug") -> None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = Path(root) / f"run-{stamp}"
        self.screenshots_dir = self.run_dir / "screenshots"
        self.request_log_path = self.run_dir / "history_requests.json"
        self.response_log_path = self.run_dir / "responses.json"
        self.events_log_path = self.run_dir / "events.log"
        self._requests: list[dict[str, Any]] = []
        self._responses: list[dict[str, Any]] = []
        self._turn = 0

        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        _write_json(self.request_log_path, self._requests)
        _write_json(self.response_log_path, self._responses)

    def next_turn(self) -> int:
        self._turn += 1
        return self._turn

    def capture_screenshot(self, *, turn: int) -> str | None:
        try:
            import pyautogui
        except Exception as exc:
            self._log_event(f"turn {turn}: screenshot unavailable: {exc}")
            return None

        path = self.screenshots_dir / f"turn_{turn:04d}.png"
        try:
            image = pyautogui.screenshot()
            image.save(path)
        except Exception as exc:
            self._log_event(f"turn {turn}: screenshot failed: {exc}")
            return None
        return str(path)

    def record_request(
        self,
        *,
        turn: int,
        provider: str,
        model: str,
        encoded_history: Any,
        request_config: Any,
        screenshot_path: str | None,
    ) -> None:
        payload = {
            "turn": turn,
            "provider": provider,
            "model": model,
            "screenshot_path": screenshot_path,
            "encoded_history": _to_jsonable(encoded_history),
            "request_config": _to_jsonable(request_config),
        }
        self._requests.append(payload)
        _write_json(self.request_log_path, self._requests)

    def record_response(self, *, turn: int, response: Any) -> None:
        payload = {
            "turn": turn,
            "response": _to_jsonable(response),
        }
        self._responses.append(payload)
        _write_json(self.response_log_path, self._responses)

    def _log_event(self, text: str) -> None:
        self.events_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_log_path.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        encoded = base64.b64encode(value).decode("ascii")
        return {"__bytes_base64__": encoded}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]

    if hasattr(value, "model_dump_json"):
        dumped = value.model_dump_json()
        return json.loads(dumped)

    if hasattr(value, "model_dump"):
        return _to_jsonable(value.model_dump())

    if hasattr(value, "to_dict"):
        return _to_jsonable(value.to_dict())

    if hasattr(value, "__dict__"):
        return _to_jsonable(vars(value))

    return repr(value)
