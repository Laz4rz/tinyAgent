def normalized_to_pixels(x: float, y: float) -> tuple[int, int]:
    import pyautogui

    width, height = pyautogui.size()
    return int(float(x) * width), int(float(y) * height)


def resolve_combo_key(key: str, *, platform_name: str) -> str:
    lowered = key.lower()
    if lowered == "mod":
        return "command" if platform_name == "darwin" else "ctrl"
    if lowered == "cmd":
        return "command"
    return lowered


def unsupported_combo_keys(pyautogui, keys: list[str]) -> list[str]:
    try:
        keyboard_keys = pyautogui.KEYBOARD_KEYS
    except AttributeError:
        return []

    supported = {str(key).lower() for key in keyboard_keys}
    return [key for key in keys if key not in supported]


def modifier_keys() -> tuple[str, ...]:
    return ("shift", "ctrl", "alt", "command", "option", "win")


def release_keys(pyautogui, keys: list[str], *, max_attempts: int = 2) -> list[str]:
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    release_errors: list[str] = []
    for key in keys:
        last_error: str | None = None
        for _ in range(max_attempts):
            error = _safe_key_up(pyautogui, key)
            if error is None:
                last_error = None
                break
            last_error = error
        if last_error is not None:
            release_errors.append(f"{key}: {last_error}")
    return release_errors


def _safe_key_up(pyautogui, key: str) -> str | None:
    has_failsafe = hasattr(pyautogui, "FAILSAFE")
    if has_failsafe:
        original_failsafe = pyautogui.FAILSAFE
        pyautogui.FAILSAFE = False
    try:
        pyautogui.keyUp(key)
        return None
    except Exception as exc:
        return str(exc)
    finally:
        if has_failsafe:
            pyautogui.FAILSAFE = original_failsafe
