import time
import sys

from .helpers import modifier_keys, normalized_to_pixels, release_keys, resolve_combo_key


def move_mouse(x: float, y: float) -> str:
    """Move mouse cursor to normalized screen coordinates.

    Args:
        x: Normalized X in [0.0, 1.0] across screen width.
        y: Normalized Y in [0.0, 1.0] across screen height.
    """
    try:
        import pyautogui

        px, py = normalized_to_pixels(x, y)
        pyautogui.moveTo(px, py)
        return f"Successfully moved mouse to ({x}, {y})"
    except Exception as exc:
        return f"Failed to move mouse to ({x}, {y}): {exc}"


def click(x: float, y: float, button: str = "left") -> str:
    """Click at normalized screen coordinates.

    Args:
        x: Normalized X in [0.0, 1.0] across screen width.
        y: Normalized Y in [0.0, 1.0] across screen height.
        button: "left" or "right" (case-insensitive). Defaults to "left".
    """
    try:
        import pyautogui

        px, py = normalized_to_pixels(x, y)
        pyautogui.click(x=px, y=py, button=button.lower())
        return f"Successfully clicked {button} at ({x}, {y})"
    except Exception as exc:
        return f"Failed to click {button} at ({x}, {y}): {exc}"


def press_combo(*keys: str, hold_ms: int = 0) -> str:
    """Press a chord of keys, optionally hold, then release in reverse order.

    Args:
        *keys: Key names understood by pyautogui (e.g., "ctrl", "shift", "a").
            You can also use "mod" as a cross-platform alias for the primary
            shortcut modifier ("command" on macOS, "ctrl" elsewhere).
        hold_ms: Milliseconds to hold keys down before releasing.

    Example:
        press_combo("ctrl", "a", hold_ms=50)
    """
    try:
        import pyautogui
    except Exception as exc:
        return f"Failed to press combo {', '.join(keys)}: {exc}"

    pressed: list[str] = []
    error_message: str | None = None
    try:
        for key in keys:
            lowered = resolve_combo_key(key, platform_name=sys.platform)
            pyautogui.keyDown(lowered)
            pressed.append(lowered)
        if hold_ms:
            time.sleep(hold_ms / 1000)
    except Exception as exc:
        error_message = str(exc)
    finally:
        release_targets = list(dict.fromkeys([*reversed(pressed), *modifier_keys()]))
        release_errors = release_keys(pyautogui, release_targets)

    combo = ", ".join(keys)
    if error_message:
        if release_errors:
            return (
                f"Failed to press combo {combo}: {error_message}. "
                f"Key release errors: {'; '.join(release_errors)}"
            )
        return f"Failed to press combo {combo}: {error_message}"

    if release_errors:
        return f"Failed to fully release combo {combo}: {'; '.join(release_errors)}"

    return f"Successfully pressed combo: {combo}"


def return_to_user(message: str) -> str:
    """Return control to the user with a final message or question.

    Args:
        message: What the user should see when the agent yields control.
    """
    return message
