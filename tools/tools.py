import time
import sys

from .helpers import normalized_to_pixels


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
    pressed: list[str] = []
    try:
        import pyautogui

        for key in keys:
            lowered = key.lower()
            if lowered == "mod":
                lowered = "command" if sys.platform == "darwin" else "ctrl"
            elif lowered == "cmd":
                lowered = "command"
            pyautogui.keyDown(lowered)
            pressed.append(lowered)
        if hold_ms:
            time.sleep(hold_ms / 1000)
    except Exception as exc:
        return f"Failed to press combo {', '.join(keys)}: {exc}"
    finally:
        for key in reversed(pressed):
            pyautogui.keyUp(key)
    return f"Successfully pressed combo: {', '.join(keys)}"


def return_to_user(message: str) -> str:
    """Return control to the user with a final message or question.

    Args:
        message: What the user should see when the agent yields control.
    """
    return message
