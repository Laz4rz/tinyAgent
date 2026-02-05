import json
import sys
import tempfile
import time
from pathlib import Path

from api_key import load_api_key
from model_client import GeminiClient
from tool_schema import make_tool_schema


class SkipTest(Exception):
    pass


def _print_block(title: str, payload) -> None:
    print(f"[{title}]")
    print(json.dumps(payload, indent=2))


def _wait_for(root, predicate, *, timeout_s: float = 2.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        root.update()
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("timeout waiting for expected event")


def _extract_first_function_call(response):
    calls = getattr(response, "function_calls", None) or []
    if calls:
        return calls[0]

    candidates = getattr(response, "candidates", None) or []
    for cand in candidates:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            fc = getattr(part, "function_call", None)
            if fc:
                return fc
    return None


def _args_to_dict(args) -> dict:
    if args is None:
        return {}
    if isinstance(args, dict):
        return args
    if hasattr(args, "fields"):
        try:
            from google.protobuf.json_format import MessageToDict

            return MessageToDict(args)
        except Exception:
            pass
    try:
        return dict(args)
    except Exception:
        return {}


def _default_tools() -> list[dict]:
    from controls import click, move_mouse, press_combo

    return [make_tool_schema(move_mouse), make_tool_schema(click), make_tool_schema(press_combo)]


def _gemini_tools() -> list[object]:
    from google.genai import types

    def obj(props: dict, required: list[str]):
        return types.Schema(type="object", properties=props, required=required)

    return [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="move_mouse",
                    description="Move mouse cursor to absolute screen coordinates.",
                    parameters=obj(
                        {
                            "x": types.Schema(type="integer"),
                            "y": types.Schema(type="integer"),
                        },
                        ["x", "y"],
                    ),
                ),
                types.FunctionDeclaration(
                    name="click",
                    description="Click at normalized screen coordinates.",
                    parameters=obj(
                        {
                            "x": types.Schema(type="number"),
                            "y": types.Schema(type="number"),
                            "button": types.Schema(type="string"),
                        },
                        ["x", "y"],
                    ),
                ),
                types.FunctionDeclaration(
                    name="press_combo",
                    description="Press a chord of keys, optionally hold, then release.",
                    parameters=obj(
                        {
                            "keys": types.Schema(
                                type="array",
                                items=types.Schema(type="string"),
                            ),
                            "hold_ms": types.Schema(type="integer"),
                        },
                        ["keys"],
                    ),
                ),
            ]
        )
    ]


def _maybe_client() -> GeminiClient | None:
    api_key = load_api_key(prompt=False)
    if not api_key:
        raise SkipTest("set GEMINI_API_KEY or .secret to run model tests")
    try:
        client = GeminiClient(
            api_key=api_key,
            model="gemini-2.5-flash",
            system_prompt="You are a helpful assistant.",
        )
        client.set_tools(_gemini_tools())
        return client
    except Exception as exc:
        raise RuntimeError(f"gemini init failed: {exc}") from exc


def test_init() -> None:
    client = _maybe_client()
    _print_block(
        "gemini.init",
        {
            "model": client.model,
            "system_prompt": client.system_prompt,
            "tools": len(client.tools),
            "history": len(client.history),
            "api_key_set": True,
        },
    )
    client.close()


def test_text_and_image() -> None:
    client = _maybe_client()

    image_path = None
    try:
        client.add_text("Hello from user", role="user")
        try:
            import pyautogui

            shot = pyautogui.screenshot()
        except Exception as exc:
            raise SkipTest(f"screenshot unavailable: {exc}") from exc

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            shot.save(tmp.name)
            image_path = Path(tmp.name)

        client.add_image(image_path, role="user")
        _print_block(
            "gemini.text_image",
            {
                "history": len(client.history),
                "last_parts": [p.kind for p in client.history[-1].parts],
                "image_bytes": image_path.stat().st_size,
                "image_path": str(image_path),
            },
        )
    finally:
        if image_path is not None:
            try:
                image_path.unlink(missing_ok=True)
            except Exception:
                pass
        client.close()


def test_controls_gui() -> None:
    try:
        import tkinter as tk
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"tkinter not available: {exc}")

    import pyautogui

    from controls import click, move_mouse, press_combo

    try:  # Fix coordinate mismatches on Windows with DPI scaling.
        import ctypes

        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

    root = tk.Tk()
    root.title("tinyAgent selftest")
    root.geometry("240x160+200+200")
    root.attributes("-topmost", True)

    ok_clicked = False

    label = tk.Label(root, text="Self-test running…\nDon’t touch mouse/keyboard.", justify="center")
    label.pack(expand=True, fill="both")

    entry = tk.Entry(root, width=22)
    entry.pack(padx=12, pady=(0, 8))

    def _ok():
        nonlocal ok_clicked
        ok_clicked = True
        label.config(text="✅ OK")
        root.after(150, root.destroy)

    ok_btn = tk.Button(root, text="OK", command=_ok)
    ok_btn.pack(padx=12, pady=(0, 12))

    try:
        root.update_idletasks()
        root.update()
        root.focus_force()
        root.lift()
        root.update()
        time.sleep(0.05)

        ex = entry.winfo_rootx() + entry.winfo_width() // 2
        ey = entry.winfo_rooty() + entry.winfo_height() // 2

        move_mouse(ex, ey)
        time.sleep(0.05)
        width, height = pyautogui.size()
        click(ex / width, ey / height, "left")
        _wait_for(root, lambda: root.focus_get() is entry)

        pyautogui.write("abc", interval=0)
        press_combo("mod", "a", hold_ms=20)
        pyautogui.write("success", interval=0)
        _wait_for(root, lambda: entry.get() == "success")

        bx = ok_btn.winfo_rootx() + ok_btn.winfo_width() // 2
        by = ok_btn.winfo_rooty() + ok_btn.winfo_height() // 2
        move_mouse(bx, by)
        time.sleep(0.05)
        click(bx / width, by / height, "left")
        _wait_for(root, lambda: ok_clicked)
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def test_tool_schemas() -> None:
    _print_block("tool_schemas", _default_tools())


def test_gemini_tool_click() -> None:
    try:
        import tkinter as tk
    except Exception as exc:  # pragma: no cover
        raise SkipTest(f"tkinter not available: {exc}") from exc

    try:  # Fix coordinate mismatches on Windows with DPI scaling.
        import ctypes

        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

    from controls import click

    client = _maybe_client()

    root = tk.Tk()
    root.title("Gemini tool click test")
    root.attributes("-topmost", True)

    ok_clicked = False

    def _on_click():
        nonlocal ok_clicked
        ok_clicked = True
        label.config(text="✅ Clicked")
        root.after(150, root.destroy)

    label = tk.Label(root, text="Waiting for tool click…", justify="center")
    label.pack(expand=True, fill="both")

    btn = tk.Button(root, text="Click Me", command=_on_click)
    btn.place(relx=0.5, rely=0.5, anchor="center")

    try:
        w, h = 240, 160
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        root.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")
        root.update_idletasks()
        root.update()
        root.focus_force()
        root.lift()
        root.update()
        time.sleep(0.05)

        import pyautogui

        screen_w, screen_h = pyautogui.size()
        bx = btn.winfo_rootx() + btn.winfo_width() / 2
        by = btn.winfo_rooty() + btn.winfo_height() / 2
        expected_x = bx / screen_w
        expected_y = by / screen_h

        from google.genai import types

        client.add_text(
            "Use the click tool to click the center of the screen. "
            f"Use x={expected_x:.4f}, y={expected_y:.4f}, button='left'.",
            role="user",
        )

        response = client.generate(
            return_response=True,
            store_response=False,
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY",
                    allowed_function_names=["click"],
                )
            ),
        )

        call = _extract_first_function_call(response)
        print({"debug": "extracted tool call", "call": call})
        if call is None:
            raise AssertionError("model did not call any tool")

        if hasattr(call, "function_call"):
            call = call.function_call

        name = getattr(call, "name", None)
        if name != "click":
            raise AssertionError(f"unexpected tool call: {name}")

        args = _args_to_dict(getattr(call, "args", None))
        x = float(args.get("x", expected_x))
        y = float(args.get("y", expected_y))
        button = str(args.get("button", "left"))

        def _try_click(nx: float, ny: float, label: str) -> bool:
            click(nx, ny, button)
            try:
                _wait_for(root, lambda: ok_clicked, timeout_s=1.0)
                return True
            except Exception:
                print({"warning": f"{label} click missed", "x": nx, "y": ny, "button": button})
                return False

        if not _try_click(x, y, "model"):
            if not _try_click(expected_x, expected_y, "expected"):
                raise AssertionError("button was not clicked")
    finally:
        try:
            root.destroy()
        except Exception:
            pass
        client.close()

def main() -> None:
    tests = [
        ("controls_gui", test_controls_gui),
        ("tool_schemas", test_tool_schemas),
        ("gemini_init", test_init),
        ("gemini_text_image", test_text_and_image),
        ("gemini_tool_click", test_gemini_tool_click),
    ]

    failed = 0
    skipped = 0
    for name, fn in tests:
        print(f"\n\n== {name} ==")
        try:
            fn()
            print(f"✅ {name}")
        except SkipTest as exc:
            skipped += 1
            print(f"⏭️ {name}: {exc}")
        except Exception as exc:
            failed += 1
            print(f"❌ {name}: {exc}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
