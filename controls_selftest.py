import json
import time


def _wait_for(root, predicate, *, timeout_s: float = 2.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        root.update()
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("timeout waiting for expected event")


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
    press_combo("ctrl", "a", hold_ms=20)
    pyautogui.write("success", interval=0)
    _wait_for(root, lambda: entry.get() == "success")

    bx = ok_btn.winfo_rootx() + ok_btn.winfo_width() // 2
    by = ok_btn.winfo_rooty() + ok_btn.winfo_height() // 2
    move_mouse(bx, by)
    time.sleep(0.05)
    click(bx / width, by / height, "left")
    _wait_for(root, lambda: ok_clicked)


def test_tool_schemas() -> None:
    from controls import click, move_mouse, press_combo
    from tool_schema import make_tool_schema

    schemas = [make_tool_schema(move_mouse), make_tool_schema(click), make_tool_schema(press_combo)]
    print(json.dumps(schemas, indent=2))


def main() -> None:
    tests = [
        ("controls_gui", test_controls_gui),
        ("tool_schemas", test_tool_schemas),
    ]

    for name, fn in tests:
        print(f"== {name} ==")
        fn()
        print(f"✅ {name}")


if __name__ == "__main__":
    main()
