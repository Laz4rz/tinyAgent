def normalized_to_pixels(x: float, y: float) -> tuple[int, int]:
    import pyautogui

    width, height = pyautogui.size()
    return int(float(x) * width), int(float(y) * height)
