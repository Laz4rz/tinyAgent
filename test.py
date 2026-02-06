# lock_test.py
import sys
import time
import pyautogui

win_key = sys.argv[1] if len(sys.argv) > 1 else "win"
print("KEY:", win_key)
print("Starting in 3s...")
time.sleep(3)

pyautogui.keyDown(win_key)
pyautogui.press("l")
pyautogui.keyUp(win_key)

print("Dispatched.")