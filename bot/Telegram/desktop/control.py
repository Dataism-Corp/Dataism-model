import pyautogui, time
def open_notepad():
    pyautogui.hotkey("win","r")
    time.sleep(1)
    pyautogui.typewrite("notepad\n")
    time.sleep(1)
    pyautogui.typewrite("Hello from Dateria!\n")
