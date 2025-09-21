# tools/kill.py
_kill_flag = False

def trigger_kill():
    global _kill_flag
    _kill_flag = True
    return {"status": "killed", "note": "Kill flag set"}

def reset_kill():
    global _kill_flag
    _kill_flag = False
    return {"status": "reset", "note": "Kill flag cleared"}

def check_kill():
    return _kill_flag
