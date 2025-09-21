# tools/desktop/policy.py
WHITELISTED_PATHS = [r"C:\Users\Admin\Desktop", "/home/dateria/project"]
ALLOWED_APPS = ["notepad.exe", "calc.exe"]

def check(action, args, user_id):
    """
    Return 'allow', 'ask', or 'deny'.
    Simple rules:
      - read/scroll/screenshot => allow
      - run/write/install => ask
      - unknown => ask
    """
    if action in ("type","click","hotkey","screenshot","read"):
        return "allow"
    if action in ("run","install","write","file.write"):
        return "ask"
    return "ask"
