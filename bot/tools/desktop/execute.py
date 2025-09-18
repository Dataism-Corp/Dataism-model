import requests
import re
import base64
import os
import socket

def _detect_windows_host():
    # 1. Check env override
    ip = os.getenv("WINDOWS_HOST_IP")
    if ip:
        return ip.strip()

    # 2. Try host.docker.internal
    try:
        socket.gethostbyname("host.docker.internal")
        return "host.docker.internal"
    except socket.error:
        pass

    # 3. Parse /etc/resolv.conf (WSL gateway)
    try:
        with open("/etc/resolv.conf") as f:
            txt = f.read()
        m = re.search(r"nameserver\s+(\d+\.\d+\.\d+\.\d+)", txt)
        if m:
            return m.group(1)
    except Exception:
        pass

    # 4. Default fallback
    return "127.0.0.1"


WIN_IP = _detect_windows_host()
BASE_URL = f"http://{WIN_IP}:5001/desktop"


def execute(action: str, params: dict):
    """Low-level executor that directly calls the Windows agent."""
    try:
        url = f"{BASE_URL}/{action}"
        r = requests.get(url, params=params, timeout=15)
        data = r.json()

        # If screenshot → decode and save locally
        if action == "screenshot" and "b64" in data:
            img_bytes = base64.b64decode(data["b64"])
            os.makedirs("logs", exist_ok=True)
            out_path = os.path.join("logs", f"screenshot_{os.getpid()}.png")
            with open(out_path, "wb") as f:
                f.write(img_bytes)
            data["saved_to"] = out_path
            del data["b64"]

        return data
    except Exception as e:
        return {"error": str(e)}


def run_desktop_command(action: str, params: dict):
    """
    Entry point used by telegram_bot.
    Policy + approval logic is handled in telegram_bot, 
    so here we just forward to execute().
    """
    return execute(action, params)


if __name__ == "__main__":
    print("🔍 Windows Agent IP:", WIN_IP)
    print(run_desktop_command("type", {"text": "hello"}))
    print(run_desktop_command("screenshot", {}))
