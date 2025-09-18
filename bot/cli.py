import os, json, requests, sys
from dotenv import load_dotenv

load_dotenv()
BEARER = os.getenv("BEARER_TOKEN", "changeme")
API = "http://127.0.0.1:8000"
HEAD = {"Authorization": f"Bearer {BEARER}", "Content-Type": "application/json"}

print("Dateria CLI. Type /clear to reset session, /exit to quit.\n")
sid = "cli"

while True:
    try:
        msg = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye.")
        break

    if not msg:
        continue

    if msg == "/exit":
        break

    if msg == "/clear":
        requests.post(f"{API}/session/reset")
        print("Session cleared.")
        continue

    payload = {"messages": [{"role": "user", "content": msg}], "session_id": sid, "stream": True}
    r = requests.post(f"{API}/v1/chat/completions", headers=HEAD, json=payload, stream=True, timeout=120)

    print("Dateria:", end=" ", flush=True)
    out = []

    for line in r.iter_lines():
        if line and b"delta" in line:
            try:
                data = json.loads(line.decode("utf-8").replace("data: ", ""))
                token = data.get("delta", "")
                out.append(token)
                # 👇 FIX: print without space between tokens
                print(token, end="", flush=True)
            except Exception:
                pass

    print("\n")
