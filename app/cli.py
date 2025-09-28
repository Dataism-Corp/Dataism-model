
#!/usr/bin/env python3
# app/cli.py

import argparse
import requests
import time

# --- Hardcoded Telegram credentials (for testing only) ---
BOT_TOKEN = "8266904564:AAHaPMqWZKQEq-cXi-NBuOGFB8J4DpK4s9Y"
CHAT_ID = "6520801941"   # your user ID

BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"


def cmd_chat(args):
    """Simple test: send a message to your chat via Telegram."""
    text = args.prompt
    r = requests.post(f"{BASE_URL}/sendMessage", json={"chat_id": CHAT_ID, "text": text})
    print(r.json())


def cmd_serve_telegram(args):
    """Basic polling loop to echo messages."""
    offset = None
    print("[info] Bot started. Ctrl+C to stop.")
    while True:
        try:
            params = {"timeout": 20}
            if offset:
                params["offset"] = offset
            updates = requests.get(f"{BASE_URL}/getUpdates", params=params, timeout=25).json()

            for u in updates.get("result", []):
                offset = u["update_id"] + 1
                message = u.get("message")
                if not message:
                    continue
                chat = message["chat"]["id"]
                text = (message.get("text") or "").strip()

                if text == "/ping":
                    reply = "pong 🟢"
                else:
                    reply = f"echo: {text}"

                requests.post(f"{BASE_URL}/sendMessage", json={"chat_id": chat, "text": reply})

        except KeyboardInterrupt:
            print("\n[info] Bot stopped.")
            break
        except Exception as e:
            print(f"[warn] Error: {e}")
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_chat = sub.add_parser("chat", help="Send a message through Telegram")
    p_chat.add_argument("prompt", help="Message to send")
    p_chat.set_defaults(func=cmd_chat)

    p_bot = sub.add_parser("serve-telegram", help="Run polling bot")
    p_bot.set_defaults(func=cmd_serve_telegram)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
