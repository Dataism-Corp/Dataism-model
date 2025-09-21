#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# load .env
set -a
source ./.env
set +a

# logs
mkdir -p logs
API_LOG=logs/api.log
BOT_LOG=logs/bot.log

# kill previous
pkill -f "uvicorn main_api:app" || true
pkill -f "bots/telegram_bot.py" || true

# start api
nohup uvicorn main_api:app --host 127.0.0.1 --port 8000 --reload > "$API_LOG" 2>&1 &

# wait a moment
sleep 2

# start bot
nohup python bots/telegram_bot.py > "$BOT_LOG" 2>&1 &

echo "Started. API on 127.0.0.1:8000 | Logs: $API_LOG, $BOT_LOG"
