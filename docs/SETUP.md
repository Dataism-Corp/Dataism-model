# Telegram Qwen Bot Setup Guide

This guide provides detailed instructions for setting up the Telegram Qwen Bot, including model weight placement, configuration, and deployment options.

## Table of Contents
- [System Requirements](#system-requirements)
- [Model Weight Setup](#model-weight-setup)
- [Environment Configuration](#environment-configuration)
- [Installation Methods](#installation-methods)
- [Configuration Options](#configuration-options)
- [Running the Bot](#running-the-bot)
- [Deployment Options](#deployment-options)
- [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements
- **OS**: Linux, macOS, or Windows (with WSL recommended)
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum
- **Storage**: 50GB free space (for model weights)
- **Network**: Stable internet connection for initial model download

### Recommended Requirements
- **RAM**: 16GB or higher
- **GPU**: CUDA-compatible GPU with 8GB+ VRAM
- **Storage**: 100GB+ SSD storage
- **CPU**: Modern multi-core processor

### Optimal Performance
- **GPU**: 24GB+ VRAM (RTX 4090, A100, etc.)
- **RAM**: 32GB+ system memory
- **Storage**: NVMe SSD with high read speeds

## Model Weight Setup

### Automatic Download (Recommended)
The bot will automatically download model weights on first use:

1. **Models Used**:
   - Chat: `Qwen/Qwen2.5-14B-Instruct` (~28GB)
   - Code: `Qwen/Qwen2.5-Coder-14B` (~28GB)

2. **Default Cache Location**:
   - Linux/macOS: `~/.cache/huggingface/transformers/`
   - Windows: `%USERPROFILE%\.cache\huggingface\transformers\`

3. **Disk Space Requirements**:
   - Total: ~60GB for both models
   - Additional: ~10GB for tokenizers and configuration files

### Manual Model Download (Advanced)
For offline setups or custom locations:

```bash
# Install huggingface-hub
pip install huggingface-hub

# Download models manually
python -c "
from transformers import AutoModel, AutoTokenizer
AutoModel.from_pretrained('Qwen/Qwen2.5-14B-Instruct')
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-14B-Instruct')
AutoModel.from_pretrained('Qwen/Qwen2.5-Coder-14B')
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-14B')
"
```

### Custom Model Paths
To use models from custom locations, update your `.env` file:

```bash
# Use local model paths
MODEL_CHAT=/path/to/local/qwen-chat-model
MODEL_CODE=/path/to/local/qwen-code-model

# Use different Hugging Face models
MODEL_CHAT=microsoft/DialoGPT-large
MODEL_CODE=Salesforce/codegen-2B-multi
```

## Environment Configuration

### Step 1: Copy Environment Template
```bash
cp .env.example bot/Telegram/.env
```

### Step 2: Configure Required Variables

#### Essential Configuration
```bash
# Bot Authentication - REQUIRED
TELEGRAM_BOT_TOKEN=your_bot_token_from_botfather
BEARER_TOKEN=generate_a_secure_random_string
ADMIN_CHAT_ID=your_telegram_user_id

# Model Configuration - REQUIRED
MODEL_CHAT=Qwen/Qwen2.5-14B-Instruct
MODEL_CODE=Qwen/Qwen2.5-Coder-14B
```

#### Optional Integrations
```bash
# Google Search (for web search capabilities)
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_custom_search_engine_id

# YouTube (for video processing)
YOUTUBE_API_KEY=your_youtube_data_api_key

# Desktop Automation
AUTHORIZED_DESKTOP_CHAT_IDS=user_id_1,user_id_2
```

### Step 3: Obtain Required Tokens

#### Telegram Bot Token
1. Message [@BotFather](https://t.me/botfather) on Telegram
2. Send `/newbot` command
3. Follow prompts to create your bot
4. Copy the token provided

#### Admin Chat ID
1. Message [@userinfobot](https://t.me/userinfobot) on Telegram
2. Copy your user ID from the response

#### Bearer Token
Generate a secure random string:
```bash
# Linux/macOS
openssl rand -hex 32

# Python
python -c "import secrets; print(secrets.token_hex(32))"
```

## Installation Methods

### Method 1: Automated Setup (Recommended)
```bash
# Clone repository
git clone <repository-url>
cd Dataism-model

# Run automated setup
bash scripts/setup.sh

# Configure environment
cp .env.example bot/Telegram/.env
# Edit bot/Telegram/.env with your credentials

# Initialize database
python bot/Telegram/init_db.py

# Verify setup
bash scripts/verify_bot.sh
```

### Method 2: Manual Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r bot/Telegram/requirements.txt

# Configure environment (as above)
cp .env.example bot/Telegram/.env

# Initialize database
python bot/Telegram/init_db.py
```

### Method 3: Docker Setup (Advanced)
```dockerfile
# Create Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . .

RUN pip install -r bot/Telegram/requirements.txt

EXPOSE 8000
CMD ["bash", "scripts/supervisor.sh"]
```

```bash
# Build and run
docker build -t telegram-qwen-bot .
docker run -p 8000:8000 -v ./bot/Telegram/.env:/app/bot/Telegram/.env telegram-qwen-bot
```

## Configuration Options

### Model Configuration
Customize model behavior in your `.env`:

```bash
# Model-specific settings
MAX_NEW_TOKENS=2048        # Maximum response length
TEMPERATURE=0.7            # Response creativity (0.0-1.0)
TOP_P=0.9                 # Nucleus sampling parameter
REPETITION_PENALTY=1.1    # Avoid repetitive responses

# Hardware optimization
DEVICE=auto               # auto, cpu, cuda:0, etc.
LOW_CPU_MEM_USAGE=true   # Memory optimization
TORCH_DTYPE=float16      # Use half precision
```

### API Configuration
```bash
# Server settings
API_HOST=127.0.0.1       # API bind address
API_PORT=8000            # API port
API_WORKERS=1            # Number of worker processes

# Security
CORS_ORIGINS=*           # CORS allowed origins
MAX_REQUEST_SIZE=100MB   # Maximum upload size
```

### Bot Behavior
```bash
# Response settings
MAX_CONTEXT_MESSAGES=20  # Conversation context length
STREAM_RESPONSES=true    # Enable streaming responses
AUTO_CLEAR_CONTEXT=false # Auto-clear after inactivity

# Features
ENABLE_WEB_SEARCH=true   # Enable Google search
ENABLE_FILE_UPLOAD=true  # Enable file processing
ENABLE_DESKTOP_CONTROL=false  # Enable automation features
```

## Running the Bot

### Quick Start
```bash
# Verify everything is setup correctly
bash scripts/verify_bot.sh

# Start all services
bash scripts/supervisor.sh
```

### Development Mode
```bash
# Start API with hot reload
cd bot/Telegram
uvicorn main_api:app --reload --host 0.0.0.0 --port 8000

# In a separate terminal, start bot
python bots/telegram_bot.py
```

### Production Mode
```bash
# Start with process manager
bash scripts/supervisor.sh

# Or use systemd service (Linux)
sudo systemctl enable telegram-qwen-bot
sudo systemctl start telegram-qwen-bot
```

### Service Management
```bash
# Check service status
curl http://127.0.0.1:8000/health

# View logs
tail -f bot/Telegram/logs/api.log
tail -f bot/Telegram/logs/bot.log

# Stop services
pkill -f "uvicorn main_api:app"
pkill -f "telegram_bot.py"
```

## Deployment Options

### Local Development
- Run on localhost for testing
- Use development configuration
- Manual process management

### VPS/Cloud Server
- Deploy on Linux VPS
- Use systemd for process management
- Configure firewall for API port
- Set up reverse proxy (nginx/caddy)

### Container Deployment
- Docker containerization
- Kubernetes orchestration
- Automatic scaling and recovery

### Example Nginx Configuration
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Troubleshooting

### Common Issues

#### Models Won't Load
```bash
# Check available disk space
df -h

# Check GPU memory
nvidia-smi

# Force CPU mode
echo "DEVICE=cpu" >> bot/Telegram/.env
```

#### Bot Not Responding
```bash
# Check bot token
grep TELEGRAM_BOT_TOKEN bot/Telegram/.env

# Verify network connectivity
curl -s https://api.telegram.org/bot<YOUR_TOKEN>/getMe

# Check logs
tail -f bot/Telegram/logs/bot.log
```

#### API Errors
```bash
# Check API status
curl http://127.0.0.1:8000/health

# Restart API
pkill -f "uvicorn main_api:app"
cd bot/Telegram && uvicorn main_api:app --host 127.0.0.1 --port 8000
```

#### Memory Issues
```bash
# Monitor memory usage
free -h
htop

# Enable memory optimization
echo "LOW_CPU_MEM_USAGE=true" >> bot/Telegram/.env
echo "TORCH_DTYPE=float16" >> bot/Telegram/.env
```

### Performance Optimization

#### GPU Acceleration
1. Install CUDA toolkit
2. Install PyTorch with CUDA support:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
3. Verify GPU availability:
   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.cuda.get_device_name(0))
   ```

#### Memory Management
- Use model sharding for large models
- Enable gradient checkpointing
- Use mixed precision training
- Adjust batch sizes based on available memory

#### Response Speed
- Pre-load models during startup
- Use model caching
- Optimize tokenization
- Enable response streaming

### Getting Help

- **Logs**: Check `bot/Telegram/logs/` for detailed error information
- **Verification**: Run `bash scripts/verify_bot.sh` for diagnostic information
- **Documentation**: Review the main README.md for additional guidance
- **Community**: Check the project's issue tracker for known problems

## Security Considerations

### Token Security
- Keep your `.env` file secure and never commit it to version control
- Use strong, unique tokens for all services
- Regularly rotate API keys and tokens

### Network Security
- Run API on localhost unless external access is needed
- Use HTTPS in production deployments
- Configure firewall rules appropriately

### Bot Permissions
- Limit admin privileges to trusted users only
- Review and audit desktop automation features
- Monitor bot usage and logs regularly

---

This setup guide should get you up and running with the Telegram Qwen Bot. For additional support, refer to the main documentation or project repository.