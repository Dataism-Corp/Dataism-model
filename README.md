# Telegram Qwen Bot

A powerful Telegram bot powered by Qwen AI models for intelligent chat and code generation. Features multi-modal processing, web search, file handling, and desktop automation capabilities.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM (16GB+ recommended)

### 1. Clone and Setup
```bash
git clone <repository-url>
cd Dataism-model
```

### 2. Environment Configuration
```bash
# Copy environment template
cp .env.example bot/Telegram/.env

# Edit the environment file with your credentials
nano bot/Telegram/.env  # or use your preferred editor
```

**Required Variables:**
- `TELEGRAM_BOT_TOKEN` - Get from [@BotFather](https://t.me/botfather)
- `BEARER_TOKEN` - Generate a secure random token
- `ADMIN_CHAT_ID` - Your Telegram user ID (get from [@userinfobot](https://t.me/userinfobot))

### 3. Install Dependencies
```bash
# Install Python packages
pip install -r bot/Telegram/requirements.txt

# Or use the provided setup script
bash scripts/setup.sh
```

### 4. Initialize Database
```bash
python bot/Telegram/init_db.py
```

### 5. Start the Bot
```bash
# Quick verification (recommended first)
bash scripts/verify_bot.sh

# Start all services
bash scripts/supervisor.sh
```

The bot will be available at your configured Telegram bot token, and the API will run on `http://127.0.0.1:8000`.

## 🧪 Verification

Run the smoke test to ensure everything is configured correctly:

```bash
bash scripts/verify_bot.sh
```

This will check:
- ✅ Project structure
- ✅ Environment configuration  
- ✅ Python dependencies
- ✅ Code syntax validation
- ✅ Database setup
- ✅ Startup scripts

## 🤖 Bot Features

### Chat Commands
- `/start` - Initialize bot
- `/clear` - Reset conversation history
- `/verify phase2` - Run system verification

### Capabilities
- 🧠 **Intelligent Chat** - Powered by Qwen-2.5-14B-Instruct
- 💻 **Code Generation** - Specialized Qwen-2.5-Coder-14B model
- 📄 **Document Processing** - PDF, images, and file uploads
- 🌐 **Web Search** - Google Custom Search integration
- 🖼️ **Image Analysis** - OCR and image understanding
- 🔧 **Desktop Automation** - System control capabilities

## 📁 Project Structure

```
├── bot/Telegram/           # Core bot implementation
│   ├── main_api.py        # FastAPI web server
│   ├── bots/telegram_bot.py  # Telegram bot interface
│   ├── model_chat.py      # Chat model integration
│   ├── model_code.py      # Code model integration
│   ├── tools/             # External integrations
│   └── .env              # Environment configuration
├── scripts/               # Startup and utility scripts
│   ├── setup.sh          # Dependency installation
│   ├── supervisor.sh     # Service startup
│   └── verify_bot.sh     # Smoke testing
├── docs/                  # Documentation
│   └── setup-guide.md    # Detailed setup instructions
└── .env.example          # Environment template
```

## 🔧 Configuration

### Model Configuration
The bot uses two specialized models:
- **Chat Model**: `Qwen/Qwen2.5-14B-Instruct` - General conversation
- **Code Model**: `Qwen/Qwen2.5-Coder-14B` - Code generation and analysis

Models are automatically downloaded from Hugging Face on first use.

### Hardware Requirements
- **Minimum**: 8GB RAM, CPU-only (slower)
- **Recommended**: 16GB+ RAM, CUDA GPU
- **Optimal**: 24GB+ GPU memory for full model loading

### Optional Integrations
- **Google Search** - Requires Google API key and Custom Search Engine ID
- **YouTube Processing** - Requires YouTube Data API key
- **Desktop Control** - Linux/macOS for automation features

## 🚨 Troubleshooting

### Common Issues

**Bot doesn't respond to commands:**
```bash
# Check bot token and restart
tail -f bot/Telegram/logs/bot.log
```

**Models loading slowly:**
```bash
# Monitor GPU usage
nvidia-smi

# Check model download progress
tail -f bot/Telegram/logs/api.log
```

**Database errors:**
```bash
# Reinitialize database
rm bot/Telegram/memory.db
python bot/Telegram/init_db.py
```

**Permission denied on scripts:**
```bash
# Make scripts executable
chmod +x scripts/*.sh
```

### Performance Optimization

1. **GPU Acceleration**: Ensure CUDA is properly installed
2. **Model Caching**: Models are cached after first download
3. **Memory Management**: Bot automatically manages context windows
4. **Concurrent Requests**: API supports multiple simultaneous users

## 📚 Documentation

- **[Setup Guide](docs/setup-guide.md)** - Comprehensive installation instructions
- **[API Documentation](http://127.0.0.1:8000/docs)** - Interactive API docs (when running)
- **Environment Variables** - See `.env.example` for all options

## 🤝 Development

### Running in Development Mode
```bash
# Start API with hot reload
cd bot/Telegram
uvicorn main_api:app --reload --host 127.0.0.1 --port 8000

# Run bot in separate terminal
python bots/telegram_bot.py
```

### Testing
```bash
# Run verification suite
python bot/Telegram/verify.py

# Quick smoke test
bash scripts/verify_bot.sh
```

## 📄 License

[License information to be added]

---

Built with ❤️ using Qwen AI models and modern Python frameworks.

10-phase AI system project.  
Includes verification tools, phases 1–10 code, and documentation.  

---

## 🚀 Current Status
- **Phases Completed:** Phase 1 + Phase 2 (in progress verification)  
- **Next Active Phase:** Phase 3 – engineers will start from here  
- **Hardware Environment:** NVIDIA 5090 GPU (local machine)  
- **Verification:** Dedicated verification tool required before merging any work  

---

## 📌 Project Overview
The Dataism-model is designed as a **ChatGPT-style system** running fully on a local GPU.  
This project is structured into **10 Phases** for controlled development, scaling, and verification.  

---

## ✅ Verification Standards
Before any engineer can contribute, they must:  
1. **Pass the verification test** (building a working verification tool).  
2. **Push changes through GitHub repo** – no direct access to local system.  
3. Ensure all phases can be validated with **PASS/FAIL outputs** and logs.  

---

## 📂 Phases Breakdown (High Level)
- **Phase 1:** Core repo + environment setup  
- **Phase 2:** Verification tool prototype  
- **Phase 3:** Expanded verification + tool refinement  
- **Phase 4–8:** System scaling, self-improving loops, reinforcement, reasoning layers  
- **Phase 9:** Autonomous engineering capability (system can evolve itself)  
- **Phase 10:** Elite self-upgrading, ROI factory, and post-human scale  

---

## 📖 Contribution Rules
- All commits must be made via **pull requests**.  
- Verification tool must confirm results before merge.  
- Code must be modular and well-documented.  

---

## 🔑 Notes for Engineers
- Expected dedication: **30+ hours per week**.  
- Communication: Clear progress updates + verification logs.  
- Goal: Move fast but **never sacrifice verification and trust** in outputs.  

---

## 🏁 Vision
The end result will be a **self-evolving AI system** capable of natural language reasoning,  
autonomous upgrades, and ROI-driven decision making — running 24/7 on local GPUs.
