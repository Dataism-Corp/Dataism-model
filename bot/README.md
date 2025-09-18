# Dateria AI Assistant

A comprehensive AI-powered assistant platform with multi-modal capabilities, intelligent routing, and extensive tool integrations. Features both API and Telegram bot interfaces with support for chat, code generation, document processing, and system automation.

## 🚀 Features

### Core AI Capabilities
- **Dual AI Models**: Separate optimized models for chat (`model_chat.py`) and code generation (`model_code.py`)
- **Intelligent Routing**: Smart request routing based on content analysis using configurable rules (`router.yaml`)
- **Streaming Responses**: Real-time token streaming for interactive conversations
- **Session Management**: Persistent conversation history with SQLite storage

### Multi-Modal Processing
- **Document Processing**: PDF text extraction and analysis
- **Image Processing**: OCR and image understanding capabilities
- **Web Content**: URL scraping and content extraction
- **File Upload**: Support for various file formats

### Interfaces
- **REST API**: FastAPI-based service with OpenAI-compatible endpoints
- **Telegram Bot**: Full-featured bot with command support and media handling
- **CLI Client**: Command-line interface for direct interaction
- **Desktop Integration**: System automation and desktop control capabilities

### Advanced Features
- **RAG System**: Document storage and retrieval for knowledge augmentation
- **Verification System**: Comprehensive testing framework with evidence collection
- **Tool Integration**: Extensible tool system for external API calls
- **Memory Management**: Intelligent context and session handling

## 📋 Requirements

### System Requirements
- Python 3.8+ 
- CUDA-compatible GPU (recommended for optimal performance)
- 8GB+ RAM (16GB+ recommended for larger models)

### Dependencies
The project uses modern ML and web frameworks:

```
fastapi              # Web API framework
uvicorn[standard]    # ASGI server
transformers         # Hugging Face model loading
torch                # PyTorch ML framework
python-dotenv        # Environment configuration
telebot              # Telegram bot API
pdfplumber           # PDF processing
pytesseract          # OCR capabilities
trafilatura          # Web content extraction
sentence-transformers # Embeddings for RAG
```

See `requirements.txt` for complete dependency list.

## 🛠️ Installation

### Quick Setup

1. **Clone and Navigate**
   ```bash
   git clone <repository-url>
   cd dateria-project
   ```

2. **Environment Setup**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configuration**
   Create `.env` file:
   ```env
   BEARER_TOKEN=your_secure_token_here
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token
   MODEL_CHAT=Qwen/Qwen2.5-14B-Instruct
   MODEL_CODE=your_code_model_id
   ```

5. **Initialize Database**
   ```bash
   python init_db.py
   ```

### Alternative Setup (Ubuntu/WSL)
Use the provided setup script:
```bash
chmod +x setup.sh
./setup.sh
```

## 🚀 Usage

### API Server

Start the FastAPI server:
```bash
python main_api.py
# or
uvicorn main_api:app --host 0.0.0.0 --port 8000
```

The API provides OpenAI-compatible endpoints:
- `POST /v1/chat/completions` - Chat completions with streaming
- `POST /session/reset` - Clear conversation history
- `POST /upload` - File upload and processing

### CLI Client

Interactive command-line interface:
```bash
python cli.py
```

Commands:
- `/clear` - Reset conversation session
- `/exit` - Quit the CLI
- Any other input - Send message to AI

### Telegram Bot

Start the Telegram bot:
```bash
python bots/telegram_bot.py
```

Bot commands:
- `/start` - Initialize bot interaction
- `/verify phase2` - Run verification tests
- `/clear` - Clear conversation history
- Send files, images, or URLs for processing

### Verification System

Run comprehensive tests:
```bash
# Run all available verification phases
python verify.py

# Run specific phase
python verify.py --phase 2

# Quick verification for development
python verification_bot.py
```

## 🏗️ Architecture

### Core Components

```
├── main_api.py          # FastAPI web server
├── cli.py               # Command-line interface
├── model_chat.py        # Chat model integration
├── model_code.py        # Code generation model
├── verification_bot.py  # Verification command handling
├── verify.py           # Comprehensive testing framework
├── rag_store.py        # Document storage and retrieval
└── router.yaml         # Request routing configuration
```

### Bot Integration
```
bots/
└── telegram_bot.py     # Telegram bot implementation
```

### Tools and Utilities
```
tools/
├── apis.py             # External API integrations
├── media.py            # Media processing utilities
├── kill.py             # Process management
└── desktop/            # Desktop automation tools
```

### Data Storage
```
├── memory.db           # SQLite database for sessions
├── evidence_bundles/   # Verification test results
├── verification_history/ # Historical test data
└── logs/              # Application logs
```

## 🔧 Configuration

### Router Configuration (`router.yaml`)
Controls how requests are routed between chat and code models:

```yaml
overrides:
  force_chat: []      # Patterns to force chat model
  force_code: []      # Patterns to force code model
rules:
  code_regex:         # Patterns indicating code requests
    - "```"
    - "\\b(def|class)\\b"
    - "\\b(SELECT|INSERT|UPDATE|DELETE)\\b"
thresholds:
  min_tokens_for_code: 6
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `BEARER_TOKEN` | API authentication token | `changeme` |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token | - |
| `MODEL_CHAT` | Chat model identifier | `Qwen/Qwen2.5-14B-Instruct` |
| `MODEL_CODE` | Code model identifier | - |

## 🧪 Testing and Verification

The project includes a comprehensive verification system ensuring quality and reliability:

### Verification Phases
- **Phase 2**: Core functionality verification
- **Evidence Collection**: Detailed test execution logs
- **Independent Validation**: Bulletproof testing methodology

### Running Tests
```bash
# Full verification suite
python verify.py

# Phase-specific testing
python verify.py --phase 2

# Development verification
python verification_bot.py
```

### Test Coverage
- API endpoint functionality
- Model integration and responses
- Database schema and operations
- File processing capabilities
- Bot command handling
- System integration tests

## 📝 API Reference

### Chat Completions
```http
POST /v1/chat/completions
Content-Type: application/json
Authorization: Bearer your_token

{
  "messages": [
    {"role": "user", "content": "Hello, world!"}
  ],
  "session_id": "unique_session_id",
  "stream": true
}
```

### File Upload
```http
POST /upload
Authorization: Bearer your_token
Content-Type: multipart/form-data

file: <binary_file_data>
```

### Session Management
```http
POST /session/reset
Authorization: Bearer your_token
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Run verification tests: `python verify.py`
4. Ensure all tests pass
5. Submit a pull request

### Development Guidelines
- Maintain 100% test coverage for new features
- Follow existing code patterns and structure
- Update verification tests for new functionality
- Document API changes and new features

## 📊 Monitoring and Logs

- **Application Logs**: Stored in `logs/` directory
- **Verification History**: Tracked in `verification_history/`
- **Evidence Bundles**: Detailed test execution data in `evidence_bundles/`
- **Database**: Session and conversation data in `memory.db`

## 🔒 Security

- Bearer token authentication for API access
- Session isolation and management
- Input validation and sanitization
- Secure file upload handling

## 📚 Documentation

- `VERIFICATION_README.md` - Detailed verification system documentation
- `VERIFICATION_SUMMARY.md` - Test execution summaries
- `router.yaml` - Request routing configuration
- API documentation available at `/docs` when server is running

## 🐛 Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**
   - Check GPU availability: Fallback to CPU automatically implemented
   - Memory issues: Models auto-shard across available devices

2. **Model Loading**
   - Verify model IDs in environment variables
   - Check HuggingFace cache and authentication

3. **Database Issues**
   - Run `python init_db.py` to recreate database
   - Check file permissions on `memory.db`

4. **Bot Not Responding**
   - Verify `TELEGRAM_BOT_TOKEN` in `.env`
   - Check bot permissions and webhook configuration

### Performance Optimization

- Use GPU when available for model inference
- Adjust `max_new_tokens` and `temperature` for response quality
- Monitor memory usage with large models
- Use streaming responses for better user experience

## 📄 License

[License information to be added]

---

**Dateria AI Assistant** - Intelligent, multi-modal AI assistant with comprehensive tool integration and verification systems.