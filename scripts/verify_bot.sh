#!/bin/bash
# verify_bot.sh - Quick smoke test for Telegram Qwen Bot
# This script performs basic validation that the bot can start and respond

set -e

echo "🧪 Starting Telegram Qwen Bot Verification..."
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print test results
print_result() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅ PASS${NC}: $2"
        return 0
    else
        echo -e "${RED}❌ FAIL${NC}: $2"
        return 1
    fi
}

print_warning() {
    echo -e "${YELLOW}⚠️  WARN${NC}: $1"
}

# Track overall test status
TESTS_PASSED=0
TESTS_FAILED=0

echo "📁 Working directory: $(pwd)"
echo ""

# Test 1: Check if we're in the right directory
echo "🔍 Test 1: Verifying project structure..."
if [ -d "bot/Telegram" ] && [ -f "bot/Telegram/main_api.py" ]; then
    print_result 0 "Project structure found"
    ((TESTS_PASSED++))
else
    print_result 1 "Project structure not found - ensure you're in the project root"
    ((TESTS_FAILED++))
fi

# Test 2: Check environment file
echo "🔍 Test 2: Checking environment configuration..."
if [ -f "bot/Telegram/.env" ]; then
    print_result 0 "Environment file exists"
    ((TESTS_PASSED++))
    
    # Check for required environment variables
    if grep -q "TELEGRAM_BOT_TOKEN=" "bot/Telegram/.env" && 
       grep -q "MODEL_CHAT=" "bot/Telegram/.env" && 
       grep -q "BEARER_TOKEN=" "bot/Telegram/.env"; then
        print_result 0 "Required environment variables present"
        ((TESTS_PASSED++))
    else
        print_result 1 "Missing required environment variables"
        ((TESTS_FAILED++))
    fi
else
    print_result 1 "Environment file missing - copy .env.example to bot/Telegram/.env and configure"
    ((TESTS_FAILED++))
fi

# Test 3: Check Python requirements
echo "🔍 Test 3: Validating Python environment..."
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    print_result 1 "Python not found in PATH"
    ((TESTS_FAILED++))
    PYTHON_CMD=""
fi

if [ -n "$PYTHON_CMD" ]; then
    print_result 0 "Python interpreter found: $($PYTHON_CMD --version)"
    ((TESTS_PASSED++))
    
    # Check if we can import key dependencies
    echo "🔍 Test 4: Checking Python dependencies..."
    cd bot/Telegram
    
    if $PYTHON_CMD -c "import fastapi, transformers, torch, telebot" 2>/dev/null; then
        print_result 0 "Core dependencies available"
        ((TESTS_PASSED++))
    else
        print_result 1 "Missing Python dependencies - run: pip install -r requirements.txt"
        ((TESTS_FAILED++))
    fi
    
    # Test 5: Quick API syntax check
    echo "🔍 Test 5: Checking API syntax..."
    if $PYTHON_CMD -m py_compile main_api.py 2>/dev/null; then
        print_result 0 "API code syntax valid"
        ((TESTS_PASSED++))
    else
        print_result 1 "API code has syntax errors"
        ((TESTS_FAILED++))
    fi
    
    # Test 6: Quick Bot syntax check
    echo "🔍 Test 6: Checking bot syntax..."
    if $PYTHON_CMD -m py_compile bots/telegram_bot.py 2>/dev/null; then
        print_result 0 "Bot code syntax valid"
        ((TESTS_PASSED++))
    else
        print_result 1 "Bot code has syntax errors"
        ((TESTS_FAILED++))
    fi
    
    cd ../..
fi

# Test 7: Check database initialization
echo "🔍 Test 7: Checking database..."
if [ -f "bot/Telegram/memory.db" ]; then
    print_result 0 "Database file exists"
    ((TESTS_PASSED++))
else
    print_warning "Database not initialized - run: python bot/Telegram/init_db.py"
fi

# Test 8: Check startup scripts
echo "🔍 Test 8: Validating startup scripts..."
if [ -f "scripts/supervisor.sh" ] && [ -f "scripts/setup.sh" ]; then
    print_result 0 "Startup scripts available"
    ((TESTS_PASSED++))
else
    print_result 1 "Startup scripts missing in scripts/ directory"
    ((TESTS_FAILED++))
fi

# Summary
echo ""
echo "📊 VERIFICATION SUMMARY"
echo "======================="
echo "Tests Passed: $TESTS_PASSED"
echo "Tests Failed: $TESTS_FAILED"

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 ALL TESTS PASSED${NC}"
    echo "✅ Bot appears ready to run!"
    echo ""
    echo "Next steps:"
    echo "1. Configure environment: cp .env.example bot/Telegram/.env"
    echo "2. Install dependencies: pip install -r bot/Telegram/requirements.txt"
    echo "3. Initialize database: python bot/Telegram/init_db.py"
    echo "4. Start the bot: bash scripts/supervisor.sh"
    exit 0
else
    echo -e "${RED}❌ VERIFICATION FAILED${NC}"
    echo "Please fix the issues above before running the bot."
    exit 1
fi