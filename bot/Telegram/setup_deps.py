#!/usr/bin/env python3
"""
Phase 2 Setup and Dependencies Installation
==========================================

Helper script to install required dependencies and prepare the system for full verification.
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path

def check_module(module_name, install_name=None):
    """Check if a module is installed and importable."""
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is not None:
            return True, f"✅ {module_name} is installed"
        else:
            install_hint = install_name or module_name
            return False, f"❌ {module_name} not found. Install with: pip install {install_hint}"
    except ImportError:
        install_hint = install_name or module_name
        return False, f"❌ {module_name} import error. Install with: pip install {install_hint}"

def install_package(package_name):
    """Install a package using pip."""
    try:
        print(f"📦 Installing {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True, f"✅ Successfully installed {package_name}"
    except subprocess.CalledProcessError as e:
        return False, f"❌ Failed to install {package_name}: {e}"

def main():
    print("🔧 Phase 2 Setup and Dependencies Check")
    print("=" * 50)
    
    # Required packages for Phase 2
    required_packages = [
        ("telebot", "pytelegrambotapi"),      # Telegram bot library
        ("sentence_transformers", "sentence-transformers"),  # For RAG embeddings
        ("torch", "torch"),                    # PyTorch for models
        ("transformers", "transformers"),      # Hugging Face transformers
        ("requests", "requests"),              # HTTP requests
        ("sqlite3", None),                     # Built-in, no install needed
        ("json", None),                        # Built-in, no install needed
        ("dotenv", "python-dotenv"),           # Environment variables
    ]
    
    print("📋 Checking required dependencies...")
    
    missing_packages = []
    for module_name, install_name in required_packages:
        if install_name is None:  # Built-in module
            installed, message = check_module(module_name)
            print(f"   {message}")
        else:
            installed, message = check_module(module_name, install_name)
            print(f"   {message}")
            if not installed:
                missing_packages.append(install_name)
    
    if not missing_packages:
        print("\n✅ All dependencies are installed!")
        print("\n📝 Next steps:")
        print("   1. Stop the Telegram bot if running (Ctrl+C)")
        print("   2. Run core verification: python core_verify.py")
        print("   3. Run full verification: python quick_verify.py")
        return
    
    print(f"\n📦 Missing packages: {missing_packages}")
    
    # Ask user if they want to install missing packages
    try:
        install_choice = input("\n🤔 Install missing packages automatically? (y/n): ").lower().strip()
    except KeyboardInterrupt:
        print("\n\n👋 Setup cancelled by user.")
        return
    
    if install_choice in ['y', 'yes']:
        print("\n🚀 Installing missing packages...")
        
        failed_installs = []
        for package in missing_packages:
            success, message = install_package(package)
            print(f"   {message}")
            if not success:
                failed_installs.append(package)
        
        if failed_installs:
            print(f"\n❌ Failed to install: {failed_installs}")
            print("💡 Try installing manually:")
            for package in failed_installs:
                print(f"   pip install {package}")
        else:
            print("\n🎉 All packages installed successfully!")
            print("\n📝 Next steps:")
            print("   1. Stop the Telegram bot if running (Ctrl+C)")
            print("   2. Run verification: python quick_verify.py")
    else:
        print("\n📝 Manual installation required:")
        print("   pip install " + " ".join(missing_packages))
        print("\n📖 Then run: python quick_verify.py")
    
    print(f"\n📊 Check current status anytime with: python status.py")

if __name__ == "__main__":
    main()
