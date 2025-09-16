import os
import json
import sqlite3
from pathlib import Path


def create_test_files():
    """Create all required test files"""
    
    # Create test_files directory
    test_dir = Path("test_files")
    test_dir.mkdir(exist_ok=True)
    
    print("Creating test files...")
    
    # 1. Create config.json
    config_data = {
        "project_name": "AI ChatGPT System",
        "version": "1.0.0",
        "phase": 2,
        "gpu": "NVIDIA 5090",
        "settings": {
            "max_tokens": 4096,
            "temperature": 0.7,
            "model_path": "./models/chatgpt-local"
        },
        "database": {
            "path": "./test_files/data.db",
            "timeout": 30
        },
        "api": {
            "host": "localhost",
            "port": 8080,
            "endpoints": ["/chat", "/health", "/models"]
        }
    }
    
    with open(test_dir / "config.json", 'w') as f:
        json.dump(config_data, f, indent=2)
    print("✅ Created config.json")
    
    # 2. Create requirements.txt
    requirements = [
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "sqlite3",
        "requests>=2.31.0",
        "pydantic>=2.0.0",
        "python-multipart>=0.0.6"
    ]
    
    with open(test_dir / "requirements.txt", 'w') as f:
        f.write('\n'.join(requirements))
    print("✅ Created requirements.txt")
    
    # 3. Create SQLite database
    db_path = test_dir / "data.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username VARCHAR(50) NOT NULL UNIQUE,
            email VARCHAR(100) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT 1
        )
    ''')
    
    # Create projects table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name VARCHAR(100) NOT NULL,
            description TEXT,
            user_id INTEGER,
            status VARCHAR(20) DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Insert sample data
    sample_users = [
        ("john_doe", "john@example.com"),
        ("jane_smith", "jane@example.com"),
        ("ai_engineer", "engineer@ai-project.com")
    ]
    
    cursor.executemany(
        "INSERT OR IGNORE INTO users (username, email) VALUES (?, ?)", 
        sample_users
    )
    
    sample_projects = [
        ("ChatGPT Local System", "Main AI chatbot project", 1, "active"),
        ("Verification Tool", "Automated testing framework", 3, "completed"),
        ("Phase 2 Implementation", "Core system components", 1, "in_progress")
    ]
    
    cursor.executemany(
        "INSERT OR IGNORE INTO projects (name, description, user_id, status) VALUES (?, ?, ?, ?)",
        sample_projects
    )
    
    conn.commit()
    conn.close()
    print("✅ Created data.db with sample tables and data")
    
    print(f"\n{'='*50}")
    print("Test environment setup complete!")
    print(f"{'='*50}")
    print("Files created in ./test_files/:")
    print("  - config.json (project configuration)")
    print("  - requirements.txt (Python dependencies)")  
    print("  - data.db (SQLite database with users & projects tables)")
    print(f"{'='*50}")


if __name__ == "__main__":
    create_test_files()