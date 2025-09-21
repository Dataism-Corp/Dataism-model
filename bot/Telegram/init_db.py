#!/usr/bin/env python3
"""
Database initialization script
Run this if you encounter database table issues
"""

import os
import sqlite3

# Use the same path as the project
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT_ROOT, "memory.db")

def init_database():
    """Initialize all required database tables"""
    print(f"Initializing database at: {DB_PATH}")
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # memory table for RAG
    c.execute("""CREATE TABLE IF NOT EXISTS memory(
        id TEXT PRIMARY KEY,
        ts TEXT,
        session_id TEXT,
        source TEXT,
        title TEXT,
        snippet TEXT,
        url TEXT,
        content TEXT,
        emb BLOB
    )""")
    print("✅ Created/verified 'memory' table")

    # cache table for API caching
    c.execute("""CREATE TABLE IF NOT EXISTS cache(
        key TEXT PRIMARY KEY,
        ts INTEGER,
        ttl INTEGER,
        payload TEXT
    )""")
    print("✅ Created/verified 'cache' table")

    # sessions table for chat logs
    c.execute("""CREATE TABLE IF NOT EXISTS sessions(
        session_id TEXT,
        ts TEXT,
        role TEXT,
        content TEXT
    )""")
    print("✅ Created/verified 'sessions' table")

    # routing_logs table for analytics  
    c.execute("""CREATE TABLE IF NOT EXISTS routing_logs(
        ts TEXT,
        user_text TEXT,
        selected_tool TEXT,
        confidence REAL,
        reason TEXT
    )""")
    print("✅ Created/verified 'routing_logs' table")

    conn.commit()
    conn.close()
    print(f"🎉 Database initialization complete!")

if __name__ == "__main__":
    init_database()
