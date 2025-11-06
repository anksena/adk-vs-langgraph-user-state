import sqlite3
import os
from typing import List

USER_DB = "user_data.db"
CHECKPOINT_DB = "lg_user_state.db"
ALL_DBS = [USER_DB, CHECKPOINT_DB]

def create_user_database():
    """Creates the user database and the users table if they don't exist."""
    conn = sqlite3.connect(USER_DB)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            name TEXT
        )
    """)
    conn.commit()
    conn.close()

def get_user_name(user_id: str) -> str | None:
    """Gets the user's name from the database."""
    conn = sqlite3.connect(USER_DB)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM users WHERE user_id = ?", (user_id,))
    result = cursor.fetchone()
    conn.close()
    if result:
        return result[0]
    else:
        return None

def persist_user_name_to_db(user_id: str, name: str):
    """Saves the user's name to the database."""
    conn = sqlite3.connect(USER_DB)
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO users (user_id, name) VALUES (?, ?)", (user_id, name))
    conn.commit()
    conn.close()

def cleanup_db(db_files: List[str]):
    """Cleans up the database files and their journal files."""
    for db_file in db_files:
        if os.path.exists(db_file):
            os.remove(db_file)
        if os.path.exists(f"{db_file}-shm"):
            os.remove(f"{db_file}-shm")
        if os.path.exists(f"{db_file}-wal"):
            os.remove(f"{db_file}-wal")