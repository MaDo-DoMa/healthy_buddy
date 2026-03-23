import sqlite3
from datetime import datetime
from config import DB_PATH

def init_db():
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                text      TEXT    NOT NULL,
                done      INTEGER NOT NULL DEFAULT 0,
                created   TEXT    NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key   TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        con.commit()

def db_load_tasks():
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("SELECT id, text, done FROM tasks ORDER BY id DESC")
        return cur.fetchall()

def db_add_task(text):
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("INSERT INTO tasks (text, done, created) VALUES (?, 0, ?)", (text, datetime.now().isoformat()))
        con.commit()
        return cur.lastrowid

def db_toggle_task(task_id, done):
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("UPDATE tasks SET done = ? WHERE id = ?", (done, task_id))
        con.commit()

def db_delete_task(task_id):
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        con.commit()

def db_save_setting(key, value):
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, str(value)))
        con.commit()

def db_load_setting(key, default=None):
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("SELECT value FROM settings WHERE key = ?", (key,))
        row = cur.fetchone()
        return row[0] if row else default
