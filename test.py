import sqlite3
conn = sqlite3.connect("users.db")
cur = conn.cursor()
cur.execute("DROP TABLE IF EXISTS user_history")
cur.execute("""
    CREATE TABLE user_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_email TEXT NOT NULL,
        ticker TEXT NOT NULL,
        time TEXT NOT NULL
    )
""")
conn.commit()
conn.close()