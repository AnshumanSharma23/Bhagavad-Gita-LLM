import sqlite3

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('feedback_of_users.db')
c = conn.cursor()

# Create table to store queries, responses, and feedback with embeddings
c.execute('''
CREATE TABLE IF NOT EXISTS feedback_of_users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT,
    response TEXT,
    liked BOOLEAN,
    embedding BLOB
)
''')

conn.commit()
conn.close()
