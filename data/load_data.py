"""
Load and store the IMDB reviews into a local SQLite database for easy querying.
Tables created:
 - train(id INTEGER PRIMARY KEY, text TEXT, label INTEGER)
 - test(id INTEGER PRIMARY KEY, text TEXT, label INTEGER)
"""
import sqlite3
from datasets import load_dataset
imdb = load_dataset("imdb")

def prepare_db(db_path="data/imdb_data.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # Create tables
    cur.execute("""
        CREATE TABLE IF NOT EXISTS train (
            id INTEGER PRIMARY KEY,
            text TEXT NOT NULL,
            label INTEGER NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS test (
            id INTEGER PRIMARY KEY,
            text TEXT NOT NULL,
            label INTEGER NOT NULL
        )
    """)
    conn.commit()

    # Load dataset
    dataset = load_dataset("imdb")
    # Insert train split
    train_data = dataset['train']
    for i, example in enumerate(train_data):
        cur.execute(
            "INSERT OR IGNORE INTO train (id, text, label) VALUES (?, ?, ?)",
            (i, example['text'], example['label'])
        )
    # Insert test split
    test_data = dataset['test']
    for i, example in enumerate(test_data):
        cur.execute(
            "INSERT OR IGNORE INTO test (id, text, label) VALUES (?, ?, ?)",
            (i, example['text'], example['label'])
        )
    conn.commit()
    conn.close()

if __name__ == "__main__":
    prepare_db()
    print("IMDB data loaded into SQLite database.")
