import sqlite3

def test_imdb_db(path="data/imdb_data.db"):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    for table in ("train", "test"):
        #count rows
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        cnt = cur.fetchone()[0]
        print(f"{table}: {cnt} rows")
        #show one sample row
        cur.execute(f"SELECT label, substr(text,1,50) FROM {table} LIMIT 1")
        label, snippet = cur.fetchone()
        print(f" → sample {table} row: label={label}, text=\"{snippet}...\"\n")
    conn.close()

if __name__ == "__main__":
    test_imdb_db()

