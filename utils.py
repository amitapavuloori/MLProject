import sqlite3
import json
from datetime import datetime

def get_db_connection(db_path="experiment_log.db"):
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            model TEXT NOT NULL,
            params TEXT NOT NULL,
            train_loss REAL,
            val_loss REAL,
            val_accuracy REAL
        )
        """
    )
    conn.commit()
    return conn

def log_experiment(model_name, params: dict, train_loss=None, val_loss=None, val_acc=None):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO experiments (timestamp, model, params, train_loss, val_loss, val_accuracy) VALUES (?, ?, ?, ?, ?, ?)",
        (
            datetime.utcnow().isoformat(),
            model_name,
            json.dumps(params),
            train_loss,
            val_loss,
            val_acc
        )
    )
    conn.commit()
    conn.close()

