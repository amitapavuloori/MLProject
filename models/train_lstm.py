import sqlite3
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from utils import log_experiment
import sqlite3

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        tokens = self.tokenizer(self.texts[idx])
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        return torch.tensor(tokens, dtype=torch.long), self.labels[idx]

def collate_fn(batch):
    texts, labels = zip(*batch)
    texts_pad = pad_sequence(texts, batch_first=True, padding_value=0)
    return texts_pad, torch.tensor(labels)

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim=100, hidden_dim=128, num_layers=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(
            emb_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim*2, 2)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        # use last hidden state
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)
