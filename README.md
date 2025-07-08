# MLProject

imdb_sentiment/
├── artifacts/ # TF-IDF & LogisticRegression
├── data/
  ├── imdb_data.db.zip # Decompress the DB file
│ └── load_data.py
├── models/
│ ├── train_lr.py # TF-IDF + Logistic Regression
│ ├── train_lstm.py # Bi-LSTM sequence model
│ └── train_transformer.py # DistilBERT fine‐tuning
├── experiment_log.db 
├── utils.py 
├── test_db.py # Test file
├── predict.py # Classifying a review string or file
├── requirements.txt 
└── report.txt



How to run it:
1.Once you have trained the model:
  python3 predict.py \
  artifacts/best_lr_model.pkl \
  artifacts/best_tfidf.pkl \
  "I absolutely loved this movie—highly recommend!"
  
2. You can pass a txt file:
python3 predict.py artifacts/best_lr_model.pkl artifacts/best_tfidf.pkl sample.txt
