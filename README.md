# MLProject

How to run it:
1. Once you have trained the model:
  python3 predict.py \
  artifacts/best_lr_model.pkl \
  artifacts/best_tfidf.pkl \
  "I absolutely loved this movie—highly recommend!"
  
2. You can pass a txt file:
python3 predict.py artifacts/best_lr_model.pkl artifacts/best_tfidf.pkl sample.txt
