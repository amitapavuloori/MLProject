import sqlite3
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, log_loss
from utils import log_experiment

#load data
def load_data(db_path="data/imdb_data.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT text, label FROM train")
    data = cur.fetchall()
    conn.close()
    texts, labels = zip(*data)
    return list(texts), np.array(labels)

if __name__ == "__main__":
    texts, labels = load_data()
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    #vectorization + grid search
    vectorizer = TfidfVectorizer(max_features=10000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)

    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'ngram_range': [(1,1), (1,2)]
    }
    best_score = 0
    best_params = None
    for C in param_grid['C']:
        for ngram in param_grid['ngram_range']:
            vec = TfidfVectorizer(max_features=10000, ngram_range=ngram)
            Xtr = vec.fit_transform(X_train)
            Xvl = vec.transform(X_val)
            clf = LogisticRegression(C=C, solver='liblinear', class_weight='balanced')
            clf.fit(Xtr, y_train)
            preds = clf.predict(Xvl)
            acc = accuracy_score(y_val, preds)
            loss = log_loss(y_val, clf.predict_proba(Xvl))
            params = {'C': C, 'ngram_range': ngram}
            log_experiment('LogisticRegression', params, train_loss=None, val_loss=loss, val_acc=acc)
            if acc > best_score:
                best_score, best_params = acc, params
    print(f"Best validation accuracy: {best_score:.4f} with params {best_params}")
    #re‐fit on the entire training set
    print("Retraining best model on full training set…")
    #combine X_train + X_val back into 'texts' and 'labels' from load_data()
    texts, labels = load_data()            # your full 25k train
    # 2)create & fit vectorizer
    vec = TfidfVectorizer(max_features=10000,
                          ngram_range=tuple(best_params['ngram_range']))
    X_full = vec.fit_transform(texts)
    # 3)train final classifier
    final_clf = LogisticRegression(C=best_params['C'],
                                   solver='liblinear',
                                   class_weight='balanced')
    final_clf.fit(X_full, labels)

    # 4)pickle both objects
    import pickle, os
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/best_tfidf.pkl", "wb") as f:
        pickle.dump(vec, f)
    with open("artifacts/best_lr_model.pkl", "wb") as f:
        pickle.dump(final_clf, f)
    print("Saved vectorizer ➔ artifacts/best_tfidf.pkl")
    print("Saved model      ➔ artifacts/best_lr_model.pkl")

