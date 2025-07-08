"""
Usage:
  # Predict from a raw string:
  
  FOR NEGATIVE
  python3 predict.py artifacts/best_lr_model.pkl artifacts/best_tfidf.pkl \
  "A boring, overlong snoozefest with no redeeming qualities."

  FOR POSITIVE
  python3 predict.py artifacts/best_lr_model.pkl artifacts/best_tfidf.pkl \
      "I absolutely loved this movie! Brilliant acting and story."

  # Or predict from a text file:
    python3 predict.py artifacts/best_lr_model.pkl artifacts/best_tfidf.pkl sample.txt
"""
import argparse
import pickle
import sys

def main():
    parser = argparse.ArgumentParser(
        description="IMDB Sentiment Predictor"
    )
    parser.add_argument(
        "model",
        help="Path to trained LogisticRegression .pkl model file"
    )
    parser.add_argument(
        "vectorizer",
        help="Path to pickled TfidfVectorizer (.pkl)"
    )
    parser.add_argument(
        "review",
        help="Either a path to a text file containing the review, or the raw review string"
    )
    args = parser.parse_args()

    # Load the TF-IDF vectorizer
    try:
        with open(args.vectorizer, "rb") as f:
            tfidf = pickle.load(f)
    except Exception as e:
        print(f"Error loading vectorizer: {e}", file=sys.stderr)
        sys.exit(1)

    # Load the trained classifier
    try:
        with open(args.model, "rb") as f:
            clf = pickle.load(f)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

    # Read the input review text (file or raw)
    try:
        with open(args.review, "r", encoding="utf-8") as rf:
            text = rf.read().strip()
    except (OSError, IOError):
        text = args.review.strip()

    if not text:
        print("No review text provided.", file=sys.stderr)
        sys.exit(1)

    # Vectorize and predict
    X = tfidf.transform([text])
    pred = clf.predict(X)[0]
    probs = clf.predict_proba(X)[0]

    label = "POSITIVE" if pred == 1 else "NEGATIVE"
    print(f"Prediction: {label}")
    print(f"  confidence → NEG: {probs[0]:.3f}, POS: {probs[1]:.3f}")

if __name__ == "__main__":
    main()
