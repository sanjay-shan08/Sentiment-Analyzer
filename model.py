# model.py
# handles training and running the sentiment model
#
# uses TF-IDF vectorizer + Logistic Regression (classic combo for text classification)
# trained on NLTK's built-in twitter_samples corpus (5000 pos + 5000 neg tweets)
#
# first run will download the NLTK data and train the model (~5-10 seconds)
# after that it saves to a .pkl file so it loads fast

import os
import re
import pickle

import nltk
from nltk.corpus import twitter_samples
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

nltk.download("twitter_samples", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

MODEL_FILE = "sentiment_model.pkl"


def _clean_text(text):
    """
    Basic text cleaning for social media posts.
    Removes urls, @mentions, extra spaces etc.
    """
    if not isinstance(text, str):
        return ""

    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.lower()

    return text


def train_model():
    """
    Trains a TF-IDF + Logistic Regression pipeline on NLTK twitter samples.
    Saves the model to disk so we don't retrain every time.
    """
    print("Training model on NLTK Twitter Samples dataset...")

    # load the data
    pos_tweets = twitter_samples.strings("positive_tweets.json")  # 5000 tweets
    neg_tweets = twitter_samples.strings("negative_tweets.json")  # 5000 tweets

    all_tweets = pos_tweets + neg_tweets
    labels = [1] * len(pos_tweets) + [0] * len(neg_tweets)  # 1 = positive, 0 = negative

    # clean the text
    cleaned = [_clean_text(t) for t in all_tweets]

    # build the pipeline
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=15000,
                    ngram_range=(1, 2),  # unigrams + bigrams
                    stop_words="english",
                    min_df=2,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    C=1.5,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        cleaned, labels, test_size=0.2, random_state=42, stratify=labels
    )

    pipeline.fit(X_train, y_train)

    acc = pipeline.score(X_test, y_test)
    print(f"  -> Model trained! Test accuracy: {acc:.2%}")
    print(f"  -> Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # save to disk
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(pipeline, f)

    print(f"  -> Saved to {MODEL_FILE}")
    return pipeline


def load_model():
    """
    Loads the model from disk if it exists, otherwise trains a new one.
    """
    if os.path.exists(MODEL_FILE):
        print("Loading existing model from disk...")
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)
        return model
    else:
        return train_model()


def predict(texts, model):
    """
    Runs sentiment prediction on a list of texts.
    Returns a list of dicts with sentiment label, confidence, and a -1 to 1 score.
    """
    if not texts:
        return []

    cleaned = [_clean_text(t) for t in texts]
    valid_indices = [i for i, t in enumerate(cleaned) if len(t) > 2]
    valid_texts = [cleaned[i] for i in valid_indices]

    results = [{"sentiment": "Neutral", "confidence": 0.5, "sent_score": 0.0}] * len(texts)

    if not valid_texts:
        return results

    preds = model.predict(valid_texts)
    probs = model.predict_proba(valid_texts)

    for idx, pred, prob in zip(valid_indices, preds, probs):
        label = "Positive" if pred == 1 else "Negative"
        confidence = float(max(prob))
        # score goes from -1 (very negative) to +1 (very positive)
        # prob[1] is probability of positive class
        sent_score = round(float(prob[1]) * 2 - 1, 3)

        results[idx] = {
            "sentiment": label,
            "confidence": round(confidence, 3),
            "sent_score": sent_score,
        }

    return results