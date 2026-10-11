"""
Train the spam detector and export it as model.js for the website.

Run (from inside the training folder):  python train_export.py
Needs emails.csv (Kaggle dataset with a text column and a spam/label column)
in the same folder.
"""
import glob
import json
import os
from datetime import date

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from preprocess import prepare_email

DATA_DIR = "data"          # put any number of CSV files here
FALLBACK_CSV = "emails.csv"  # used if data/ has no CSV files
SOURCES = []
OUTPUT_PATH = "../model.js"
MAX_FEATURES = 20000  # keeps model.js small so the page loads fast


def _load_one(path):
    df = pd.read_csv(path, encoding="latin-1")
    cols = {c.lower().strip(): c for c in df.columns}

    text_col = next((cols[c] for c in ["text", "message", "body", "email", "content"] if c in cols), None)
    label_col = next((cols[c] for c in ["label_num", "label", "category", "spam", "class", "target"] if c in cols), None)
    if text_col is None or label_col is None:
        raise SystemExit(f"Could not detect columns. Found: {list(df.columns)}. "
                         "Edit load_data() in train_export.py.")

    df = df[[text_col, label_col]].dropna()
    df.columns = ["text", "label"]

    if df["label"].dtype == object:
        df["label"] = df["label"].str.lower().str.strip().map(
            {"ham": 0, "spam": 1, "not spam": 0, "legitimate": 0, "0": 0, "1": 1}
        )
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df = df.drop_duplicates(subset="text")

    return df


def load_data():
    """Load every CSV in data/ (or emails.csv), merge them and clean the text."""
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    if not files and os.path.exists(FALLBACK_CSV):
        files = [FALLBACK_CSV]
    if not files:
        raise SystemExit("No CSV found. Put your dataset in training/data/ (or training/emails.csv).")

    parts = []
    SOURCES.clear()
    for f in files:
        part = _load_one(f)
        print(f"Loaded {os.path.basename(f)}: {len(part)} emails")
        SOURCES.append({"file": os.path.basename(f), "emails": int(len(part))})
        parts.append(part)

    df = pd.concat(parts, ignore_index=True).drop_duplicates(subset="text")
    df["clean"] = df["text"].apply(prepare_email)
    return df[df["clean"].str.len() > 0]


def train(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2), min_df=2, max_df=0.95,
        sublinear_tf=True, max_features=MAX_FEATURES,
    )
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)

    candidates = {
        "naive_bayes": MultinomialNB(alpha=0.1),
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    }

    results = {}
    best = None
    for name, clf in candidates.items():
        clf.fit(Xtr, y_train)
        preds = clf.predict(Xte)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        metrics = {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "f1": round(f1_score(y_test, preds), 4),
            "precision": round(precision_score(y_test, preds), 4),
            "recall": round(recall_score(y_test, preds), 4),
            "confusion": [int(tn), int(fp), int(fn), int(tp)],
            "false_positives": int(fp),
            "false_negatives": int(fn),
        }
        results[name] = metrics

        print(f"\n===== {name} =====")
        print(f"Accuracy: {metrics['accuracy']} | Spam F1: {metrics['f1']}")
        print(classification_report(y_test, preds, target_names=["ham", "spam"]))
        print(f"False positives (real email marked spam): {fp}")
        print(f"False negatives (spam missed): {fn}")

        if best is None or metrics["f1"] > best[2]["f1"]:
            best = (name, clf, metrics)

    return vectorizer, best, results, (X_test, y_test)


def export_model(vectorizer, best, results, n_emails, path=OUTPUT_PATH):
    name, clf, metrics = best

    # Both models are linear: score = weights . tfidf + bias, probability = sigmoid(score)
    if name == "naive_bayes":
        weights = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
        bias = clf.class_log_prior_[1] - clf.class_log_prior_[0]
    else:
        weights = clf.coef_[0]
        bias = clf.intercept_[0]

    model = {
        "model_name": name,
        "n_emails": int(n_emails),
        "trained_at": date.today().isoformat(),
        "sources": SOURCES,
        "metrics": metrics,
        "all_results": results,
        "bias": float(bias),
        "stop_words": sorted(ENGLISH_STOP_WORDS),
        "terms": [str(t) for t in vectorizer.get_feature_names_out()],
        "idf": [round(float(v), 6) for v in vectorizer.idf_],
        "weights": [round(float(v), 6) for v in weights],
    }

    with open(path, "w", encoding="utf-8") as f:
        f.write("// Generated by train_export.py - do not edit by hand\n")
        f.write("const SPAM_MODEL = ")
        json.dump(model, f, separators=(",", ":"))
        f.write(";\n")

    print(f"\nExported {name} with {len(model['terms'])} features -> {path}")


if __name__ == "__main__":
    data = load_data()
    print("Dataset size:", len(data))
    print(data["label"].value_counts().rename({0: "ham", 1: "spam"}))
    vec, best_model, all_results, _ = train(data)
    export_model(vec, best_model, all_results, len(data))
