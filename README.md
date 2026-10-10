# Email Spam Detector

A machine learning project that classifies an email as **spam** or **legitimate (ham)**.
The trained model runs entirely in the browser, so the website is fully static and needs no server.

**Live demo:** https://akib2005.github.io/email-spam-detector/

## How it works

1. **Clean** the email: remove HTML, replace links, email addresses and numbers with tokens, lowercase everything.
2. **Vectorize** with TF-IDF (words and word pairs, 20,000 features).
3. **Classify** with a linear model (Naive Bayes or Logistic Regression, whichever scores the better spam F1).
4. **Export** the learned weights to `model.js`. The page recomputes the same TF-IDF and score in JavaScript (`spam.js`).

The JavaScript version was checked against the Python model on 300+ emails: probabilities agree to within 1e-7 and no predictions differ.

## Results (20% held-out test set, 1,139 emails)

| Model | Accuracy | Real emails wrongly flagged | Spam missed |
|---|---|---|---|
| Naive Bayes (used) | 99.2% | 2 | 7 |
| Logistic Regression | 98.9% | 10 | 2 |

Naive Bayes is used because it flags far fewer real emails as spam, which is the more costly mistake in an inbox.

## Limitations

- Trained on the Enron email corpus, so it is best at classic spam and phishing. Modern marketing emails can get uncertain scores.
- Some words are learned from Enron-specific patterns (for example, "password" leans toward legitimate because of internal IT emails there).
- It looks at text only, not sender reputation, headers or attachments.

## Project structure

```
index.html        the website
spam.js           classifier logic (runs in the browser)
model.js          exported trained model
train_export.py   trains the model and writes model.js
preprocess.py     email cleaning used during training
requirements.txt  Python dependencies for training
```

## Run locally

Open `index.html` in a browser. No server or installation is needed.

## Retrain the model

1. Download the email spam dataset from Kaggle and save it as `emails.csv` in this folder (columns: `text` and `spam`).
2. Run:

```
pip install -r requirements.txt
python train_export.py
```

This regenerates `model.js`.

## Tech

Python, scikit-learn, pandas, JavaScript, HTML/CSS, GitHub Pages
