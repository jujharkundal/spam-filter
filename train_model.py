
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

from src.model import TfidfVectorizer, MultinomialNaiveBayes, compute_metrics

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "SMSSpamCollection"
MODEL_DIR = ROOT / "model"

if not DATA_PATH.is_file():
    raise FileNotFoundError(
        f"Dataset file not found: {DATA_PATH}. "
        "Place the real SMSSpamCollection file in data/."
    )

df = pd.read_csv(DATA_PATH, sep="\t", header=None, names=["label", "message"], encoding="utf-8")
df["label"] = df["label"].str.strip().str.lower()
df = df[df["label"].isin(["ham", "spam"])].copy()

if len(df) < 5000:
    raise ValueError(f"Expected the complete 5000+ SMS dataset; found {len(df)}.")

rng = np.random.default_rng(42)
train_indices, test_indices = [], []
for label in ["ham", "spam"]:
    idx = np.where(df["label"].values == label)[0]
    rng.shuffle(idx)
    n_train = int(len(idx) * 0.8)
    train_indices.extend(idx[:n_train])
    test_indices.extend(idx[n_train:])
rng.shuffle(train_indices)
rng.shuffle(test_indices)

X_train = df.iloc[train_indices]["message"].astype(str).tolist()
X_test = df.iloc[test_indices]["message"].astype(str).tolist()
y_train = (df.iloc[train_indices]["label"].values == "spam").astype(int)
y_test = (df.iloc[test_indices]["label"].values == "spam").astype(int)

vectorizer = TfidfVectorizer(max_features=3000)
vectorizer.fit(X_train)
X_train_counts = vectorizer.get_counts(X_train)
X_test_counts = vectorizer.get_counts(X_test)

model = MultinomialNaiveBayes(alpha=1.0).fit(X_train_counts, y_train)
prob = model.predict_proba(X_test_counts)[:, list(model.classes_).index(1)]
pred = (prob >= 0.5).astype(int)
metrics = compute_metrics(y_test, pred)

MODEL_DIR.mkdir(exist_ok=True)
with open(MODEL_DIR / "spam_filter_model.pkl", "wb") as f:
    pickle.dump({
        "vectorizer": vectorizer,
        "model": model,
        "threshold": 0.5,
        "dataset_size": len(df),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "metrics": metrics,
    }, f)

pd.DataFrame({
    "message": X_test,
    "actual_label": np.where(y_test == 1, "SPAM", "HAM"),
    "predicted_label": np.where(pred == 1, "SPAM", "HAM"),
    "spam_probability": prob,
    "correct": y_test == pred,
}).to_csv(MODEL_DIR / "test_predictions.csv", index=False)

print("Model trained and saved successfully.")
print(metrics)
