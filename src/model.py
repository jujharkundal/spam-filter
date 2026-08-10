
import math
import re
import string
from collections import Counter
import numpy as np

STOPWORDS = set("""
a an the is are was were be been being to of and or in on at for with as by
this that it its from your you i we they he she our their them his her
""".split())

def tokenize(text: str):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]

class TfidfVectorizer:
    def __init__(self, max_features=3000):
        self.vocab_ = {}
        self.idf_ = None
        self.max_features = max_features

    def fit(self, corpus):
        doc_freq = Counter()
        term_freq_total = Counter()
        tokenized_docs = [tokenize(d) for d in corpus]

        for tokens in tokenized_docs:
            term_freq_total.update(tokens)
            doc_freq.update(set(tokens))

        vocab_words = list(doc_freq.keys())
        if self.max_features is not None and len(vocab_words) > self.max_features:
            vocab_words = [w for w, _ in term_freq_total.most_common(self.max_features)]

        self.vocab_ = {w: i for i, w in enumerate(sorted(vocab_words))}

        N = len(corpus)
        self.idf_ = np.zeros(len(self.vocab_))
        for w, i in self.vocab_.items():
            df = doc_freq[w]
            self.idf_[i] = math.log((1 + N) / (1 + df)) + 1.0
        return self

    def transform(self, corpus):
        rows = np.zeros((len(corpus), len(self.vocab_)))
        for r, doc in enumerate(corpus):
            tokens = tokenize(doc)
            if not tokens:
                continue
            counts = Counter(tokens)
            total_tokens = len(tokens)
            for w, c in counts.items():
                if w in self.vocab_:
                    j = self.vocab_[w]
                    rows[r, j] = (c / total_tokens) * self.idf_[j]
            norm = np.linalg.norm(rows[r])
            if norm > 0:
                rows[r] /= norm
        return rows

    def fit_transform(self, corpus):
        return self.fit(corpus).transform(corpus)

    def get_counts(self, corpus):
        rows = np.zeros((len(corpus), len(self.vocab_)), dtype=float)
        for r, doc in enumerate(corpus):
            counts = Counter(tokenize(doc))
            for w, c in counts.items():
                if w in self.vocab_:
                    rows[r, self.vocab_[w]] = c
        return rows

class MultinomialNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X_counts, y):
        X_counts = np.asarray(X_counts, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_features = X_counts.shape[1]
        self.class_log_prior_ = np.zeros(len(self.classes_))
        self.feature_log_prob_ = np.zeros((len(self.classes_), n_features))

        for idx, c in enumerate(self.classes_):
            X_c = X_counts[y == c]
            self.class_log_prior_[idx] = math.log(X_c.shape[0] / X_counts.shape[0])
            feature_counts = X_c.sum(axis=0) + self.alpha
            self.feature_log_prob_[idx] = np.log(feature_counts / feature_counts.sum())
        return self

    def joint_log_likelihood(self, X):
        return np.asarray(X, dtype=float) @ self.feature_log_prob_.T + self.class_log_prior_

    def predict_log_proba(self, X):
        jll = self.joint_log_likelihood(X)
        max_jll = jll.max(axis=1, keepdims=True)
        log_prob_x = max_jll + np.log(
            np.exp(jll - max_jll).sum(axis=1, keepdims=True)
        )
        return jll - log_prob_x

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        spam_col = list(self.classes_).index(1)
        return (proba[:, spam_col] >= threshold).astype(int)

def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
