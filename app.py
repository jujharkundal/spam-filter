import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

from src.model import tokenize

st.set_page_config(
    page_title="Spam Filter",
    page_icon="📩",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "spam_filter_model.pkl"

if not MODEL_PATH.exists():
    st.error("Model file not found.")
    st.stop()

with open(MODEL_PATH, "rb") as f:
    artifact = pickle.load(f)

vectorizer = artifact["vectorizer"]
nb_model = artifact["model"]
threshold = artifact["threshold"]
metrics = artifact["metrics"]

st.title("📩 Spam Filter")

tab1, tab2 = st.tabs(["Predict", "Performance"])

with tab1:
    message = st.text_area(
        "Enter SMS",
        height=150,
        placeholder="Enter your message..."
    )

    if st.button("Predict", type="primary"):
        if not message.strip():
            st.warning("Enter a message.")
        else:
            X = vectorizer.get_counts([message])

            spam_index = list(nb_model.classes_).index(1)

            spam_probability = float(
                nb_model.predict_proba(X)[0, spam_index]
            )

            ham_probability = 1 - spam_probability

            prediction = (
                "SPAM"
                if spam_probability >= threshold
                else "HAM"
            )

            if prediction == "SPAM":
                st.error("🚨 SPAM")
            else:
                st.success("✅ HAM")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Prediction",
                prediction
            )

            col2.metric(
                "Spam Probability",
                f"{spam_probability:.2%}"
            )

            col3.metric(
                "Ham Probability",
                f"{ham_probability:.2%}"
            )

            tokens = tokenize(message)

            known_tokens = [
                token
                for token in tokens
                if token in vectorizer.vocab_
            ]

            st.write(
                f"Tokens: {len(tokens)}"
            )

            st.write(
                f"Vocabulary matches: {len(known_tokens)}"
            )

with tab2:
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Accuracy",
        f"{metrics['accuracy']:.2%}"
    )

    col2.metric(
        "Precision",
        f"{metrics['precision']:.2%}"
    )

    col3.metric(
        "Recall",
        f"{metrics['recall']:.2%}"
    )

    col4.metric(
        "F1 Score",
        f"{metrics['f1']:.2%}"
    )

    st.subheader("Confusion Matrix")

    cm = pd.DataFrame(
        [
            [metrics["tn"], metrics["fp"]],
            [metrics["fn"], metrics["tp"]]
        ],
        index=["HAM", "SPAM"],
        columns=["HAM", "SPAM"]
    )

    st.dataframe(
        cm,
        use_container_width=True
    )

    st.subheader("Dataset")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total SMS",
        f"{artifact['dataset_size']:,}"
    )

    col2.metric(
        "Training SMS",
        f"{artifact['train_size']:,}"
    )

    col3.metric(
        "Test SMS",
        f"{artifact['test_size']:,}"
    )
