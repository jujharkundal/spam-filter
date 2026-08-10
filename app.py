
import pickle
from pathlib import Path
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "spam_filter_model.pkl"
PREDICTIONS_PATH = BASE_DIR / "model" / "test_predictions.csv"

st.set_page_config(
    page_title="Spam Filter From Scratch",
    page_icon="📩",
    layout="wide",
)

if not MODEL_PATH.exists():
    st.error(
        "Trained model not found. Run train_model.py first to create "
        "model/spam_filter_model.pkl."
    )
    st.stop()

with open(MODEL_PATH, "rb") as f:
    artifact = pickle.load(f)

vectorizer = artifact["vectorizer"]
nb_model = artifact["model"]
threshold = artifact["threshold"]
metrics = artifact["metrics"]

st.title("📩 Spam Filter — Built From Scratch")
st.caption("Custom tokenization + TF-IDF + Multinomial Naive Bayes | No sklearn classifier/vectorizer")

tab_predict, tab_performance, tab_about = st.tabs(
    ["🔍 Predict", "📊 Model Performance", "🧠 How It Works"]
)

with tab_predict:
    st.subheader("Classify a new SMS")
    message = st.text_area(
        "Enter an SMS message",
        height=150,
        placeholder="Paste a message here and click Predict...",
    )

    if st.button("🔍 Predict", type="primary"):
        if not message.strip():
            st.warning("Please enter an SMS message.")
        else:
            X = vectorizer.get_counts([message])
            probability = float(nb_model.predict_proba(X)[0, list(nb_model.classes_).index(1)])
            prediction = "SPAM" if probability >= threshold else "HAM"

            if prediction == "SPAM":
                st.error("🚨 SPAM")
            else:
                st.success("✅ HAM — Legitimate")

            c1, c2, c3 = st.columns(3)
            c1.metric("Classification", prediction)
            c2.metric("Spam probability", f"{probability:.2%}")
            c3.metric("Ham probability", f"{1 - probability:.2%}")

            tokens = vectorizer._tokenize(message)
            known_tokens = [t for t in tokens if t in vectorizer.vocab_]
            st.write(f"**Tokens found:** {len(tokens)}")
            st.write(f"**Vocabulary tokens used:** {len(known_tokens)}")

with tab_performance:
    st.subheader("Performance on the held-out SMS dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    c2.metric("Precision", f"{metrics['precision']:.2%}")
    c3.metric("Recall", f"{metrics['recall']:.2%}")
    c4.metric("F1 Score", f"{metrics['f1']:.2%}")

    st.write("### Confusion Matrix")
    cm = pd.DataFrame(
        [
            [metrics["tn"], metrics["fp"]],
            [metrics["fn"], metrics["tp"]],
        ],
        index=["Actual HAM", "Actual SPAM"],
        columns=["Predicted HAM", "Predicted SPAM"],
    )
    st.dataframe(cm, use_container_width=True)

    st.write("### Dataset")
    d1, d2, d3 = st.columns(3)
    d1.metric("Total SMS", f"{artifact['dataset_size']:,}")
    d2.metric("Training SMS", f"{artifact['train_size']:,}")
    d3.metric("Test SMS", f"{artifact['test_size']:,}")

    st.info(
        f"Primary evaluation metric: **precision**. "
        f"The test set produced **{metrics['fp']} false positives**."
    )

with tab_about:
    st.subheader("Model pipeline")
    st.code(
        """SMS
 ↓
Custom preprocessing + tokenization
 ↓
Custom vocabulary
 ↓
TF-IDF mathematics
 ↓
Word-count representation
 ↓
Multinomial Naive Bayes
 ↓
Spam probability
 ↓
HAM / SPAM""",
        language="text",
    )
    st.write(
        "The model components are implemented from scratch with Python, NumPy, "
        "and math. Streamlit is only the interface used to serve the trained model."
    )
