# 📩 Spam Filter From Scratch

A spam SMS classifier implemented from first principles using Python, NumPy, and mathematical implementations of TF-IDF and Multinomial Naive Bayes.

## What this project demonstrates

- Custom text tokenization and preprocessing
- Custom TF-IDF mathematics
- Multinomial Naive Bayes from scratch
- Laplace smoothing
- Log-probability calculations
- Precision-first evaluation to reduce false positives
- Streamlit deployment interface

**No `sklearn` TF-IDF vectorizer or Naive Bayes classifier is used.**

## Dataset

SMS Spam Collection:

- 5,574 SMS messages
- 4,827 ham
- 747 spam

The real dataset is stored at `data/SMSSpamCollection`.

There is **no synthetic-data fallback**. If the dataset is missing or incomplete, training stops with an error.

## Repository

```text
spam-filter-from-scratch/
├── app.py
├── train_model.py
├── data/
│   └── SMSSpamCollection
├── model/
│   ├── spam_filter_model.pkl
│   └── test_predictions.csv
├── notebooks/
│   └── Spam_Filter.ipynb
├── src/
│   └── model.py
├── requirements.txt
└── README.md
```

## Run locally

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

The Streamlit app lets a user enter a new SMS and receive:

- HAM / SPAM classification
- Spam probability
- Ham probability
- Token/vocabulary information
- Accuracy, precision, recall and F1
- Confusion matrix
- Dataset statistics

## Model pipeline

```text
SMS
 ↓
Custom preprocessing
 ↓
Tokenization
 ↓
Vocabulary
 ↓
TF-IDF / word-count representation
 ↓
Multinomial Naive Bayes
 ↓
Spam probability
 ↓
HAM / SPAM
```

## Evaluation focus

Precision is the primary metric because a false positive means a legitimate message is incorrectly classified as spam.

## Future improvements

- Threshold optimization in the deployed model
- Better feature explanations
- Additional spam-specific features
- Streamlit deployment
