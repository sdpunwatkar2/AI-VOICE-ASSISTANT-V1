# train.py
# Fixed + robust intent classifier training script

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib
import os

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)
DATA_PATH = 'dataset.csv'


def load_dataset(path):
    df = pd.read_csv(path)
    if 'text' not in df.columns or 'intent' not in df.columns:
        raise ValueError("Dataset must have 'text' and 'intent' columns")
    return df['text'].astype(str), df['intent'].astype(str)


def train():
    X, y = load_dataset(DATA_PATH)

    # Show class balance
    print("\nClass distribution:\n", y.value_counts())

    # Ensure safe split (fallback if stratify fails)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )
    except ValueError:
        print("⚠️ Not enough samples for stratify. Falling back to random split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42
        )

    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=20000)),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    # Determine safe CV folds
    min_class_count = y.value_counts().min()
    cv_folds = min(3, min_class_count) if min_class_count > 1 else 2
    print(f"Using CV folds: {cv_folds}")

    params = {
        'clf__C': [0.1, 1, 5],
        'tfidf__max_df': [0.8, 0.95]
    }

    gs = GridSearchCV(pipe, params, cv=cv_folds, verbose=1, n_jobs=-1)
    gs.fit(X_train, y_train)

    print("\nBest params:", gs.best_params_)
    print("Train score:", gs.score(X_train, y_train))
    print("Test score:", gs.score(X_test, y_test))

    model_path = os.path.join(MODELS_DIR, 'intent_model.joblib')
    joblib.dump(gs.best_estimator_, model_path)
    print(f"✅ Model saved at {model_path}")


if __name__ == '__main__':
    train()
