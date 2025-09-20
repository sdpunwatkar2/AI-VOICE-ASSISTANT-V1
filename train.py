
# Simple training script for intent classification.
# Replace dataset.csv with your diversified dataset for stricter, better training.
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
    # Expecting columns: text,intent
    return df['text'].astype(str), df['intent'].astype(str)

def train():
    X, y = load_dataset(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=20000)),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    # Grid search with strict parameters (example)
    params = {
        'clf__C': [0.1, 1, 5],
        'tfidf__max_df': [0.8, 0.95]
    }
    gs = GridSearchCV(pipe, params, cv=5, verbose=1, n_jobs=-1)
    gs.fit(X_train, y_train)
    print("Best params:", gs.best_params_)
    print("Train score:", gs.score(X_train, y_train))
    print("Test score:", gs.score(X_test, y_test))
    joblib.dump(gs.best_estimator_, os.path.join(MODELS_DIR, 'intent_model.joblib'))

if __name__ == '__main__':
    train()
