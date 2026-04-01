import pandas as pd
import numpy as np
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from src.features import extract_features


def log(step, message):
    print(f"[{step}] {message}")


# ─────────────────────────────────────────────
# LOAD DATA
# LQ_EDIT excluded — inconsistent HTML formatting
# makes features unreliable for that class.
# Binary classification: HQ=1, LQ_CLOSE=0
# ─────────────────────────────────────────────

def load_data(filepath):
    log("LOAD", f"Loading data from {filepath}")

    df = pd.read_csv(filepath)
    log("LOAD", f"Raw shape: {df.shape}")

    # Drop LQ_EDIT — format inconsistency
    df = df[df['Y'] != 'LQ_EDIT'].copy()
    log("FILTER", f"After dropping LQ_EDIT: {df.shape}")

    label_map = {
        'HQ':       1,
        'LQ_CLOSE': 0
    }

    df['label'] = df['Y'].map(label_map)
    df = df.dropna(subset=['label'])

    log("LOAD", f"Final dataset shape: {df.shape}")
    log("LOAD", f"Class balance:\n{df['Y'].value_counts()}")
    return df


# ─────────────────────────────────────────────
# FEATURE MATRIX
# ─────────────────────────────────────────────

def build_feature_matrix(df):
    log("FEATURE", f"Building features for {len(df)} rows...")

    features = df.apply(
        lambda row: extract_features(
            row['Title'],
            row['Body'],
            row['Tags']
        ),
        axis=1
    )

    feature_df = pd.DataFrame(list(features))
    log("FEATURE", f"Matrix shape: {feature_df.shape}")
    log("FEATURE", f"Columns: {list(feature_df.columns)}")
    return feature_df


# ─────────────────────────────────────────────
# TRAIN MODEL
# ─────────────────────────────────────────────

def train_model(X, y):
    log("TRAIN", "Splitting dataset 80/20")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    log("TRAIN", f"Train: {len(X_train)}, Test: {len(X_test)}")

    # ── Logistic Regression with Scaling ──
    # Scaling is critical here because features have
    # very different ranges — body_word_count goes to 500+
    # while binary features are just 0 or 1.
    # Without scaling LR over-weights large-range features.
    log("MODEL", "Training Logistic Regression + Scaler")

    lr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            max_iter=3000,
            solver='saga',
            class_weight='balanced',
            random_state=42
        ))
    ])

    lr_pipeline.fit(X_train, y_train)
    lr_pred = lr_pipeline.predict(X_test)
    lr_acc  = accuracy_score(y_test, lr_pred)

    log("RESULT", f"Logistic Regression Accuracy: {lr_acc:.3f}")
    print(classification_report(y_test, lr_pred,
          target_names=['LQ_CLOSE', 'HQ']))

    # ── Random Forest ──
    # Does not need scaling — tree-based models
    # split on feature thresholds, not magnitudes
    log("MODEL", "Training Random Forest")

    rf = RandomForestClassifier(
        n_estimators=200,      # more trees = more stable
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )

    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc  = accuracy_score(y_test, rf_pred)

    log("RESULT", f"Random Forest Accuracy: {rf_acc:.3f}")
    print(classification_report(y_test, rf_pred,
          target_names=['LQ_CLOSE', 'HQ']))

    # Feature importances — this tells you
    # which features actually drive predictions
    log("INFO", "Feature importances (Random Forest):")
    importances = pd.Series(
        rf.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)
    print(importances)

    # Select the better model
    best_model = rf   if rf_acc >= lr_acc else lr_pipeline
    best_name  = "Random Forest" \
                 if rf_acc >= lr_acc \
                 else "Logistic Regression"

    log("SELECT", f"Best: {best_name} "
                  f"(accuracy: {max(rf_acc, lr_acc):.3f})")
    return best_model, X_test, y_test


# ─────────────────────────────────────────────
# SAVE / LOAD
# ─────────────────────────────────────────────

def save_model(model, path='models/classifier.pkl'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    log("SAVE", f"Model saved: {path}")


def load_model(path='models/classifier.pkl'):
    return joblib.load(path)


# ─────────────────────────────────────────────
# PREDICT
# Binary now: returns probability of being HQ
# ─────────────────────────────────────────────

def predict_score(model, title, body, tags):
    """
    Returns answerability score as 0-100%.
    P(HQ) * 100 = your quality percentage.
    """
    features   = extract_features(title, body, tags)
    feature_df = pd.DataFrame([features])

    # Align to trained column order
    columns    = joblib.load('models/feature_columns.pkl')
    feature_df = feature_df[columns]

    # [P(LQ_CLOSE), P(HQ)]
    proba = model.predict_proba(feature_df)[0]

    return {
        'overall_score':  round(proba[1] * 100, 1),
        'p_high_quality': round(proba[1] * 100, 1),
        'p_lq_close':     round(proba[0] * 100, 1),
        'features':       features
    }


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    log("START", "Pipeline started")

    try:
        BASE_DIR  = os.path.dirname(
                        os.path.dirname(
                            os.path.abspath(__file__)
                        )
                    )
        data_path = os.path.join(BASE_DIR, 'data', 'train.csv')

        df = load_data(data_path)
        X  = build_feature_matrix(df)
        y  = df['label']

        log("DATA", f"X: {X.shape}, y: {y.shape}")

        os.makedirs('models', exist_ok=True)
        joblib.dump(X.columns.tolist(),
                    'models/feature_columns.pkl')
        log("SAVE", "Feature column order saved")

        model, X_test, y_test = train_model(X, y)
        save_model(model)

        log("DONE", "Pipeline complete. Model ready.")

    except Exception as e:
        log("ERROR", str(e))
        raise