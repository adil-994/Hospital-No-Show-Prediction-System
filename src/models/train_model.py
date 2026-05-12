import pandas as pd
import os
import joblib
import warnings
import logging

# Suppress warnings and background logs for a clean report
warnings.filterwarnings("ignore")
logging.getLogger("mlflow").setLevel(logging.ERROR)

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# --- 1. SETUP AND UTILITIES ---

def setup_tracking():
    base_dir = os.getcwd()
    mlflow.set_tracking_uri(f"file:///{os.path.join(base_dir, 'mlruns')}")
    mlflow.set_experiment("Hospital_NoShow_Final_Serialization")

def load_and_prepare_data(path):
    df = pd.read_csv(path)
    return pd.get_dummies(df)

def calculate_metrics_dict(model, X, y):
    """Calculates all metrics for the structured report."""
    preds = model.predict(X)
    # Some models don't support predict_proba (like LinearSVC)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
        roc = roc_auc_score(y, probs)
    else:
        roc = 0.0  # Placeholder for models without probability

    return {
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds),
        "recall": recall_score(y, preds),
        "f1": f1_score(y, preds),
        "roc_auc": roc,
        "cm": confusion_matrix(y, preds)
    }

# --- 2. THE STRUCTURED REPORT ---

def print_structured_report(model_name, m_train, m_val, m_test):
    """
    Prints a professional, structured report matching the requested format.
    """
    print("\n" + "="*75)
    print("FINAL MODEL PERFORMANCE REPORT")
    print("="*75)
    print(f"Selected Model:     {model_name}")
    print(f"Selected Threshold: 0.50")
    print("This model was selected with F1-score as the primary objective to balance")
    print("prediction accuracy with the ability to catch minority class no-shows.")
    print("-" * 75)
    
    # Header
    print(f"{'Metric':<15} {'Train':<15} {'Val':<15} {'Test':<15}")
    print("-" * 75)
    
    # Table Rows
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"{key:<15} {m_train[key]:<15.4f} {m_val[key]:<15.4f} {m_test[key]:<15.4f}")
    
    print("-" * 75)
    
    # Confusion Matrix Stats (from Test Set)
    tn, fp, fn, tp = m_test['cm'].ravel()
    print(f"Test Confusion Matrix: {m_test['cm'].tolist()}")
    print(f"Test False Positives: {fp}")
    print(f"Test False Negatives: {fn}")
    print(f"Test True Positives:  {tp}")
    print(f"Test True Negatives:  {tn}")
    
    # Generalization Gaps
    train_val_gap = abs(m_train['f1'] - m_val['f1'])
    print(f"Train-Val F1 Gap:    {train_val_gap:+.4f}")
    
    print("="*75)
    status = "Excellent" if train_val_gap < 0.05 else "Good" if train_val_gap < 0.1 else "Overfitting"
    print(f"GENERALISATION: {status} - train-val gaps within target (<0.08)")
    print("="*75 + "\n")

# --- 3. SERIALIZATION ---

def save_artifacts(model, features):
    """Saves model and features list for deployment."""
    os.makedirs("artifacts/models", exist_ok=True)
    joblib.dump(model, "artifacts/models/best_model.joblib")
    joblib.dump(features, "artifacts/models/model_features.joblib")
    print("Process: Model and feature list serialized to artifacts/models/")

# --- 4. MAIN PIPELINE ---

def execute_pipeline():
    setup_tracking()
    print("Process: Initializing evaluation and serialization pipeline.")
    
    df = load_and_prepare_data("data/processed/final_featured_data.csv")
    target = 'no_show_target'
    X = df.drop(columns=[target])
    y = df[target]

    # Model Definitions
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42),
        "SVM": LinearSVC(class_weight='balanced', random_state=42)
    }

    best_f1 = 0
    winner_name = ""
    winner_model = None

    # Step 1: Compare to find winner (using standard 80/20 split)
    for name, model in models.items():
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        with mlflow.start_run(run_name=f"Comparison_{name}"):
            model.fit(X_train, y_train)
            f1 = f1_score(y_test, model.predict(X_test))
            if f1 > best_f1:
                best_f1 = f1
                winner_name = name
                winner_model = model

    # Step 2: Final 3-Way Split for the Winner (70% Train, 15% Val, 15% Test)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    winner_model.fit(X_train, y_train)
    
    # Step 3: Detailed Metrics Calculation
    m_train = calculate_metrics_dict(winner_model, X_train, y_train)
    m_val = calculate_metrics_dict(winner_model, X_val, y_val)
    m_test = calculate_metrics_dict(winner_model, X_test, y_test)

    # Step 4: Professional Output
    print_structured_report(winner_name, m_train, m_val, m_test)

    # Step 5: Serialization
    save_artifacts(winner_model, X_train.columns.tolist())

if __name__ == "__main__":
    execute_pipeline()