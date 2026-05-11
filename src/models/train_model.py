import pandas as pd
import os
import warnings
import logging

# Suppress all warnings and MLflow noise for a clean terminal output
warnings.filterwarnings("ignore")
logging.getLogger("mlflow").setLevel(logging.ERROR)

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score

def setup_tracking():
    """
    Sets up the MLflow experiment location.
    Using a local directory keeps all experimental history organized.
    """
    base_dir = os.getcwd()
    mlflow.set_tracking_uri(f"file:///{os.path.join(base_dir, 'mlruns')}")
    mlflow.set_experiment("Hospital_NoShow_Refined_Analysis")

def load_and_prepare_data(path):
    """
    Loads the data and converts categorical text into numerical columns.
    """
    if not os.path.exists(path):
        print(f"Error: Data file not found at {path}")
        return None
    df = pd.read_csv(path)
    return pd.get_dummies(df)

def get_feature_sets(df):
    """
    Defines the two experimental feature sets: Medical Only and Medical + Weather.
    """
    medical = ['Age', 'Gender', 'Hypertension', 'Diabetes', 'Alcoholism', 
               'Handicap', 'SMS_received', 'waiting_days', 'day_of_week', 'is_weekend']
    # Dynamically add age group columns created by pd.get_dummies
    medical += [col for col in df.columns if 'age_group' in col]
    
    weather = ['temperature_2m_mean', 'precipitation_sum', 
               'relative_humidity_2m_mean', 'windspeed_10m_max', 'is_rainy']
    
    return medical, weather

def train_and_log(model, X_train, X_test, y_train, y_test, run_name):
    """
    Trains the model and logs performance metrics to MLflow.
    """
    with mlflow.start_run(run_name=run_name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        f1 = f1_score(y_test, preds)
        acc = accuracy_score(y_test, preds)
        
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")
        
        return f1, acc

def execute_pipeline():
    setup_tracking()
    print("Pipeline: Starting Model Evaluation...")
    
    df = load_and_prepare_data("data/processed/final_featured_data.csv")
    if df is None:
        return

    med_features, weather_features = get_feature_sets(df)
    target = 'no_show_target'
    y = df[target]
    
    # We define our models here. 
    # All models use 'balanced' weights to handle the class imbalance.
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42),
        "SVM": LinearSVC(class_weight='balanced', C=1.0, random_state=42)
    }

    best_f1 = 0
    best_model_name = ""

    for name, model in models.items():
        print(f"\nProcessing Algorithm: {name}")
        
        # Experiment A: Medical Only
        X_med = df[med_features]
        X_tr_a, X_te_a, y_tr_a, y_te_a = train_test_split(X_med, y, test_size=0.2, random_state=42)
        f1_a, acc_a = train_and_log(model, X_tr_a, X_te_a, y_tr_a, y_te_a, f"{name}_Medical_Only")
        print(f" - Medical Only   | F1: {f1_a:.4f} | Acc: {acc_a:.4f}")

        # Experiment B: Medical + Weather
        X_all = df[med_features + weather_features]
        X_tr_b, X_te_b, y_tr_b, y_te_b = train_test_split(X_all, y, test_size=0.2, random_state=42)
        f1_b, acc_b = train_and_log(model, X_tr_b, X_te_b, y_tr_b, y_te_b, f"{name}_With_Weather")
        print(f" - With Weather   | F1: {f1_b:.4f} | Acc: {acc_b:.4f}")

        # Final winner selection based on the Weather-enhanced experiment
        if f1_b > best_f1:
            best_f1 = f1_b
            best_model_name = name

    print("-" * 50)
    print(f"Final Selection: {best_model_name} is the best performing model.")
    print(f"Reasoning: Highest F1-score achieved ({best_f1:.4f}).")
    print("-" * 50)

if __name__ == "__main__":
    execute_pipeline()