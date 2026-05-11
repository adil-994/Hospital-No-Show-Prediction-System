import pandas as pd
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

def run_clean_training():
    # 1. Setup Paths
    # This line ensures MLflow saves in your current project folder/mlruns
    base_dir = os.getcwd()
    mlflow.set_tracking_uri(f"file:///{os.path.join(base_dir, 'mlruns')}")
    mlflow.set_experiment("Hospital_NoShow_Analysis")

    # 2. Load Data
    data_path = "data/processed/final_featured_data.csv"
    if not os.path.exists(data_path):
        print("❌ Error: File not found. Run feature engineering first!")
        return
        
    df = pd.read_csv(data_path)
    df = pd.get_dummies(df) # Convert categories to numbers

    # 3. Organize our Clues (Features)
    medical_cols = [
        'Age', 'Gender', 'Hypertension', 'Diabetes', 'Alcoholism', 
        'Handicap', 'SMS_received', 'waiting_days', 'day_of_week', 'is_weekend'
    ]
    # Add the age groups created by get_dummies
    medical_cols += [col for col in df.columns if 'age_group' in col]
    
    weather_cols = [
        'temperature_2m_mean', 'precipitation_sum', 
        'relative_humidity_2m_mean', 'windspeed_10m_max', 'is_rainy'
    ]

    target = 'target'

    # --- EXPERIMENT A: Medical Only ---
    print("\n🧪 Training Experiment A (Medical Only)...")
    with mlflow.start_run(run_name="Medical_Only"):
        X = df[medical_cols]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        f1 = f1_score(y_test, preds)
        
        mlflow.log_param("features", "medical_only")
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")
        print(f"✅ Finished A. F1 Score: {f1:.4f}")

    # --- EXPERIMENT B: Medical + Weather ---
    print("🧪 Training Experiment B (Medical + Weather)...")
    with mlflow.start_run(run_name="With_Weather"):
        X = df[medical_cols + weather_cols]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        f1 = f1_score(y_test, preds)
        
        mlflow.log_param("features", "medical_plus_weather")
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")
        print(f"✅ Finished B. F1 Score: {f1:.4f}")

if __name__ == "__main__":
    run_clean_training()