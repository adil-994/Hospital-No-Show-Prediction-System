import pandas as pd
import os
import joblib  # Added for serialization
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import mlflow
import mlflow.sklearn

def setup_tuning_tracking():
    """
    Sets up a separate MLflow experiment for the tuning process.
    """
    base_dir = os.getcwd()
    mlflow.set_tracking_uri(f"file:///{os.path.join(base_dir, 'mlruns')}")
    mlflow.set_experiment("Random_Forest_Hyperparameter_Tuning")

def get_tuning_data(path):
    """
    Loads the full feature set (Medical + Weather) for the best model.
    """
    df = pd.read_csv(path)
    df = pd.get_dummies(df)
    
    # Define all features (Medical + Weather)
    features = [col for col in df.columns if col != 'no_show_target']
    X = df[features]
    y = df['no_show_target']
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def run_hyperparameter_search(X_train, y_train):
    """
    Searches for the best settings for the Random Forest.
    We check different depths and tree counts to find the 'Sweet Spot'.
    """
    # Define the 'Grid' of settings to try
    param_grid = {
        'n_estimators': [100, 200, 300], # 3x4x3 = 36 -> 5
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'class_weight': ['balanced'] 
    }

    rf = RandomForestClassifier(random_state=42)

    # RandomizedSearchCV tries a random subset of combinations
    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_grid,
        n_iter=5, 
        scoring='f1',
        cv=3, 
        verbose=1,
        random_state=42,
        n_jobs=-1 
    )

    print("Process: Searching for the best model parameters. This may take a moment.")
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_

def log_best_model(model, params, X_test, y_test):
    """
    Logs the final 'optimized' model to MLflow.
    """
    with mlflow.start_run(run_name="Optimized_Random_Forest"):
        preds = model.predict(X_test)
        f1 = f1_score(y_test, preds)
        
        mlflow.log_params(params)
        mlflow.log_metric("optimized_f1_score", f1)
        mlflow.sklearn.log_model(model, "final_tuned_model")
        
        print(f"Success: Tuning complete. Optimized F1-Score: {f1:.4f}")

def save_final_artifacts(model, feature_list):
    """
    Saves the final model and feature names to the artifacts folder.
    This is required for the Flask application to function correctly.
    """
    model_dir = "artifacts/models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    # Save the model object
    joblib.dump(model, os.path.join(model_dir, "best_model.joblib"))
    
    # Save the feature names to ensure the Flask app uses the same column order
    joblib.dump(feature_list, os.path.join(model_dir, "model_features.joblib"))
    
    print(f"Process: Final model and features saved to {model_dir}")

def execute_tuning_pipeline():
    setup_tuning_tracking()
    
    X_train, X_test, y_train, y_test = get_tuning_data("data/processed/final_featured_data.csv")
    
    # 1. Find the best version of the model
    best_rf_model, best_settings = run_hyperparameter_search(X_train, y_train)
    
    # 2. Log results to MLflow
    log_best_model(best_rf_model, best_settings, X_test, y_test)
    
    # 3. Serialize (Save) for Deployment
    # We pass X_train.columns to save the exact feature order
    save_final_artifacts(best_rf_model, X_train.columns.tolist())
    
    print("-" * 50)
    print("Final Best Settings Found:")
    for param, value in best_settings.items():
        print(f"{param}: {value}")
    print("-" * 50)

if __name__ == "__main__":
    execute_tuning_pipeline()