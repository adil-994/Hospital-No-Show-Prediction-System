import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# --- LOAD MODELS (Bulletproof Version) ---
def load_artifacts():
    # In Docker, we are in /app. We need to look for /app/artifacts/...
    # This logic works both on Windows and inside Docker.
    
    # Get the directory where app.py is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels to reach the project root (LAB_PROJECT)
    root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
    
    model_path = os.path.join(root_dir, "artifacts", "models", "best_model.joblib")
    features_path = os.path.join(root_dir, "artifacts", "models", "model_features.joblib")
    
    print(f"DEBUG: Looking for model at: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"FATAL: Model file not found at {model_path}")
        
    model = joblib.load(model_path)
    features = joblib.load(features_path)
    return model, features

# Try to load. If this fails, the error will show in 'docker logs'
try:
    model, model_features = load_artifacts()
    print("SUCCESS: Model and Features loaded successfully.")
except Exception as e:
    print(f"FAILURE: Could not load artifacts. Error: {e}")
    # We exit if model can't be loaded
    exit(1) 

# ... rest of your routes (home, predict) stay the same ...

if __name__ == "__main__":
    print("STARTING: Flask server is initializing on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=False)