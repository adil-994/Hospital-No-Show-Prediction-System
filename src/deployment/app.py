from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# --- LOAD MODELS ---

def load_artifacts():
    """
    Loads the saved model and the list of features.
    It is vital to load the feature list so the data stays in the correct order.
    """
    model_path = "artifacts/models/best_model.joblib"
    features_path = "artifacts/models/model_features.joblib"
    
    model = joblib.load(model_path)
    features = joblib.load(features_path)
    return model, features

# Load once when the app starts
model, model_features = load_artifacts()

# --- ROUTES ---

@app.route('/')
def home():
    """Renders the main input form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Processes form data, matches it to model features, and returns a prediction.
    """
    try:
        # 1. Get data from the form
        user_input = request.form.to_dict()
        
        # 2. Create a blank dataframe with all required features (filled with 0)
        # This ensures that even columns not in the form (like age_groups) exist.
        input_df = pd.DataFrame(0, index=[0], columns=model_features)
        
        # 3. Fill the dataframe with the user's data
        # We convert numeric strings to floats/ints
        for key, value in user_input.items():
            if key in input_df.columns:
                input_df.at[0, key] = float(value)

        # 4. Handle Categorical Logic (Age Group Example)
        # If age is 70, we set age_group_Senior to 1
        age = float(user_input.get('Age', 0))
        if age >= 60 and 'age_group_Senior' in input_df.columns:
            input_df.at[0, 'age_group_Senior'] = 1
        elif age >= 35 and 'age_group_Adult' in input_df.columns:
            input_df.at[0, 'age_group_Adult'] = 1

        # 5. Make the Prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        # 6. Prepare the result message
        if prediction == 1:
            result_text = "High Risk of No-Show"
            result_class = "danger"
        else:
            result_text = "Likely to Attend"
            result_class = "success"

        return render_template('index.html', 
                               prediction_text=result_text, 
                               confidence=f"{probability:.1%}",
                               result_class=result_class)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    print("STARTING: Flask server is initializing on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=False)