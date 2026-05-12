import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier

def live_prediction_demo():
    """
    A simple demonstration of how the model can be used for new patients.
    It takes a sample input and predicts if the patient will be a No-show.
    """
    print("Demo: Initializing the prediction system.")
    
    # In a real app, we would load a saved .pkl file. 
    # For this demo, we will use the best settings we discovered.
    model = RandomForestClassifier(
        n_estimators=200, 
        max_depth=20, 
        class_weight='balanced', 
        random_state=42
    )

    # 1. Load the training data to 'fit' the model once for the demo
    # (In deployment, you would just load the saved model file)
    train_df = pd.read_csv("data/processed/final_featured_data.csv")
    train_df = pd.get_dummies(train_df)
    
    X = train_df.drop(columns=['no_show_target'])
    y = train_df['no_show_target']
    model.fit(X, y)

    # 2. Create a 'Fake Patient' to test
    # Scenario: A 75-year-old patient, booked 20 days ago, it is a rainy day.
    print("\nScenario: 75-year-old patient, 20-day wait, rainy weather.")
    
    # We must provide all columns the model expects
    sample_patient = pd.DataFrame([X.iloc[0].values], columns=X.columns)
    
    # Update specific values for our test case
    sample_patient['Age'] = 20
    sample_patient['waiting_days'] = 6
    sample_patient['is_rainy'] = 1
    sample_patient['SMS_received'] = 0

    # 3. Make Prediction
    prediction = model.predict(sample_patient)[0]
    probability = model.predict_proba(sample_patient)[0][1]

    print("-" * 30)
    if prediction == 1:
        print(f"Result: HIGH RISK of No-Show")
        print(f"Confidence: {probability:.1%}")
    else:
        print(f"Result: LIKELY TO ATTEND")
        print(f"Confidence: {(1-probability):.1%}")
    print("-" * 30)

if __name__ == "__main__":
    live_prediction_demo()