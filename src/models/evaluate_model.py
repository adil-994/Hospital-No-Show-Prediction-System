import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

def load_testing_data(path):
    """
    Loads the prepared dataset and splits it to get the test set.
    We use the same random_state to ensure we evaluate on unseen data.
    """
    df = pd.read_csv(path)
    df = pd.get_dummies(df)
    
    X = df.drop(columns=['no_show_target'])
    y = df['no_show_target']
    
    # We only need the test portion for the final evaluation
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_test, y_test

def plot_confusion_matrix(model, X_test, y_test):
    """
    Creates a visual square showing how many no-shows were correctly caught.
    """
    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    
    plt.title('Confusion Matrix: Final Model Performance', pad=20)
    plt.xlabel('Predicted Outcome (0=Show, 1=No-Show)')
    plt.ylabel('Actual Outcome (0=Show, 1=No-Show)')
    
    os.makedirs('reports/figures', exist_ok=True)
    plt.savefig('reports/figures/confusion_matrix.png', bbox_inches='tight')
    plt.show()
    print("Success: Confusion Matrix saved to reports/figures/")

def plot_final_feature_importance(model, feature_names):
    """
    Ranks the features based on the saved model's weights.
    """
    importance = model.feature_importances_
    data = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    data = data.sort_values(by='Importance', ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=data, palette='viridis')
    plt.title('Final Feature Ranking: What Drives No-Shows?', pad=20)
    
    plt.savefig('reports/figures/final_feature_importance.png', bbox_inches='tight')
    plt.show()
    print("Success: Feature Importance chart saved to reports/figures/")

def run_final_evaluation():
    """
    Loads the best tuned model from artifacts and performs a final evaluation.
    No retraining is performed here.
    """
    model_path = "artifacts/models/best_model.joblib"
    data_path = "data/processed/final_featured_data.csv"

    # 1. Check if the tuned model exists
    if not os.path.exists(model_path):
        print(f"Error: Tuned model not found at {model_path}. Run the tuning script first!")
        return

    print("Process: Loading the final tuned model for evaluation.")
    
    # 2. Load the serialized model
    final_model = joblib.load(model_path)
    
    # 3. Load the test data
    X_test, y_test = load_testing_data(data_path)
    
    # 4. Generate Visuals using the loaded model
    plot_confusion_matrix(final_model, X_test, y_test)
    plot_final_feature_importance(final_model, X_test.columns)
    
    # 5. Print the text-based classification report
    print("\nDetailed Classification Report (Final Model):")
    predictions = final_model.predict(X_test)
    print(classification_report(y_test, predictions))

if __name__ == "__main__":
    run_final_evaluation()