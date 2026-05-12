import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

def load_final_data(path):
    """
    Loads the prepared dataset for the final evaluation.
    """
    df = pd.read_csv(path)
    df = pd.get_dummies(df)
    
    X = df.drop(columns=['no_show_target'])
    y = df['no_show_target']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def plot_confusion_matrix(model, X_test, y_test):
    """
    Creates a visual square showing True Positives, True Negatives, 
    and where the model was confused.
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
    print("Final Analysis: Confusion Matrix saved to reports/figures/confusion_matrix.png")

def plot_final_feature_importance(model, feature_names):
    """
    Ranks the features for the tuned model. 
    This shows which clues were the most influential in the final version.
    """
    importance = model.feature_importances_
    data = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    data = data.sort_values(by='Importance', ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=data, palette='viridis')
    plt.title('Final Feature Ranking: What Drives No-Shows?', pad=20)
    
    plt.savefig('reports/figures/final_feature_importance.png', bbox_inches='tight')
    plt.show()
    print("Final Analysis: Feature Importance chart saved to reports/figures/final_feature_importance.png")

def run_final_evaluation():
    """
    Orchestrates the final evaluation using the best parameters 
    discovered during the tuning phase.
    """
    print("Process: Starting final model evaluation.")
    
    X_train, X_test, y_train, y_test = load_final_data("data/processed/final_featured_data.csv")
    
    # We use the 'Best Settings' found in the previous tuning step
    final_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=10,
        class_weight='balanced',
        random_state=42
    )
    
    final_model.fit(X_train, y_train)
    
    # Generate Visuals
    plot_confusion_matrix(final_model, X_test, y_test)
    plot_final_feature_importance(final_model, X_train.columns)
    
    # Print the text-based report for the Viva documentation
    print("\nDetailed Classification Report:")
    predictions = final_model.predict(X_test)
    print(classification_report(y_test, predictions))

if __name__ == "__main__":
    run_final_evaluation()