import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import os

def create_report_plots():
    # 1. Load the data
    df = pd.read_csv("data/processed/final_featured_data.csv")
    df_numeric = pd.get_dummies(df) # Convert categories to numbers for the model
    
    X = df_numeric.drop(columns=['target'])
    y = df_numeric['target']

    # 2. Use the Random Forest model
    # We use this because it's the 'best' at explaining itself
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X, y)

    # 3. Calculate which features were most important
    features = X.columns
    importance = model.feature_importances_
    
    # Put them in a table and sort them
    plot_data = pd.DataFrame({'Feature': features, 'Importance': importance})
    plot_data = plot_data.sort_values(by='Importance', ascending=False).head(10) # Top 10

    # 4. Create the Bar Chart
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=plot_data, palette='magma')
    
    plt.title('Top 10 Factors Influencing Patient No-Shows')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature Name')
    
    # Save it to your reports folder
    os.makedirs('reports/figures', exist_ok=True)
    plt.savefig('reports/figures/feature_importance.png')
    print("🎨 Plot saved to reports/figures/feature_importance.png")
    plt.show()

if __name__ == "__main__":
    create_report_plots()