import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def check_data_health(df):
    """
    Checks for missing values and general structure.
    This is the first thing a doctor does: check the vitals.
    """
    print("📋 Data Health Check:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("✅ No missing values found!")
    else:
        print("⚠️ Missing values detected:")
        print(missing[missing > 0])

def plot_class_balance(df):
    """
    Visualizes the 'Imbalance'. 
    If 80% show up and 20% don't, the model might get lazy.
    """
    plt.figure(figsize=(7, 5))
    sns.countplot(x='no_show_target', data=df, palette='Set2')
    plt.title('Class Imbalance: Attended (0) vs No-Show (1)')
    plt.xlabel('Patient Outcome')
    plt.ylabel('Number of Appointments')
    
    # Calculate percentages for the report
    total = len(df)
    no_shows = df['no_show_target'].sum()
    print(f"📈 Balance: {no_shows/total:.1%} of patients are No-Shows.")
    
    plt.savefig('reports/figures/class_balance.png')
    plt.show()

def plot_waiting_impact(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='no_show_target', y='waiting_days', data=df, palette='Pastel1')
    plt.title('Impact of Waiting Time on Attendance', pad=20)
    plt.xlabel('Patient Outcome (0=Show, 1=No-Show)')
    plt.ylabel('Days Waited')
    plt.ylim(0, 60)
    
    # This command prevents labels from cutting off
    plt.tight_layout() 
    
    plt.savefig('reports/figures/waiting_impact.png', bbox_inches='tight') # bbox_inches captures everything
    plt.show()

def plot_weather_correlation(df):
    # Shorten the names just for the graph so they don't cut off
    plot_df = df[['no_show_target', 'temperature_2m_mean', 'precipitation_sum', 'is_rainy', 'waiting_days']].copy()
    plot_df.columns = ['No-Show', 'Temp', 'Precip', 'Is Rainy', 'Wait Days']
    
    corr = plot_df.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True)
    
    plt.title('Correlation: Weather & Waiting Time vs No-Show', pad=20)
    
    # Rotate labels so they fit
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    # 'bbox_inches' is the secret to fixing cut-off edges
    plt.savefig('reports/figures/weather_correlation.png', bbox_inches='tight') 
    plt.show()

def run_full_eda():
    """
    Main function to generate the EDA report.
    """
    data_path = "data/processed/final_featured_data.csv"
    if not os.path.exists(data_path):
        print("❌ Error: Run feature engineering first!")
        return

    df = pd.read_csv(data_path)
    os.makedirs('reports/figures', exist_ok=True)

    check_data_health(df)
    plot_class_balance(df)
    plot_waiting_impact(df)
    plot_weather_correlation(df)
    
    print("✅ EDA complete. All charts saved in reports/figures/")

if __name__ == "__main__":
    run_full_eda()