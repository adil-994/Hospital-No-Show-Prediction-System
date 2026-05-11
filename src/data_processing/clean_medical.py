import pandas as pd
import os

def fix_column_typos(df):
    """
    Renames columns to fix typos found in the original Kaggle dataset.
    This makes the code more professional and easier to read.
    """
    df = df.rename(columns={
        'Hipertension': 'Hypertension',
        'Handcap': 'Handicap',
        'No-show': 'No_show',
        'Showed_up': 'Showed_up' # Ensuring consistency
    })
    return df

def remove_invalid_records(df):
    """
    Removes rows that contain logical errors, like negative age.
    """
    initial_count = len(df)
    df = df[df['Age'] >= 0]
    
    removed = initial_count - len(df)
    if removed > 0:
        print(f"🧹 Removed {removed} records with negative Age.")
    return df

def drop_unnecessary_columns(df):
    """
    Removes columns that don't help the model learn general patterns.
    """
    # 1. We drop IDs because they are unique to individuals (Data Leakage)
    # 2. We drop Date.diff because we will calculate our own 'waiting_days' later.
    cols_to_remove = ['PatientId', 'AppointmentID', 'Date.diff']
    
    # We only drop them if they actually exist in the file
    existing_cols = [c for c in cols_to_remove if c in df.columns]
    df = df.drop(columns=existing_cols)
    
    return df

def clean_medical_data():
    """
    The main function that orchestrates the medical data cleaning process.
    """
    input_path = "data/raw/healthcare_noshows_appt.csv"
    output_path = "data/processed/cleaned_medical.csv"

    if not os.path.exists(input_path):
        print(f"❌ Error: {input_path} not found.")
        return

    # Load
    df = pd.read_csv(input_path)
    print(f"📊 Starting cleanup. Original rows: {len(df)}")

    # Process step-by-step
    df = fix_column_typos(df)
    df = remove_invalid_records(df)
    df = drop_unnecessary_columns(df)
    
    # Remove duplicates
    if df.duplicated().sum() > 0:
        df = df.drop_duplicates()
        print("🧹 Duplicate rows removed.")

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Clean medical data saved to: {output_path}")

if __name__ == "__main__":
    clean_medical_data()