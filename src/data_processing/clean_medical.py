import pandas as pd
import os

def clean_medical_data():
    input_path = "data/raw/healthcare_noshows_appt.csv"
    output_path = "data/processed/cleaned_medical.csv"

    if not os.path.exists(input_path):
        print(f"❌ Error: {input_path} not found.")
        return

    # 1. Load the raw medical data
    df = pd.read_csv(input_path)
    print(f"📊 Original Medical Shape: {df.shape}")

    # 2. Fix Typos in Column Names (Standard for this dataset)
    # Hipertension -> Hypertension
    # Handcap -> Handicap
    df = df.rename(columns={
        'Hipertension': 'Hypertension',
        'Handcap': 'Handicap',
        'No-show': 'No_show'  # Standardizing for easier coding
    })

    # 3. Data Quality: Fix Age
    # There is often a record with Age -1 (an error). We remove it.
    df = df[df['Age'] >= 0]
    
    # 4. Data Quality: Convert Dates to Datetime objects
    df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
    df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])

    # 5. Drop Identifiers
    # PatientId and AppointmentID are unique to individuals and don't help 
    # predict patterns. We remove them to prevent "data leakage".
    cols_to_drop = ['PatientId', 'AppointmentID']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # 6. Check for duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"🧹 Removing {duplicates} duplicate rows...")
        df = df.drop_duplicates()

    # 7. Final Check
    print(f"✅ Cleaned Medical Shape: {df.shape}")
    print(f"📋 Columns: {df.columns.tolist()}")

    # 8. Save the cleaned file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"💾 Saved cleaned medical data to: {output_path}")

if __name__ == "__main__":
    clean_medical_data()