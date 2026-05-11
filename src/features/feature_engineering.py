import pandas as pd
import numpy as np
import os

def engineer_features():
    input_path = "data/processed/medical_weather_merged.csv"
    output_path = "data/processed/final_featured_data.csv"

    if not os.path.exists(input_path):
        print(f"❌ Error: {input_path} not found. Did you run the merge script?")
        return

    df = pd.read_csv(input_path)
    print("🧠 Starting Feature Engineering...")

    # 1. Date Conversion (Added errors='coerce' for safety)
    df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'], errors='coerce')
    df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'], errors='coerce')

    # 2. Feature: Waiting Days 
    # Subtracting timestamps then getting .days
    df['waiting_days'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days
    df['waiting_days'] = df['waiting_days'].clip(lower=0)

    # 3. Feature: Day of the Week (0=Mon, 6=Sun)
    df['day_of_week'] = df['AppointmentDay'].dt.dayofweek
    
    # 4. Feature: Is Weekend?
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # 5. Feature: Age Grouping
    bins = [0, 18, 35, 60, 120]
    labels = ['Child', 'Young_Adult', 'Adult', 'Senior']
    df['age_group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

    # 6. Feature: Weather Intensity (Is_Rainy)
    if 'precipitation_sum' in df.columns:
        df['is_rainy'] = (df['precipitation_sum'] > 0).astype(int)

    # 7. Target Variable Encoding (Verified: False -> 1 (No-show))
    if 'Showed_up' in df.columns:
        # We handle both boolean and string versions of True/False just in case
        df['target'] = df['Showed_up'].apply(lambda x: 1 if x == False or str(x).lower() == 'false' else 0)
        print("🎯 Target encoded: 1 for No-show, 0 for Attended.")
    else:
        print("⚠️ Warning: 'Showed_up' column not found!")

    # 8. Encode Gender
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'F': 0, 'M': 1})

    # 9. FINAL CLEANUP
    # We must drop raw dates and the original target column so the model can't "cheat"
    cols_to_drop = ['ScheduledDay', 'AppointmentDay', 'Showed_up', 'No_show']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')

    # Save the final featured dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"✅ Feature Engineering complete!")
    print(f"📊 New columns added: ['waiting_days', 'day_of_week', 'is_weekend', 'age_group', 'is_rainy', 'target']")
    print(f"💾 Final file saved to: {output_path}")

if __name__ == "__main__":
    engineer_features()