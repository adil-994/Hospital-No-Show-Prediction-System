import pandas as pd
import numpy as np
import os

def create_time_features(df):
    """
    Translates raw dates into meaningful numbers.
    The model can't understand 'May 11th', but it can understand '5 days wait'.
    """
    # 1. Convert strings to actual date objects
    df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'], errors='coerce')
    df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'], errors='coerce')

    # 2. Calculate Waiting Days (Gap between booking and appointment)
    # We use .dt.days to get a simple integer
    df['waiting_days'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days
    
    # 3. Clean negative values (errors where booking was after appointment)
    df['waiting_days'] = df['waiting_days'].clip(lower=0)

    # 4. Extract Day of Week (0=Monday, 6=Sunday)
    df['day_of_week'] = df['AppointmentDay'].dt.dayofweek
    
    # 5. Create a Weekend Flag (Saturdays and Sundays)
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    return df

def categorize_age(df):
    """
    Groups ages into life stages. Patterns of hospital visits 
    usually change based on whether the patient is a child, adult, or senior.
    """
    bins = [0, 18, 35, 60, 120]
    labels = ['Child', 'Young_Adult', 'Adult', 'Senior']
    
    df['age_group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    return df

def simplify_weather(df):
    """
    Simplifies weather data into an 'Intensity' flag.
    Sometimes the model just needs to know 'Is it raining?' rather than 'How many mm?'.
    """
    if 'precipitation_sum' in df.columns:
        # 1 if there was any rain, 0 if it was dry
        df['is_rainy'] = (df['precipitation_sum'] > 0).astype(int)
    return df

def encode_categorical_data(df):
    """
    Converts text and booleans into numbers (0 and 1).
    Machine Learning models only speak the language of math.
    """
    # 1. Map Gender (Female=0, Male=1)
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'F': 0, 'M': 1})

    # 2. Map Target: We want to predict NO-SHOWS.
    # Therefore: False (didn't show) becomes 1, True (showed up) becomes 0.
    if 'Showed_up' in df.columns:
        df['no_show_target'] = df['Showed_up'].apply(lambda x: 1 if x == False or str(x).lower() == 'false' else 0)
    
    return df

def engineer_features():
    """
    Main orchestration function for Feature Engineering.
    """
    input_path = "data/processed/medical_weather_merged.csv"
    output_path = "data/processed/final_featured_data.csv"

    if not os.path.exists(input_path):
        print(f"❌ Error: {input_path} not found.")
        return

    # Load
    df = pd.read_csv(input_path)
    print(f"🧠 Starting Feature Engineering on {len(df)} rows...")

    # Execute Modules
    df = create_time_features(df)
    df = categorize_age(df)
    df = simplify_weather(df)
    df = encode_categorical_data(df)

    # Final Cleanup: Remove raw columns that the model can't use
    # We keep 'AppointmentDay' for now (plots) but remove 'ScheduledDay' and 'Showed_up'
    cols_to_drop = ['ScheduledDay', 'Showed_up', 'No_show']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Feature Engineering complete. Saved to: {output_path}")

if __name__ == "__main__":
    engineer_features()