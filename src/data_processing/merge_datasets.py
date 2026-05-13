import pandas as pd
import os

def prepare_date_columns(df_med, df_weather):
    """
    Ensures both datasets have a 'Date' column in the same format
    so we can link them together.
    """
    # Convert medical timestamps to just dates (YYYY-MM-DD)
    df_med['AppointmentDay'] = pd.to_datetime(df_med['AppointmentDay']).dt.date
    
    # Ensure weather time is also just a date
    df_weather['time'] = pd.to_datetime(df_weather['time']).dt.date
    
    return df_med, df_weather

def verify_merge(df):
    """
    Checks the final dataset for any issues, like missing weather info.
    """
    missing_count = df['temperature_2m_mean'].isnull().sum()

    if missing_count > 0:
        print(f"Warning: {missing_count} records are missing weather data.")
    else:
        print("All medical records successfully matched with weather data.")

def merge_data():
    """
    Main function to join medical records with weather conditions.
    """
    med_path = "data/processed/cleaned_medical.csv"
    weather_path = "data/processed/cleaned_weather.csv"
    output_path = "data/processed/medical_weather_merged.csv"

    # 1. Load
    if not os.path.exists(med_path) or not os.path.exists(weather_path):
        print("Required processed files were not found.")
        return

    df_med = pd.read_csv(med_path)
    df_weather = pd.read_csv(weather_path)

    # 2. Format
    df_med, df_weather = prepare_date_columns(df_med, df_weather)

    # 3. Join (Left Join keeps all medical records)
    print("Merging medical and weather datasets...")

    merged_df = pd.merge(
        df_med,
        df_weather,
        left_on='AppointmentDay',
        right_on='time',
        how='left'
    )

    # 4. Cleanup
    # We drop the extra 'time' column from weather since we already have 'AppointmentDay'
    if 'time' in merged_df.columns:
        merged_df = merged_df.drop(columns=['time'])

    # 5. Verify & Save
    verify_merge(merged_df)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged_df.to_csv(output_path, index=False)

    print(f"Merged dataset saved successfully. Final shape: {merged_df.shape}")

if __name__ == "__main__":
    merge_data()