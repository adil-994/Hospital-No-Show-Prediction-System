import pandas as pd
import os

def merge_data():
    # Define paths based on your structure
    medical_path = "data/raw/healthcare_noshows_appt.csv"
    weather_path = "data/raw/vitoria_weather_apr_jun_2016.csv"
    output_path = "data/processed/medical_weather_merged.csv"

    # 1. Load datasets
    print("🔄 Loading datasets...")
    df_med = pd.read_csv(medical_path)
    df_weather = pd.read_csv(weather_path)

    # 2. Standardize Date Formats
    # In medical data, AppointmentDay often has time (2016-04-29T18:38:08Z). We need only the date.
    df_med['AppointmentDay'] = pd.to_datetime(df_med['AppointmentDay']).dt.date
    
    # In weather data, 'time' is usually already a date string
    df_weather['time'] = pd.to_datetime(df_weather['time']).dt.date

    # 3. Perform Merge
    # We join on medical date and weather time
    print("🔗 Merging datasets on date...")
    merged_df = pd.merge(
        df_med, 
        df_weather, 
        left_on='AppointmentDay', 
        right_on='time', 
        how='left'
    )

    # 4. Cleanup
    # Drop the duplicate 'time' column from weather after merge
    if 'time' in merged_df.columns:
        merged_df.drop(columns=['time'], inplace=True)

    # 5. Check for missing values (if any appointment date didn't have weather data)
    missing_weather = merged_df['temperature_2m_mean'].isnull().sum()
    if missing_weather > 0:
        print(f"⚠️ Warning: {missing_weather} rows have no weather data. Check date ranges!")
    else:
        print("✅ Merge successful! No missing weather values.")

    # 6. Save final file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    print(f"💾 Saved merged dataset to: {output_path}")
    print(f"📊 Final Shape: {merged_df.shape}")

if __name__ == "__main__":
    merge_data()