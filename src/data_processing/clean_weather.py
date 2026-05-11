import pandas as pd
import os

def clean_weather_data():
    input_path = "data/raw/vitoria_weather_apr_jun_2016.csv"
    output_path = "data/processed/cleaned_weather.csv"

    if not os.path.exists(input_path):
        print(f"❌ Error: {input_path} not found.")
        return

    # 1. Load the raw weather data
    df = pd.read_csv(input_path)
    print(f"📊 Original Weather Columns: {df.columns.tolist()}")

    # 2. Features to remove (as requested)
    to_remove = [
        'temperature_2m_max', 
        'temperature_2m_min', 
        'rain_sum'
    ]

    # Drop columns only if they exist in the dataframe
    df_cleaned = df.drop(columns=[col for col in to_remove if col in df.columns])

    # 3. Ensure 'time' is in date format (YYYY-MM-DD)
    if 'time' in df_cleaned.columns:
        df_cleaned['time'] = pd.to_datetime(df_cleaned['time']).dt.date
    
    # 4. Final selection check
    # We want: time, temperature_2m_mean, precipitation_sum, relative_humidity_2m_mean, windspeed_10m_max
    print(f"✅ Cleaned Weather Columns: {df_cleaned.columns.tolist()}")

    # 5. Save the cleaned file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_cleaned.to_csv(output_path, index=False)
    print(f"💾 Saved cleaned weather data to: {output_path}")

if __name__ == "__main__":
    clean_weather_data()