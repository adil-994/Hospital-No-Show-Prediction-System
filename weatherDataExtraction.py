import requests
import pandas as pd

url = "https://archive-api.open-meteo.com/v1/archive?latitude=-20.3155&longitude=-40.3128&start_date=2016-04-01&end_date=2016-06-30&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,rain_sum,relative_humidity_2m_mean,windspeed_10m_max&timezone=auto"

response = requests.get(url)
data = response.json()

weather_df = pd.DataFrame(data['daily'])

weather_df.to_csv("vitoria_weather_apr_jun_2016.csv", index=False)

print(weather_df.head())