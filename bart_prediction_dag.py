import os
import io
import pandas as pd
import numpy as np
import requests
import joblib
import holidays
from google.cloud import storage
from datetime import datetime, timedelta
from airflow.decorators import dag, task

# Default arguments dictate how Airflow handles retries and scheduling
default_args = {
    'owner': 'data_science_team',
    'depends_on_past': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

@dag(
    default_args=default_args, 
    schedule='@daily', 
    start_date=datetime(2026, 3, 3), 
    catchup=False, 
    tags=['bart', 'forecasting', 'ml']
)
def bart_ridership_forecast():

    @task()
    def extract_weather_data() -> list:
        """Task 1: Fetch Stations from GCS and pull 5-day weather API data."""
        bucket_name = os.environ.get('GCP_BUCKET_NAME')
        weather_api_key = os.environ.get('WEATHER_KEY')

        # 1. Get Stations
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob("weather_data/bart-stations.csv")
        stations = pd.read_csv(io.StringIO(blob.download_as_text()))

        # 2. Get Weather
        future_docs = []
        for i in stations.index:
            params = {
                "key": weather_api_key,
                "q": f"{stations.loc[i,'lat']},{stations.loc[i,'lng']}",
                "days": 5
            }
            response = requests.get(url="http://api.weatherapi.com/v1/forecast.json", params=params).json()

            if 'forecast' in response:
                for day in response['forecast']['forecastday']:
                    for hour_data in day['hour']:
                        future_docs.append({
                            "timestamp": hour_data['time'], # Keep as string for Airflow XCom passing
                            "temp_f": hour_data['temp_f'],
                            "precip_in": hour_data['precip_in'],
                            "humidity": hour_data['humidity'],
                            "wind_mph": hour_data['wind_mph']
                        })
        
        return future_docs # Returns a list of dicts safely to Airflow XCom

    @task()
    def transform_features(future_docs: list) -> dict:
        """Task 2: Aggregate weather and engineer temporal features."""
        weather_df = pd.DataFrame(future_docs)
        weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])

        system_forecast = weather_df.groupby('timestamp').agg(
            avg_temp_f=('temp_f', 'mean'),
            max_precip_in=('precip_in', 'max'), 
            avg_humidity=('humidity', 'mean'),
            avg_wind_mph=('wind_mph', 'mean')
        ).reset_index()

        system_forecast = system_forecast.set_index('timestamp').sort_index()

        # Engineering
        system_forecast['hour_of_day'] = system_forecast.index.hour
        system_forecast['day_of_week'] = system_forecast.index.dayofweek
        system_forecast['month'] = system_forecast.index.month
        system_forecast['is_weekend'] = (system_forecast.index.dayofweek >= 5).astype(int)

        hours_in_day = 24
        system_forecast['hour_sin'] = np.sin(2 * np.pi * system_forecast['hour_of_day'] / hours_in_day)
        system_forecast['hour_cos'] = np.cos(2 * np.pi * system_forecast['hour_of_day'] / hours_in_day)

        unique_years = system_forecast.index.year.unique().tolist()
        ca_holidays = holidays.US(state='CA', years=unique_years)
        system_forecast['holiday_name'] = system_forecast.index.map(lambda x: ca_holidays.get(x))
        system_forecast['holiday_name'] = system_forecast['holiday_name'].fillna('None')
        system_forecast['is_holiday'] = system_forecast['holiday_name'].apply(lambda x: 0 if x == 'None' else 1)

        system_forecast_encoded = pd.get_dummies(system_forecast)
        
        # Convert back to dict with string timestamps for Airflow XCom passing
        system_forecast_encoded.index = system_forecast_encoded.index.astype(str)
        return system_forecast_encoded.to_dict(orient='index')

    @task()
    def generate_and_store_predictions(prepared_data_dict: dict):
        """Task 3: Download model, predict, and upload final CSV to GCS."""
        bucket_name = os.environ.get('GCP_BUCKET_NAME')
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        # 1. Rebuild DataFrame from Airflow XCom
        weather_data = pd.DataFrame.from_dict(prepared_data_dict, orient='index')
        weather_data.index = pd.to_datetime(weather_data.index)

        # 2. Download Model Artifacts to /tmp/
        model_path, features_path = "/tmp/xgboost_bart_champion.joblib", "/tmp/bart_model_features.joblib"
        bucket.blob("models/xgboost_bart_champion.joblib").download_to_filename(model_path)
        bucket.blob("models/bart_model_features.joblib").download_to_filename(features_path)

        # 3. Predict
        model = joblib.load(model_path)
        training_columns = joblib.load(features_path)
        
        weather_data = weather_data.reindex(columns=training_columns, fill_value=0).astype(float)
        predictions = model.predict(weather_data)

        final_output = pd.DataFrame({
            'timestamp': weather_data.index,
            'predicted_system_riders': np.round(predictions).astype(int) 
        }).set_index('timestamp')

        # 4. Upload to GCS
        blob = bucket.blob("predictions/latest_forecast.csv")
        blob.upload_from_string(final_output.to_csv(), content_type='text/csv')
        print(f"Success! Uploaded forecast for {len(final_output)} hours.")

    # Define the execution flow (Dependencies)
    raw_weather_data = extract_weather_data()
    engineered_features = transform_features(raw_weather_data)
    generate_and_store_predictions(engineered_features)

# Instantiate the DAG
bart_dag = bart_ridership_forecast()