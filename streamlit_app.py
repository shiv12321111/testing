import openmeteo_requests
import requests_cache
import pandas as pd
import plotly.express as px
import streamlit as st

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Define API parameters
url = "https://climate-api.open-meteo.com/v1/climate"
params = {
    "latitude": [52.5244, 52.374, 50.8505],  # Berlin, Amsterdam, Brussels
    "longitude": [13.4105, 4.8897, 4.3488],
    "start_date": "1950-01-01",
    "end_date": "2050-12-31",
    "models": ["CMCC_CM2_VHR4", "EC_Earth3P_HR"],
    "daily": ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min"]
}

# Fetch data from the API
responses = openmeteo.weather_api(url, params=params)

# Extract data for each city
data_frames = []
city_names = ["Berlin", "Amsterdam", "Brussels"]
# Ensure we do not go out of index
num_cities = min(len(responses), len(city_names))

for i in range(num_cities):
    response = responses[i]
    daily = response.Daily()
    daily_data = {
        "date": pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left"
        ),
        "temperature_2m_mean": daily.Variables(0).ValuesAsNumpy(),
        "temperature_2m_max": daily.Variables(1).ValuesAsNumpy(),
        "temperature_2m_min": daily.Variables(2).ValuesAsNumpy(),
    }
    df = pd.DataFrame(data=daily_data)
    df['city'] = city_names[i]  # Add city name
    data_frames.append(df)

# Combine data for all cities
combined_dataframe = pd.concat(data_frames)

# Streamlit app
st.title("Climate Data Visualization")

# Define colors for each city
color_map = {
    "Berlin": "purple",
    "Amsterdam": "red",
    "Brussels": "blue"
}

# Plotting the temperature data for all cities
fig = px.line(combined_dataframe, x='date', y='temperature_2m_mean',
              color='city',
              color_discrete_map=color_map,
              title='Average Daily Temperature from 1950 to 2050 for Berlin, Amsterdam, and Brussels')

# Customize the layout
fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Average Temperature (°C)',
    legend_title='City',
)

# Display the plot in the Streamlit app
st.plotly_chart(fig)
