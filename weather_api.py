import requests
import pandas as pd


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"
CURRENT_URL = "https://api.open-meteo.com/v1/forecast"


def get_coordinates(city):
    """Get latitude and longitude for a city."""

    params = {
        "name": city,
        "count": 1,
        "language": "en",
        "format": "json"
    }

    response = requests.get(GEOCODING_URL, params=params, timeout=20)
    response.raise_for_status()

    data = response.json()

    if "results" not in data or not data["results"]:
        raise ValueError(f"City not found: {city}")

    result = data["results"][0]

    return result["latitude"], result["longitude"], result.get("name", city)


def get_historical_weather(city, start_date, end_date):
    """Download historical daily weather data."""

    latitude, longitude, city_name = get_coordinates(city)

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join([
            "temperature_2m_mean",
            "relative_humidity_2m_mean",
            "precipitation_sum",
            "wind_speed_10m_max",
            "surface_pressure_mean"
        ]),
        "timezone": "auto"
    }

    response = requests.get(HISTORICAL_URL, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()

    if "daily" not in data:
        raise ValueError("Historical weather data was not returned.")

    daily = data["daily"]

    df = pd.DataFrame({
        "date": daily["time"],
        "temperature": daily["temperature_2m_mean"],
        "humidity": daily["relative_humidity_2m_mean"],
        "rainfall": daily["precipitation_sum"],
        "wind_speed": daily["wind_speed_10m_max"],
        "pressure": daily["surface_pressure_mean"]
    })

    df["date"] = pd.to_datetime(df["date"])

    print(f"Downloaded {len(df)} days of weather data for {city_name}.")

    return df


def get_current_weather(city):
    """Get current weather conditions."""

    latitude, longitude, city_name = get_coordinates(city)

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
            "surface_pressure"
        ]),
        "timezone": "auto"
    }

    response = requests.get(CURRENT_URL, params=params, timeout=20)
    response.raise_for_status()

    data = response.json()

    current = data["current"]

    return {
        "city": city_name,
        "time": current["time"],
        "temperature": current["temperature_2m"],
        "humidity": current["relative_humidity_2m"],
        "rainfall": current["precipitation"],
        "wind_speed": current["wind_speed_10m"],
        "pressure": current["surface_pressure"]
    }




   