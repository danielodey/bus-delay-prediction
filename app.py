import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime, date, time

# ============================================================
# Page configuration
# ============================================================
st.set_page_config(
    page_title="Scotland Bus Delay Predictor",
    page_icon="🚌",
    layout="centered"
)

# ============================================================
# Load model and lookup data
# ============================================================
@st.cache_resource
def load_model():
    return joblib.load("xgboost_model.pkl")

@st.cache_data
def load_lookups():
    route_names = pd.read_parquet("app_route_names.parquet")
    headsigns = pd.read_parquet("app_headsigns.parquet")
    stop_lookup = pd.read_parquet("app_stop_lookup.parquet")
    feature_cols = joblib.load("feature_columns.pkl")
    return route_names, headsigns, stop_lookup, feature_cols

model = load_model()
route_names, headsigns, stop_lookup, feature_cols = load_lookups()

# ============================================================
# OpenWeatherMap API - fetch live weather for selected city
# ============================================================
def fetch_weather(city):
    API_KEY = st.secrets["OPENWEATHER_API_KEY"]
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},GB&appid={API_KEY}&units=metric"

    try:
        response = requests.get(url, timeout=10)
        data = response.json()

        if response.status_code != 200:
            st.error(f"Weather API error: {data.get('message', 'Unknown error')}")
            return None

        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        windspeed = data["wind"]["speed"] * 3.6
        cloudcover = data["clouds"]["all"]
        rain_mm = data.get("rain", {}).get("1h", 0)
        snow_mm = data.get("snow", {}).get("1h", 0)
        precip = rain_mm + snow_mm
        weather_main = data["weather"][0]["main"]
        weather_desc = data["weather"][0]["description"]

        if snow_mm > 0:
            preciptype = "Snow"
        elif rain_mm > 0:
            preciptype = "Rain"
        else:
            preciptype = "None"

        conditions = map_conditions(weather_main, weather_desc, precip, preciptype)

        return {
            "temp": temp,
            "humidity": humidity,
            "windspeed": windspeed,
            "cloudcover": cloudcover,
            "precip": precip,
            "preciptype": preciptype,
            "conditions": conditions,
            "description": weather_desc
        }

    except Exception as e:
        st.error(f"Failed to fetch weather: {e}")
        return None


def map_conditions(weather_main, weather_desc, precip, preciptype):
    desc = weather_desc.lower()
    main = weather_main.lower()

    if preciptype == "Snow":
        if precip >= 4:
            return "Snow fall: Heavy"
        elif precip >= 1:
            return "Snow fall: Moderate"
        else:
            return "Snow fall: Slight"
    elif "drizzle" in main or "drizzle" in desc:
        if precip >= 2:
            return "Drizzle: Dense"
        elif precip >= 0.5:
            return "Drizzle: Moderate"
        else:
            return "Drizzle: Light"
    elif preciptype == "Rain":
        if precip >= 2:
            return "Rain: Moderate"
        else:
            return "Rain: Slight"
    elif "cloud" in desc or "overcast" in desc:
        if "few" in desc or "scattered" in desc:
            return "Partly cloudy"
        elif "broken" in desc:
            return "Mainly clear"
        else:
            return "Overcast"
    elif "clear" in desc:
        return "Clear sky"
    elif "mist" in desc or "fog" in desc or "haze" in desc:
        return "Overcast"
    else:
        return "Partly cloudy"


# ============================================================
# Build prediction input matching the 26 model features
# ============================================================
def build_features(direction_id, stop_sequence, day_of_week, is_weekend,
                   hour, is_rush_hour, city, weather):
    features = {
        "direction_id": direction_id,
        "stop_sequence": stop_sequence,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "hour": hour,
        "temp": weather["temp"],
        "precip": weather["precip"],
        "windspeed": weather["windspeed"],
        "humidity": weather["humidity"],
        "cloudcover": weather["cloudcover"],
        "is_rush_hour": is_rush_hour,
    }

    features["city_Glasgow"] = 1 if city == "Glasgow" else 0
    features["city_Paisley"] = 1 if city == "Paisley" else 0

    condition_cols = [
        "conditions_Drizzle: Dense", "conditions_Drizzle: Light",
        "conditions_Drizzle: Moderate", "conditions_Mainly clear",
        "conditions_Overcast", "conditions_Partly cloudy",
        "conditions_Rain: Moderate", "conditions_Rain: Slight",
        "conditions_Snow fall: Heavy", "conditions_Snow fall: Moderate",
        "conditions_Snow fall: Slight"
    ]
    for col in condition_cols:
        condition_name = col.replace("conditions_", "")
        features[col] = 1 if weather["conditions"] == condition_name else 0

    features["preciptype_Rain"] = 1 if weather["preciptype"] == "Rain" else 0
    features["preciptype_Snow"] = 1 if weather["preciptype"] == "Snow" else 0

    df = pd.DataFrame([features])
    df = df[feature_cols]
    return df


# ============================================================
# App interface
# ============================================================
st.title("🚌 Scotland Bus Delay Predictor")
st.markdown("Predict how many minutes your bus is likely to be delayed in Edinburgh, Glasgow, or Paisley.")

# --- City selection ---
city = st.selectbox("Select city", ["Edinburgh", "Glasgow", "Paisley"])

# --- Route selection (filtered by city) ---
city_routes = route_names[route_names["city"] == city].sort_values("display_name")
route_options = dict(zip(city_routes["display_name"], city_routes["route_id"]))
selected_route_name = st.selectbox("Select route", list(route_options.keys()))
selected_route_id = route_options[selected_route_name]

# --- Direction selection (filtered by route) ---
route_headsigns = headsigns[headsigns["route_id"] == selected_route_id]
direction_options = {}
for _, row in route_headsigns.iterrows():
    direction_options[f"Towards {row['headsign']}"] = row["direction_id"]
selected_direction_name = st.selectbox("Select direction", list(direction_options.keys()))
selected_direction_id = direction_options[selected_direction_name]

# --- Stop selection (filtered by route + direction) ---
route_stops = stop_lookup[
    (stop_lookup["route_id"] == selected_route_id) &
    (stop_lookup["direction_id"] == selected_direction_id)
].sort_values("stop_sequence")

stop_options = dict(zip(
    route_stops["stop_id"] + " (Stop " + route_stops["stop_sequence"].astype(str) + ")",
    route_stops[["stop_id", "stop_sequence"]].values.tolist()
))
selected_stop_name = st.selectbox("Select stop", list(stop_options.keys()))
selected_stop_id, selected_stop_sequence = stop_options[selected_stop_name]

# --- Date and time selection ---
col1, col2 = st.columns(2)
with col1:
    selected_date = st.date_input("Select date", value=date.today())
with col2:
    selected_time = st.time_input("Select time", value=datetime.now().time())

selected_hour = selected_time.hour
selected_dow = selected_date.weekday()
selected_is_weekend = 1 if selected_dow >= 5 else 0
selected_is_rush_hour = 1 if selected_hour in [7, 8, 9, 16, 17, 18] else 0

# --- Predict button ---
if st.button("Predict Delay", type="primary"):

    with st.spinner("Fetching live weather data..."):
        weather = fetch_weather(city)

    if weather:
        st.markdown("---")
        st.subheader("Current Weather")
        w_col1, w_col2, w_col3 = st.columns(3)
        w_col1.metric("Temperature", f"{weather['temp']:.1f}°C")
        w_col2.metric("Wind", f"{weather['windspeed']:.1f} km/h")
        w_col3.metric("Humidity", f"{weather['humidity']}%")
        st.markdown(f"**Conditions:** {weather['description'].title()} | **Cloud Cover:** {weather['cloudcover']}%")

        features = build_features(
            direction_id=selected_direction_id,
            stop_sequence=selected_stop_sequence,
            day_of_week=selected_dow,
            is_weekend=selected_is_weekend,
            hour=selected_hour,
            is_rush_hour=selected_is_rush_hour,
            city=city,
            weather=weather
        )

        prediction = model.predict(features)[0]
        prediction = round(max(prediction, -3), 1)

        if prediction < 0:
            status = "🟢 Ahead of Schedule"
        elif prediction <= 3:
            status = "🟢 On Time"
        elif prediction <= 5:
            status = "🟡 Slightly Delayed"
        elif prediction <= 10:
            status = "🟠 Moderately Delayed"
        else:
            status = "🔴 Heavily Delayed"

        st.markdown("---")
        st.subheader("Prediction")
        st.metric("Estimated Delay", f"{prediction:.1f} minutes")
        st.markdown(f"**Status: {status}**")

        factors = []
        if selected_is_rush_hour:
            factors.append("rush hour traffic")
        if weather["precip"] > 0:
            factors.append(f"{weather['conditions'].lower()}")
        if weather["temp"] <= 0:
            factors.append("freezing conditions")
        if weather["windspeed"] > 40:
            factors.append("high winds")

        if factors:
            st.info(f"Contributing factors: {', '.join(factors)}")

st.markdown("---")
st.caption("Powered by XGBoost | Weather data from OpenWeatherMap | Bus data from BODS")

