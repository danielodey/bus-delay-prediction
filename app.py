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
# Custom CSS for modern styling
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 { color: white; font-size: 2rem; font-weight: 700; margin: 0; }
    .main-header p { color: rgba(255,255,255,0.85); font-size: 1rem; margin-top: 0.5rem; }
    
    .weather-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #1a2940 100%);
        border: 1px solid rgba(100,150,255,0.15);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .prediction-card {
        background: linear-gradient(135deg, #1a3a2a 0%, #1a2e1a 100%);
        border: 1px solid rgba(100,255,150,0.15);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
    }
    .places-card {
        background: linear-gradient(135deg, #3a1a3a 0%, #2e1a2e 100%);
        border: 1px solid rgba(200,100,255,0.15);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .routes-card {
        background: linear-gradient(135deg, #3a2a1a 0%, #2e2a1a 100%);
        border: 1px solid rgba(255,180,100,0.15);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .section-title {
        color: #a0b4ff;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .big-metric {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        line-height: 1.2;
    }
    .metric-label {
        font-size: 0.9rem;
        color: rgba(255,255,255,0.6);
        text-align: center;
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    .status-green { background: rgba(74,222,128,0.15); color: #4ade80; border: 1px solid rgba(74,222,128,0.3); }
    .status-yellow { background: rgba(250,204,21,0.15); color: #facc15; border: 1px solid rgba(250,204,21,0.3); }
    .status-orange { background: rgba(251,146,60,0.15); color: #fb923c; border: 1px solid rgba(251,146,60,0.3); }
    .status-red { background: rgba(248,113,113,0.15); color: #f87171; border: 1px solid rgba(248,113,113,0.3); }
    
    .weather-metrics {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }
    .weather-metric { text-align: center; }
    .weather-metric .value { font-size: 1.5rem; font-weight: 700; color: #60a5fa; }
    .weather-metric .label { font-size: 0.75rem; color: rgba(255,255,255,0.5); text-transform: uppercase; letter-spacing: 0.05em; }
    
    .place-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.6rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .place-item:last-child { border-bottom: none; }
    .place-name { color: #e2e8f0; font-weight: 500; }
    .place-type { color: rgba(255,255,255,0.4); font-size: 0.85rem; }
    .place-dist { color: #c084fc; font-weight: 600; font-size: 0.9rem; }
    
    .route-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.8rem 1rem;
        background: rgba(255,255,255,0.03);
        border-radius: 8px;
        margin: 0.4rem 0;
    }
    .route-mode { color: #e2e8f0; font-weight: 600; }
    .route-detail { color: #fbbf24; font-weight: 500; }
    
    .factors-box {
        background: rgba(59,130,246,0.1);
        border: 1px solid rgba(59,130,246,0.2);
        border-radius: 8px;
        padding: 0.8rem 1rem;
        color: #93c5fd;
        font-size: 0.9rem;
        margin-top: 0.8rem;
    }
    
    .app-footer {
        text-align: center;
        color: rgba(255,255,255,0.25);
        font-size: 0.75rem;
        margin-top: 2rem;
        padding: 1rem 0;
        border-top: 1px solid rgba(255,255,255,0.05);
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102,126,234,0.4) !important;
    }
</style>
""", unsafe_allow_html=True)

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
# City coordinates
# ============================================================
CITY_COORDS = {
    "Edinburgh": {"lat": 55.9533, "lon": -3.1883},
    "Glasgow": {"lat": 55.8642, "lon": -4.2518},
    "Paisley": {"lat": 55.8456, "lon": -4.4239},
}

# ============================================================
# OpenWeatherMap API
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
# Overpass API
# ============================================================
def fetch_nearby_places(lat, lon, radius=500):
    query = f"""
    [out:json][timeout:10];
    (
      node["amenity"="cafe"](around:{radius},{lat},{lon});
      node["amenity"="restaurant"](around:{radius},{lat},{lon});
      node["shop"](around:{radius},{lat},{lon});
    );
    out body 10;
    """
    url = "https://overpass-api.de/api/interpreter"

    try:
        response = requests.post(url, data={"data": query}, timeout=15)
        if response.status_code != 200:
            return []

        data = response.json()
        places = []
        for element in data.get("elements", []):
            tags = element.get("tags", {})
            name = tags.get("name")
            if not name:
                continue

            place_lat = element.get("lat", lat)
            place_lon = element.get("lon", lon)

            dist = haversine(lat, lon, place_lat, place_lon)
            place_type = tags.get("amenity", tags.get("shop", "shop"))

            places.append({
                "name": name,
                "type": place_type.title(),
                "distance_m": round(dist),
            })

        places.sort(key=lambda x: x["distance_m"])
        return places[:5]

    except Exception:
        return []


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# ============================================================
# OpenRouteService API
# ============================================================
def fetch_alternative_routes(start_lat, start_lon, end_lat, end_lon):
    ORS_KEY = st.secrets["ORS_API_KEY"]
    alternatives = []

    try:
        url = "https://api.openrouteservice.org/v2/directions/foot-walking"
        headers = {"Authorization": ORS_KEY, "Content-Type": "application/json"}
        body = {"coordinates": [[start_lon, start_lat], [end_lon, end_lat]], "units": "km"}
        response = requests.post(url, json=body, headers=headers, timeout=15)
        if response.status_code == 200:
            data = response.json()
            route = data["routes"][0]["summary"]
            walk_mins = round(route["duration"] / 60)
            walk_km = round(route["distance"], 1)
            alternatives.append({
                "mode": "🚶 Walking",
                "duration": f"{walk_mins} mins",
                "distance": f"{walk_km} km",
            })
    except Exception:
        pass

    try:
        url = "https://api.openrouteservice.org/v2/directions/driving-car"
        headers = {"Authorization": ORS_KEY, "Content-Type": "application/json"}
        body = {"coordinates": [[start_lon, start_lat], [end_lon, end_lat]], "units": "km"}
        response = requests.post(url, json=body, headers=headers, timeout=15)
        if response.status_code == 200:
            data = response.json()
            route = data["routes"][0]["summary"]
            drive_mins = round(route["duration"] / 60)
            drive_km = round(route["distance"], 1)
            alternatives.append({
                "mode": "🚗 Taxi / Drive",
                "duration": f"{drive_mins} mins",
                "distance": f"{drive_km} km",
            })
    except Exception:
        pass

    return alternatives


# ============================================================
# Build features
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
# App Interface
# ============================================================

# Header
st.markdown("""
<div class="main-header">
    <h1>🚌 Scotland Bus Delay Predictor</h1>
    <p>Real-time delay predictions for Edinburgh, Glasgow & Paisley</p>
</div>
""", unsafe_allow_html=True)

# Selection inputs
city = st.selectbox("Select city", ["Edinburgh", "Glasgow", "Paisley"])

city_routes = route_names[route_names["city"] == city].sort_values("display_name")
route_options = dict(zip(city_routes["display_name"], city_routes["route_id"]))
selected_route_name = st.selectbox("Select route", list(route_options.keys()))
selected_route_id = route_options[selected_route_name]

route_headsigns = headsigns[headsigns["route_id"] == selected_route_id]
direction_options = {f"Towards {row['headsign']}": row["direction_id"] for _, row in route_headsigns.iterrows()}
selected_direction_name = st.selectbox("Select direction", list(direction_options.keys()))
selected_direction_id = direction_options[selected_direction_name]

route_stops = stop_lookup[
    (stop_lookup["route_id"] == selected_route_id) &
    (stop_lookup["direction_id"] == selected_direction_id)
].sort_values("stop_sequence")

stop_options = {f"{row['stop_id']} (Stop {row['stop_sequence']})": [row['stop_id'], row['stop_sequence']] for _, row in route_stops.iterrows()}
selected_stop_name = st.selectbox("Select stop", list(stop_options.keys()))
selected_stop_id, selected_stop_sequence = stop_options[selected_stop_name]

col1, col2 = st.columns(2)
with col1:
    selected_date = date.today()
    st.date_input("Date", value=selected_date, disabled=True)
with col2:
    selected_time = st.time_input("Select time", value=time(12,0))

selected_hour = selected_time.hour
selected_dow = selected_date.weekday()
selected_is_weekend = 1 if selected_dow >= 5 else 0
selected_is_rush_hour = 1 if selected_hour in [7, 8, 9, 16, 17, 18] else 0

# Predict button
if st.button("Predict Delay", type="primary"):
    with st.spinner("Fetching live weather data..."):
        weather = fetch_weather(city)

    if weather:
        # Weather card
        st.markdown(f"""
        <div class="weather-card">
            <div class="section-title">🌤️ Current Weather in {city}</div>
            <div class="weather-metrics">
                <div class="weather-metric">
                    <div class="value">{weather['temp']:.1f}°C</div>
                    <div class="label">Temperature</div>
                </div>
                <div class="weather-metric">
                    <div class="value">{weather['windspeed']:.1f}</div>
                    <div class="label">Wind (km/h)</div>
                </div>
                <div class="weather-metric">
                    <div class="value">{weather['humidity']}%</div>
                    <div class="label">Humidity</div>
                </div>
                <div class="weather-metric">
                    <div class="value">{weather['cloudcover']}%</div>
                    <div class="label">Cloud Cover</div>
                </div>
            </div>
            <div style="text-align:center; color: rgba(255,255,255,0.5); font-size: 0.85rem;">
                {weather['description'].title()}
            </div>
        </div>
        """, unsafe_allow_html=True)

        features = build_features(selected_direction_id, selected_stop_sequence, selected_dow,
                                 selected_is_weekend, selected_hour, selected_is_rush_hour, city, weather)

        prediction = model.predict(features)[0]
        prediction = round(max(prediction, -3), 1)

        if prediction < 0:
            status_text, status_class, metric_color = "Ahead of Schedule", "status-green", "#4ade80"
        elif prediction <= 3:
            status_text, status_class, metric_color = "On Time", "status-green", "#4ade80"
        elif prediction <= 5:
            status_text, status_class, metric_color = "Slightly Delayed", "status-yellow", "#facc15"
        elif prediction <= 10:
            status_text, status_class, metric_color = "Moderately Delayed", "status-orange", "#fb923c"
        else:
            status_text, status_class, metric_color = "Heavily Delayed", "status-red", "#f87171"

        # Contributing factors
        factors = []
        if selected_is_rush_hour: factors.append("rush hour traffic")
        if weather["precip"] > 0: factors.append(f"{weather['conditions'].lower()}")
        if weather["temp"] <= 0: factors.append("freezing conditions")
        if weather["windspeed"] > 40: factors.append("high winds")

        factors_html = ""
        if factors:
            factors_html = f'<div class="factors-box">📋 Contributing factors: {", ".join(factors)}</div>'

        # Prediction card
        st.markdown(f"""
        <div class="prediction-card">
            <div class="section-title" style="justify-content: center;">📊 Prediction</div>
            <div class="big-metric" style="color: {metric_color};">{prediction:.1f} min</div>
            <div class="metric-label">Estimated Delay</div>
            <div style="margin-top: 0.8rem;">
                <span class="status-badge {status_class}">{status_text}</span>
            </div>
            {factors_html}
        </div>
        """, unsafe_allow_html=True)

        # Nearby Places - same threshold as working code
        if prediction > 0:
            coords = CITY_COORDS.get(city, CITY_COORDS["Edinburgh"])
            with st.spinner("Searching for nearby places..."):
                places = fetch_nearby_places(coords["lat"], coords["lon"])

            if places:
                places_html = ""
                for place in places:
                    places_html += f"""
                    <div class="place-item">
                        <div>
                            <div class="place-name">{place['name']}</div>
                            <div class="place-type">{place['type']}</div>
                        </div>
                        <div class="place-dist">{place['distance_m']}m</div>
                    </div>
                    """
                st.markdown(f"""
                <div class="places-card">
                    <div class="section-title">☕ Nearby Places to Wait</div>
                    <div style="color: rgba(255,255,255,0.5); font-size: 0.85rem; margin-bottom: 0.8rem;">
                        Your bus is {prediction:.1f} minutes late. Here are some places nearby:
                    </div>
                    {places_html}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="places-card">
                    <div class="section-title">☕ Nearby Places to Wait</div>
                    <div style="color: rgba(255,255,255,0.4); font-size: 0.85rem;">
                        No nearby places found within 500m.
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Alternative Routes - same threshold as working code
        if prediction > 3:
            coords = CITY_COORDS.get(city, CITY_COORDS["Edinburgh"])
            last_stop = route_stops.iloc[-1] if len(route_stops) > 0 else None

            if last_stop is not None:
                dest_coords = CITY_COORDS.get(city, CITY_COORDS["Edinburgh"])
                dest_lat, dest_lon = dest_coords["lat"] + 0.02, dest_coords["lon"] + 0.02

                with st.spinner("Checking alternative routes..."):
                    alternatives = fetch_alternative_routes(
                        coords["lat"], coords["lon"], dest_lat, dest_lon
                    )

                if alternatives:
                    routes_html = ""
                    for alt in alternatives:
                        routes_html += f"""
                        <div class="route-item">
                            <div class="route-mode">{alt['mode']}</div>
                            <div class="route-detail">{alt['duration']} ({alt['distance']})</div>
                        </div>
                        """
                    st.markdown(f"""
                    <div class="routes-card">
                        <div class="section-title">🗺️ Alternative Ways to Get There</div>
                        <div style="color: rgba(255,255,255,0.5); font-size: 0.85rem; margin-bottom: 0.8rem;">
                            It might be faster to take a different route:
                        </div>
                        {routes_html}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="routes-card">
                        <div class="section-title">🗺️ Alternative Ways to Get There</div>
                        <div style="color: rgba(255,255,255,0.4); font-size: 0.85rem;">
                            Could not find alternative routes at this time.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="app-footer">
    Powered by XGBoost · Weather: OpenWeatherMap · Places: OpenStreetMap · Routes: OpenRouteService · Bus data: BODS
</div>
""", unsafe_allow_html=True)