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
# Custom CSS - clean, white, minimal
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .stApp {
        background: #ffffff;
    }
    
    /* Header */
    .main-header {
        background: #0a1f3f;
        padding: 2.5rem 2rem;
        border-radius: 14px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.3px;
    }
    .main-header p {
        color: rgba(255,255,255,0.65);
        font-size: 0.95rem;
        margin-top: 0.4rem;
        font-weight: 400;
    }
    
    /* Cards */
    .card {
        border: 1px solid #e8ecf1;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.2rem 0;
        background: #ffffff;
    }
    
    .card-header {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 1rem;
        padding-bottom: 0.6rem;
        border-bottom: 1px solid #f0f2f5;
    }
    
    .weather-header { color: #2563eb; }
    .prediction-header { color: #0a1f3f; }
    .places-header { color: #7c3aed; }
    .routes-header { color: #b45309; }
    
    /* Weather metrics */
    .weather-row {
        display: flex;
        justify-content: space-between;
        margin: 1rem 0 0.5rem 0;
    }
    .weather-item { text-align: center; flex: 1; }
    .weather-item .val {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1a1a2e;
    }
    .weather-item .lbl {
        font-size: 0.7rem;
        color: #8896ab;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 0.2rem;
    }
    .weather-desc {
        text-align: center;
        color: #8896ab;
        font-size: 0.85rem;
        margin-top: 0.6rem;
    }
    
    /* Prediction display */
    .pred-number {
        font-size: 3.2rem;
        font-weight: 700;
        text-align: center;
        line-height: 1.1;
        margin: 0.5rem 0 0.2rem 0;
    }
    .pred-unit {
        text-align: center;
        font-size: 0.85rem;
        color: #8896ab;
        font-weight: 500;
    }
    .status-pill {
        display: inline-block;
        padding: 0.35rem 0.9rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.8rem;
        margin-top: 0.6rem;
    }
    .pill-green { background: #ecfdf5; color: #059669; }
    .pill-yellow { background: #fefce8; color: #a16207; }
    .pill-orange { background: #fff7ed; color: #c2410c; }
    .pill-red { background: #fef2f2; color: #dc2626; }
    
    .factors-bar {
        background: #f0f4ff;
        border-radius: 8px;
        padding: 0.7rem 1rem;
        color: #3b5998;
        font-size: 0.85rem;
        margin-top: 1rem;
        font-weight: 500;
    }
    
    /* Place items */
    .place-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.7rem 0;
        border-bottom: 1px solid #f5f5f7;
    }
    .place-row:last-child { border-bottom: none; }
    .place-name { font-weight: 600; color: #1a1a2e; font-size: 0.95rem; }
    .place-type { color: #8896ab; font-size: 0.8rem; margin-top: 0.1rem; }
    .place-dist {
        color: #7c3aed;
        font-weight: 700;
        font-size: 0.85rem;
        white-space: nowrap;
    }
    
    /* Route items */
    .route-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.8rem 1rem;
        background: #fafafa;
        border-radius: 8px;
        margin: 0.4rem 0;
    }
    .route-mode { font-weight: 600; color: #1a1a2e; font-size: 0.95rem; }
    .route-info { color: #b45309; font-weight: 600; font-size: 0.9rem; }
    
    /* Subtitle text */
    .card-subtitle {
        color: #8896ab;
        font-size: 0.85rem;
        margin-bottom: 0.8rem;
    }
    
    /* Footer */
    .app-footer {
        text-align: center;
        color: #b0b8c4;
        font-size: 0.72rem;
        margin-top: 2.5rem;
        padding: 1.2rem 0;
        border-top: 1px solid #f0f2f5;
        letter-spacing: 0.02em;
    }
    
    /* Button override */
    .stButton > button[kind="primary"] {
        background: #0a1f3f !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.65rem 2rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        width: 100% !important;
        transition: all 0.2s ease !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #132d54 !important;
        box-shadow: 0 4px 12px rgba(10,31,63,0.2) !important;
    }
    
    /* Input labels */
    .stSelectbox label, .stTimeInput label, .stDateInput label {
        color: #3d4f6f !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
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
            "temp": temp, "humidity": humidity, "windspeed": windspeed,
            "cloudcover": cloudcover, "precip": precip, "preciptype": preciptype,
            "conditions": conditions, "description": weather_desc
        }

    except Exception as e:
        st.error(f"Failed to fetch weather: {e}")
        return None


def map_conditions(weather_main, weather_desc, precip, preciptype):
    desc = weather_desc.lower()
    main = weather_main.lower()

    if preciptype == "Snow":
        if precip >= 4: return "Snow fall: Heavy"
        elif precip >= 1: return "Snow fall: Moderate"
        else: return "Snow fall: Slight"
    elif "drizzle" in main or "drizzle" in desc:
        if precip >= 2: return "Drizzle: Dense"
        elif precip >= 0.5: return "Drizzle: Moderate"
        else: return "Drizzle: Light"
    elif preciptype == "Rain":
        if precip >= 2: return "Rain: Moderate"
        else: return "Rain: Slight"
    elif "cloud" in desc or "overcast" in desc:
        if "few" in desc or "scattered" in desc: return "Partly cloudy"
        elif "broken" in desc: return "Mainly clear"
        else: return "Overcast"
    elif "clear" in desc: return "Clear sky"
    elif "mist" in desc or "fog" in desc or "haze" in desc: return "Overcast"
    else: return "Partly cloudy"


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
        if response.status_code != 200: return []

        data = response.json()
        places = []
        for element in data.get("elements", []):
            tags = element.get("tags", {})
            name = tags.get("name")
            if not name: continue

            place_lat = element.get("lat", lat)
            place_lon = element.get("lon", lon)
            dist = haversine(lat, lon, place_lat, place_lon)
            place_type = tags.get("amenity", tags.get("shop", "shop"))

            places.append({"name": name, "type": place_type.title(), "distance_m": round(dist)})

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
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))


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
            alternatives.append({"mode": "Walking", "duration": f"{walk_mins} mins", "distance": f"{walk_km} km"})
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
            alternatives.append({"mode": "Taxi / Drive", "duration": f"{drive_mins} mins", "distance": f"{drive_km} km"})
    except Exception:
        pass

    return alternatives


# ============================================================
# Build features
# ============================================================
def build_features(direction_id, stop_sequence, day_of_week, is_weekend,
                    hour, is_rush_hour, city, weather):
    features = {
        "direction_id": direction_id, "stop_sequence": stop_sequence,
        "day_of_week": day_of_week, "is_weekend": is_weekend, "hour": hour,
        "temp": weather["temp"], "precip": weather["precip"],
        "windspeed": weather["windspeed"], "humidity": weather["humidity"],
        "cloudcover": weather["cloudcover"], "is_rush_hour": is_rush_hour,
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
    <h1>Scotland Bus Delay Predictor</h1>
    <p>Real-time delay predictions for Edinburgh, Glasgow and Paisley</p>
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
        st.markdown(
            '<div class="card">'
            '<div class="card-header weather-header">Current Weather</div>'
            '<div class="weather-row">'
            '<div class="weather-item">'
            f'<div class="val">{weather["temp"]:.1f}\u00B0C</div>'
            '<div class="lbl">Temperature</div>'
            '</div>'
            '<div class="weather-item">'
            f'<div class="val">{weather["windspeed"]:.1f}</div>'
            '<div class="lbl">Wind (km/h)</div>'
            '</div>'
            '<div class="weather-item">'
            f'<div class="val">{weather["humidity"]}%</div>'
            '<div class="lbl">Humidity</div>'
            '</div>'
            '<div class="weather-item">'
            f'<div class="val">{weather["cloudcover"]}%</div>'
            '<div class="lbl">Cloud Cover</div>'
            '</div>'
            '</div>'
            f'<div class="weather-desc">{weather["description"].title()}</div>'
            '</div>',
            unsafe_allow_html=True
        )

        features = build_features(selected_direction_id, selected_stop_sequence, selected_dow,
                                 selected_is_weekend, selected_hour, selected_is_rush_hour, city, weather)

        prediction = model.predict(features)[0]
        prediction = round(max(prediction, -3), 1)

        if prediction < 0:
            status_text, pill_class, pred_color = "Ahead of Schedule", "pill-green", "#059669"
        elif prediction <= 3:
            status_text, pill_class, pred_color = "On Time", "pill-green", "#059669"
        elif prediction <= 5:
            status_text, pill_class, pred_color = "Slightly Delayed", "pill-yellow", "#a16207"
        elif prediction <= 10:
            status_text, pill_class, pred_color = "Moderately Delayed", "pill-orange", "#c2410c"
        else:
            status_text, pill_class, pred_color = "Heavily Delayed", "pill-red", "#dc2626"

        # Contributing factors
        factors = []
        if selected_is_rush_hour: factors.append("rush hour traffic")
        if weather["precip"] > 0: factors.append(f"{weather['conditions'].lower()}")
        if weather["temp"] <= 0: factors.append("freezing conditions")
        if weather["windspeed"] > 40: factors.append("high winds")

        factors_html = ""
        if factors:
            factors_html = f'<div class="factors-bar">Contributing factors: {", ".join(factors)}</div>'

        # Prediction card
        st.markdown(
            '<div class="card">'
            '<div class="card-header prediction-header">Prediction</div>'
            '<div style="text-align: center;">'
            f'<div class="pred-number" style="color: {pred_color};">{prediction:.1f}</div>'
            '<div class="pred-unit">minutes delay</div>'
            f'<span class="status-pill {pill_class}">{status_text}</span>'
            '</div>'
            f'{factors_html}'
            '</div>',
            unsafe_allow_html=True
        )

        # Nearby Places
        if prediction > 0:
            coords = CITY_COORDS.get(city, CITY_COORDS["Edinburgh"])
            with st.spinner("Searching for nearby places..."):
                places = fetch_nearby_places(coords["lat"], coords["lon"])

            if places:
                items = []
                for pl in places:
                    pname = pl["name"]
                    ptype = pl["type"]
                    pdist = pl["distance_m"]
                    items.append(
                        '<div class="place-row">'
                        "<div>"
                        f'<div class="place-name">{pname}</div>'
                        f'<div class="place-type">{ptype}</div>'
                        "</div>"
                        f'<div class="place-dist">{pdist}m</div>'
                        "</div>"
                    )
                places_block = "".join(items)
                delay_text = f"{prediction:.1f}"
                st.markdown(
                    '<div class="card">'
                    '<div class="card-header places-header">Nearby Places to Wait</div>'
                    f'<div class="card-subtitle">Your bus is {delay_text} minutes late. Here are some places nearby.</div>'
                    f'{places_block}'
                    '</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="card">'
                    '<div class="card-header places-header">Nearby Places to Wait</div>'
                    '<div class="card-subtitle">No nearby places found within 500m.</div>'
                    '</div>',
                    unsafe_allow_html=True
                )

        # Alternative Routes
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
                    items = []
                    for alt in alternatives:
                        amode = alt["mode"]
                        adur = alt["duration"]
                        adist = alt["distance"]
                        items.append(
                            '<div class="route-row">'
                            f'<div class="route-mode">{amode}</div>'
                            f'<div class="route-info">{adur} ({adist})</div>'
                            "</div>"
                        )
                    routes_block = "".join(items)
                    st.markdown(
                        '<div class="card">'
                        '<div class="card-header routes-header">Alternative Ways to Get There</div>'
                        '<div class="card-subtitle">It might be faster to take a different route.</div>'
                        f'{routes_block}'
                        '</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div class="card">'
                        '<div class="card-header routes-header">Alternative Ways to Get There</div>'
                        '<div class="card-subtitle">Could not find alternative routes at this time.</div>'
                        '</div>',
                        unsafe_allow_html=True
                    )

# Footer
st.markdown(
    '<div class="app-footer">'
    'Powered by XGBoost &middot; Weather: OpenWeatherMap &middot; Places: OpenStreetMap &middot; Routes: OpenRouteService &middot; Bus data: BODS'
    '</div>',
    unsafe_allow_html=True
)