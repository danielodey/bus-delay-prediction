import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime, date, time

st.set_page_config(page_title="Scotland Bus Delay Predictor", page_icon="🚌", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .stApp { background: #f8f9fb; }
    
    .hero {
        background: #0b2545;
        padding: 2.8rem 2rem 2.4rem 2rem;
        border-radius: 18px;
        margin-bottom: 2rem;
    }
    .hero h1 {
        color: #fff;
        font-size: 1.85rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero p {
        color: #8da9c4;
        font-size: 0.92rem;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    .card {
        background: #ffffff;
        border-radius: 14px;
        padding: 1.6rem;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.02);
    }
    
    .card-label {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 1.1rem;
    }
    .label-weather { color: #2563eb; }
    .label-prediction { color: #0b2545; }
    .label-places { color: #7c3aed; }
    .label-routes { color: #0d9488; }
    
    .w-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.5rem;
        margin-bottom: 0.8rem;
    }
    .w-box { text-align: center; padding: 0.6rem 0; }
    .w-val { font-size: 1.35rem; font-weight: 800; color: #1e293b; }
    .w-lbl { font-size: 0.68rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 0.15rem; }
    .w-cond { text-align: center; color: #94a3b8; font-size: 0.82rem; font-weight: 500; padding-top: 0.4rem; border-top: 1px solid #f1f5f9; }
    
    .pred-wrap { text-align: center; padding: 0.5rem 0; }
    .pred-num { font-size: 3.5rem; font-weight: 800; line-height: 1; }
    .pred-sub { font-size: 0.82rem; color: #94a3b8; font-weight: 600; margin-top: 0.3rem; }
    .pill {
        display: inline-block;
        padding: 0.3rem 0.85rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 0.75rem;
        margin-top: 0.7rem;
        letter-spacing: 0.02em;
    }
    .pill-g { background: #ecfdf5; color: #059669; }
    .pill-y { background: #fefce8; color: #a16207; }
    .pill-o { background: #fff7ed; color: #ea580c; }
    .pill-r { background: #fef2f2; color: #dc2626; }
    
    .factor-strip {
        background: #eff6ff;
        border-radius: 8px;
        padding: 0.65rem 1rem;
        color: #1d4ed8;
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 1rem;
    }
    
    .p-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.75rem 0;
        border-bottom: 1px solid #f1f5f9;
    }
    .p-row:last-child { border-bottom: none; }
    .p-name { font-weight: 700; color: #1e293b; font-size: 0.92rem; }
    .p-type { color: #94a3b8; font-size: 0.78rem; font-weight: 500; }
    .p-dist { color: #7c3aed; font-weight: 800; font-size: 0.85rem; }
    
    .r-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.85rem 1.1rem;
        background: #f8f9fb;
        border-radius: 10px;
        margin: 0.35rem 0;
    }
    .r-mode { font-weight: 700; color: #1e293b; font-size: 0.92rem; }
    .r-info { color: #0d9488; font-weight: 700; font-size: 0.88rem; }
    
    .sub-text { color: #94a3b8; font-size: 0.82rem; font-weight: 500; margin-bottom: 0.9rem; }
    
    .foot {
        text-align: center;
        color: #c1c8d4;
        font-size: 0.7rem;
        margin-top: 2rem;
        padding: 1rem 0;
        font-weight: 500;
    }
    
    .stButton > button[kind="primary"] {
        background: #0b2545 !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.7rem 2rem !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        width: 100% !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        letter-spacing: 0.01em !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #134074 !important;
        box-shadow: 0 6px 20px rgba(11,37,69,0.15) !important;
    }
    
    .stSelectbox label, .stTimeInput label, .stDateInput label {
        color: #475569 !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
    }
</style>
""", unsafe_allow_html=True)

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

CITY_COORDS = {
    "Edinburgh": {"lat": 55.9533, "lon": -3.1883},
    "Glasgow": {"lat": 55.8642, "lon": -4.2518},
    "Paisley": {"lat": 55.8456, "lon": -4.4239},
}

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
        if snow_mm > 0: preciptype = "Snow"
        elif rain_mm > 0: preciptype = "Rain"
        else: preciptype = "None"
        conditions = map_conditions(weather_main, weather_desc, precip, preciptype)
        return {"temp": temp, "humidity": humidity, "windspeed": windspeed, "cloudcover": cloudcover,
                "precip": precip, "preciptype": preciptype, "conditions": conditions, "description": weather_desc}
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

def fetch_nearby_places(lat, lon, radius=500):
    query = f"""
    [out:json][timeout:10];
    (node["amenity"="cafe"](around:{radius},{lat},{lon});
     node["amenity"="restaurant"](around:{radius},{lat},{lon});
     node["shop"](around:{radius},{lat},{lon}););
    out body 10;"""
    try:
        response = requests.post("https://overpass-api.de/api/interpreter", data={"data": query}, timeout=15)
        if response.status_code != 200: return []
        data = response.json()
        places = []
        for el in data.get("elements", []):
            tags = el.get("tags", {})
            name = tags.get("name")
            if not name: continue
            dist = haversine(lat, lon, el.get("lat", lat), el.get("lon", lon))
            ptype = tags.get("amenity", tags.get("shop", "shop"))
            places.append({"name": name, "type": ptype.title(), "distance_m": round(dist)})
        places.sort(key=lambda x: x["distance_m"])
        return places[:5]
    except Exception:
        return []

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

def fetch_alternative_routes(start_lat, start_lon, end_lat, end_lon):
    ORS_KEY = st.secrets["ORS_API_KEY"]
    alternatives = []
    for profile, label in [("foot-walking", "Walking"), ("driving-car", "Taxi / Drive")]:
        try:
            headers = {"Authorization": ORS_KEY, "Content-Type": "application/json"}
            body = {"coordinates": [[start_lon, start_lat], [end_lon, end_lat]], "units": "km"}
            resp = requests.post(f"https://api.openrouteservice.org/v2/directions/{profile}",
                                json=body, headers=headers, timeout=15)
            if resp.status_code == 200:
                s = resp.json()["routes"][0]["summary"]
                alternatives.append({"mode": label, "duration": f"{round(s['duration']/60)} mins", "distance": f"{round(s['distance'],1)} km"})
        except Exception:
            pass
    return alternatives

def build_features(direction_id, stop_sequence, day_of_week, is_weekend, hour, is_rush_hour, city, weather):
    features = {"direction_id": direction_id, "stop_sequence": stop_sequence, "day_of_week": day_of_week,
                "is_weekend": is_weekend, "hour": hour, "temp": weather["temp"], "precip": weather["precip"],
                "windspeed": weather["windspeed"], "humidity": weather["humidity"],
                "cloudcover": weather["cloudcover"], "is_rush_hour": is_rush_hour}
    features["city_Glasgow"] = 1 if city == "Glasgow" else 0
    features["city_Paisley"] = 1 if city == "Paisley" else 0
    for col in ["conditions_Drizzle: Dense","conditions_Drizzle: Light","conditions_Drizzle: Moderate",
                "conditions_Mainly clear","conditions_Overcast","conditions_Partly cloudy",
                "conditions_Rain: Moderate","conditions_Rain: Slight","conditions_Snow fall: Heavy",
                "conditions_Snow fall: Moderate","conditions_Snow fall: Slight"]:
        features[col] = 1 if weather["conditions"] == col.replace("conditions_", "") else 0
    features["preciptype_Rain"] = 1 if weather["preciptype"] == "Rain" else 0
    features["preciptype_Snow"] = 1 if weather["preciptype"] == "Snow" else 0
    df = pd.DataFrame([features])
    df = df[feature_cols]
    return df

# ============================================================
# INTERFACE
# ============================================================

st.markdown('<div class="hero"><h1>Scotland Bus Delay Predictor</h1><p>Real-time delay predictions for Edinburgh, Glasgow and Paisley</p></div>', unsafe_allow_html=True)

city = st.selectbox("Select city", ["Edinburgh", "Glasgow", "Paisley"])
city_routes = route_names[route_names["city"] == city].sort_values("display_name")
route_options = dict(zip(city_routes["display_name"], city_routes["route_id"]))
selected_route_name = st.selectbox("Select route", list(route_options.keys()))
selected_route_id = route_options[selected_route_name]

route_headsigns = headsigns[headsigns["route_id"] == selected_route_id]
direction_options = {f"Towards {row['headsign']}": row["direction_id"] for _, row in route_headsigns.iterrows()}
selected_direction_name = st.selectbox("Select direction", list(direction_options.keys()))
selected_direction_id = direction_options[selected_direction_name]

route_stops = stop_lookup[(stop_lookup["route_id"] == selected_route_id) & (stop_lookup["direction_id"] == selected_direction_id)].sort_values("stop_sequence")
stop_options = {f"{row['stop_id']} (Stop {row['stop_sequence']})": [row['stop_id'], row['stop_sequence']] for _, row in route_stops.iterrows()}
selected_stop_name = st.selectbox("Select stop", list(stop_options.keys()))
selected_stop_id, selected_stop_sequence = stop_options[selected_stop_name]

c1, c2 = st.columns(2)
with c1:
    selected_date = date.today()
    st.date_input("Date", value=selected_date, disabled=True)
with c2:
    selected_time = st.time_input("Select time", value=time(12, 0))

selected_hour = selected_time.hour
selected_dow = selected_date.weekday()
selected_is_weekend = 1 if selected_dow >= 5 else 0
selected_is_rush_hour = 1 if selected_hour in [7, 8, 9, 16, 17, 18] else 0

if st.button("Predict Delay", type="primary"):
    with st.spinner("Fetching live weather data..."):
        weather = fetch_weather(city)

    if weather:
        st.markdown(
            '<div class="card">'
            '<div class="card-label label-weather">Current Weather</div>'
            '<div class="w-grid">'
            f'<div class="w-box"><div class="w-val">{weather["temp"]:.1f}\u00B0</div><div class="w-lbl">Temp</div></div>'
            f'<div class="w-box"><div class="w-val">{weather["windspeed"]:.0f}</div><div class="w-lbl">Wind km/h</div></div>'
            f'<div class="w-box"><div class="w-val">{weather["humidity"]}%</div><div class="w-lbl">Humidity</div></div>'
            f'<div class="w-box"><div class="w-val">{weather["cloudcover"]}%</div><div class="w-lbl">Clouds</div></div>'
            '</div>'
            f'<div class="w-cond">{weather["description"].title()}</div>'
            '</div>',
            unsafe_allow_html=True
        )

        features = build_features(selected_direction_id, selected_stop_sequence, selected_dow,
                                 selected_is_weekend, selected_hour, selected_is_rush_hour, city, weather)
        prediction = model.predict(features)[0]
        prediction = round(max(prediction, -3), 1)

        if prediction < 0: st_txt, pc, clr = "Ahead of Schedule", "pill-g", "#059669"
        elif prediction <= 3: st_txt, pc, clr = "On Time", "pill-g", "#059669"
        elif prediction <= 5: st_txt, pc, clr = "Slightly Delayed", "pill-y", "#a16207"
        elif prediction <= 10: st_txt, pc, clr = "Moderately Delayed", "pill-o", "#ea580c"
        else: st_txt, pc, clr = "Heavily Delayed", "pill-r", "#dc2626"

        factors = []
        if selected_is_rush_hour: factors.append("rush hour traffic")
        if weather["precip"] > 0: factors.append(weather["conditions"].lower())
        if weather["temp"] <= 0: factors.append("freezing conditions")
        if weather["windspeed"] > 40: factors.append("high winds")
        fhtml = f'<div class="factor-strip">Contributing factors: {", ".join(factors)}</div>' if factors else ""

        st.markdown(
            '<div class="card">'
            '<div class="card-label label-prediction">Prediction</div>'
            '<div class="pred-wrap">'
            f'<div class="pred-num" style="color:{clr}">{prediction:.1f}</div>'
            '<div class="pred-sub">minutes delay</div>'
            f'<div><span class="pill {pc}">{st_txt}</span></div>'
            '</div>'
            f'{fhtml}'
            '</div>',
            unsafe_allow_html=True
        )

        # Nearby places
        if prediction > 0:
            coords = CITY_COORDS.get(city, CITY_COORDS["Edinburgh"])
            with st.spinner("Searching for nearby places..."):
                places = fetch_nearby_places(coords["lat"], coords["lon"])
            if places:
                rows = []
                for pl in places:
                    rows.append(
                        '<div class="p-row"><div>'
                        f'<div class="p-name">{pl["name"]}</div>'
                        f'<div class="p-type">{pl["type"]}</div>'
                        f'</div><div class="p-dist">{pl["distance_m"]}m</div></div>'
                    )
                st.markdown(
                    '<div class="card">'
                    '<div class="card-label label-places">Nearby Places to Wait</div>'
                    f'<div class="sub-text">Your bus is {prediction:.1f} minutes late. Here are some places nearby.</div>'
                    + "".join(rows) +
                    '</div>',
                    unsafe_allow_html=True
                )

        # Alternative routes
        if prediction > 3:
            coords = CITY_COORDS.get(city, CITY_COORDS["Edinburgh"])
            last_stop = route_stops.iloc[-1] if len(route_stops) > 0 else None
            if last_stop is not None:
                dc = CITY_COORDS.get(city, CITY_COORDS["Edinburgh"])
                dlat, dlon = dc["lat"] + 0.02, dc["lon"] + 0.02
                with st.spinner("Checking alternative routes..."):
                    alts = fetch_alternative_routes(coords["lat"], coords["lon"], dlat, dlon)
                if alts:
                    rows = []
                    for a in alts:
                        rows.append(
                            '<div class="r-row">'
                            f'<div class="r-mode">{a["mode"]}</div>'
                            f'<div class="r-info">{a["duration"]} ({a["distance"]})</div>'
                            '</div>'
                        )
                    st.markdown(
                        '<div class="card">'
                        '<div class="card-label label-routes">Alternative Ways to Get There</div>'
                        '<div class="sub-text">It might be faster to take a different route.</div>'
                        + "".join(rows) +
                        '</div>',
                        unsafe_allow_html=True
                    )

st.markdown('<div class="foot">Powered by XGBoost &middot; Weather: OpenWeatherMap &middot; Places: OpenStreetMap &middot; Routes: OpenRouteService &middot; Bus data: BODS</div>', unsafe_allow_html=True)