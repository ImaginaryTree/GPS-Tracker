import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import math
from datetime import datetime
import folium
from folium.plugins import AntPath
from streamlit_folium import st_folium

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heavy Equipment GPS Tracker",
    page_icon="🚜",
    layout="wide",
)

# ── Constants / helpers ───────────────────────────────────────────────────────
DATA_STORE = "equipment_data.json"

COURSE_LABELS = {
    (0,   22):  "N",  (22,  67):  "NE", (67,  112): "E",
    (112, 157): "SE", (157, 202): "S",  (202, 247): "SW",
    (247, 292): "W",  (292, 337): "NW", (337, 360): "N",
}

def bearing_to_compass(deg):
    for (lo, hi), label in COURSE_LABELS.items():
        if lo <= deg < hi:
            return label
    return "N"

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def parse_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = [c.strip().lower() for c in df.columns]
    rename = {}
    for c in df.columns:
        if c in ("lat", "latitude"):                rename[c] = "lat"
        elif c in ("lng", "lon", "longitude"):      rename[c] = "lng"
        elif c in ("course", "heading", "bearing"): rename[c] = "course"
        elif "time" in c or "stamp" in c or "date" in c: rename[c] = "tstamp"
    df = df.rename(columns=rename)
    df["tstamp"] = pd.to_datetime(df["tstamp"], dayfirst=False)
    df = df.sort_values("tstamp").reset_index(drop=True)
    return df

def compute_metrics(df: pd.DataFrame) -> dict:
    from collections import Counter
    lats   = df["lat"].tolist()
    lngs   = df["lng"].tolist()
    times  = df["tstamp"].tolist()
    courses = df["course"].tolist()

    distances = [haversine_m(lats[i-1], lngs[i-1], lats[i], lngs[i]) for i in range(1, len(df))]
    total_dist_km = sum(distances) / 1000
    duration_h = (times[-1] - times[0]).total_seconds() / 3600

    speeds = []
    for i, d in enumerate(distances):
        dt = (times[i+1] - times[i]).total_seconds()
        speeds.append((d / dt * 3.6) if dt > 0 else 0)

    avg_speed = np.mean(speeds) if speeds else 0
    max_speed = max(speeds) if speeds else 0
    idle_pct  = sum(1 for s in speeds if s < 0.5) / len(speeds) * 100 if speeds else 0
    bbox_diag_km = haversine_m(min(lats), min(lngs), max(lats), max(lngs)) / 1000
    compass   = [bearing_to_compass(c) for c in courses]
    top_dir   = Counter(compass).most_common(1)[0][0]

    df2 = df.copy()
    df2["hour"] = df2["tstamp"].dt.hour
    hourly_counts = df2.groupby("hour").size().to_dict()

    return {
        "total_distance_km":  round(total_dist_km, 3),
        "bbox_diagonal_km":   round(bbox_diag_km, 3),
        "duration_hours":     round(duration_h, 2),
        "avg_speed_kmh":      round(avg_speed, 2),
        "max_speed_kmh":      round(max_speed, 2),
        "idle_pct":           round(idle_pct, 1),
        "dominant_direction": top_dir,
        "data_points":        len(df),
        "start_time":         str(times[0]),
        "end_time":           str(times[-1]),
        "hourly_activity":    hourly_counts,
        "lat_center":         round(np.mean(lats), 6),
        "lng_center":         round(np.mean(lngs), 6),
    }

def load_store() -> dict:
    if os.path.exists(DATA_STORE):
        with open(DATA_STORE) as f:
            return json.load(f)
    return {}

def save_store(store: dict):
    with open(DATA_STORE, "w") as f:
        json.dump(store, f, indent=2, default=str)

# ── Map builders ──────────────────────────────────────────────────────────────

def build_static_map(df: pd.DataFrame) -> folium.Map:
    lat_c = df["lat"].mean()
    lng_c = df["lng"].mean()
    m = folium.Map(location=[lat_c, lng_c], zoom_start=15, tiles="OpenStreetMap")
    coords = list(zip(df["lat"], df["lng"]))
    n = len(coords)

    # faded ghost line
    folium.PolyLine(coords, color="#aaaaaa", weight=2, opacity=0.4, dash_array="4").add_to(m)

    # animated ant-path
    AntPath(
        coords,
        color="#1a73e8", weight=4, opacity=0.9,
        delay=600, dash_array=[20, 30], pulse_color="#ffffff",
    ).add_to(m)

    folium.Marker(coords[0],  popup="▶ Start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker(coords[-1], popup="⬛ End",   icon=folium.Icon(color="red",   icon="stop")).add_to(m)

    for i in range(0, n, max(1, n // 10)):
        row = df.iloc[i]
        folium.CircleMarker(
            [row["lat"], row["lng"]], radius=4, color="#f57c00", fill=True,
            popup=f"{row['tstamp']}<br>Course: {row['course']}°"
        ).add_to(m)

    folium.LayerControl().add_to(m)
    return m


def build_playback_html(df: pd.DataFrame) -> str:
    coords     = list(zip(df["lat"].tolist(), df["lng"].tolist()))
    timestamps = [str(t) for t in df["tstamp"].tolist()]
    courses    = df["course"].tolist()
    lat_c      = df["lat"].mean()
    lng_c      = df["lng"].mean()

    coords_js  = json.dumps(coords)
    times_js   = json.dumps(timestamps)
    courses_js = json.dumps(courses)

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    * {{ box-sizing:border-box; margin:0; padding:0; }}
    body {{ font-family:sans-serif; background:#0e1117; color:#fafafa; }}
    #map {{ width:100%; height:440px; }}
    #controls {{
      display:flex; align-items:center; gap:10px;
      padding:10px 14px; background:#1e2130;
      border-top:1px solid #333; flex-wrap:wrap;
    }}
    button {{
      padding:6px 16px; border:none; border-radius:6px;
      cursor:pointer; font-size:14px; font-weight:600;
    }}
    #btnPlay  {{ background:#1a73e8; color:#fff; }}
    #btnPause {{ background:#f57c00; color:#fff; }}
    #btnReset {{ background:#555;    color:#fff; }}
    #speedSel {{ padding:5px 8px; border-radius:6px; background:#2d3250; color:#fff; border:1px solid #555; }}
    #slider   {{ flex:1; min-width:140px; accent-color:#1a73e8; }}
    #info     {{ font-size:12px; color:#aaa; min-width:240px; }}
    #prog     {{ font-size:12px; color:#1a73e8; font-weight:bold; white-space:nowrap; }}
  </style>
</head>
<body>
<div id="map"></div>
<div id="controls">
  <button id="btnPlay">&#9654; Play</button>
  <button id="btnPause">&#9646;&#9646; Pause</button>
  <button id="btnReset">&#8635; Reset</button>
  <label style="font-size:12px;color:#aaa">Speed
    <select id="speedSel">
      <option value="200">0.5x</option>
      <option value="120" selected>1x</option>
      <option value="60">2x</option>
      <option value="20">5x</option>
    </select>
  </label>
  <input id="slider" type="range" min="0" max="0" value="0"/>
  <span id="prog">1 / 0</span>
  <span id="info">-</span>
</div>
<script>
  const COORDS   = {coords_js};
  const TIMES    = {times_js};
  const COURSES  = {courses_js};
  const N        = COORDS.length;

  const map = L.map('map').setView([{lat_c}, {lng_c}], 15);
  L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
    {{attribution:'© OpenStreetMap'}}).addTo(map);

  // Ghost track
  L.polyline(COORDS, {{color:'#444', weight:2, opacity:0.5, dashArray:'5'}}).addTo(map);

  // Animated drawn path
  const drawn = L.polyline([], {{color:'#1a73e8', weight:4, opacity:0.95}}).addTo(map);

  // Truck marker
  const truckIcon = L.divIcon({{
    html: '<div style="font-size:26px;line-height:1;transform:translate(-50%,-50%)">&#128665;</div>',
    className:'', iconSize:[0,0]
  }});
  const truck = L.marker(COORDS[0], {{icon:truckIcon}}).addTo(map);

  // Start / end dots
  L.circleMarker(COORDS[0],   {{radius:7, color:'#00c853', fillColor:'#00c853', fillOpacity:1}})
   .bindTooltip('Start').addTo(map);
  L.circleMarker(COORDS[N-1], {{radius:7, color:'#d32f2f', fillColor:'#d32f2f', fillOpacity:1}})
   .bindTooltip('End').addTo(map);

  let step=0, timer=null;
  const slider = document.getElementById('slider');
  const info   = document.getElementById('info');
  const prog   = document.getElementById('prog');
  const spSel  = document.getElementById('speedSel');
  slider.max = N-1;

  function goTo(i) {{
    step = Math.max(0, Math.min(N-1, i));
    slider.value = step;
    truck.setLatLng(COORDS[step]);
    drawn.setLatLngs(COORDS.slice(0, step+1));
    prog.textContent = (step+1) + ' / ' + N;
    info.textContent = 'Time: ' + TIMES[step] + '   Heading: ' + COURSES[step] + ' deg';
  }}

  function play() {{
    if (step >= N-1) step = -1;
    clearInterval(timer);
    timer = setInterval(() => {{
      if (step >= N-1) {{ clearInterval(timer); timer=null; return; }}
      goTo(step+1);
    }}, parseInt(spSel.value));
  }}

  function pause() {{ clearInterval(timer); timer=null; }}
  function reset() {{ pause(); goTo(0); }}

  document.getElementById('btnPlay').onclick  = play;
  document.getElementById('btnPause').onclick = pause;
  document.getElementById('btnReset').onclick = reset;
  slider.oninput = () => {{ pause(); goTo(parseInt(slider.value)); }};
  spSel.onchange = () => {{ if (timer) {{ pause(); play(); }} }};

  goTo(0);
</script>
</body>
</html>"""


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🚜 Heavy Equipment GPS Dashboard")
st.caption("Upload CSV tracking files per equipment. Data is persisted as JSON for trend analysis.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Upload Tracking File")
    eq_name  = st.text_input("Equipment Name / ID", placeholder="e.g. Excavator-01")
    uploaded = st.file_uploader("GPS CSV file", type=["csv"])

    if uploaded and eq_name.strip():
        if st.button("➕ Process & Save", type="primary"):
            try:
                df_new  = parse_csv(uploaded)
                metrics = compute_metrics(df_new)
                store   = load_store()
                if eq_name not in store:
                    store[eq_name] = []
                entry = {
                    "file":    uploaded.name,
                    "loaded":  datetime.now().isoformat(),
                    "metrics": metrics,
                    "track":   df_new[["lat","lng","course","tstamp"]].astype(str).to_dict(orient="records"),
                }
                store[eq_name].append(entry)
                save_store(store)
                st.success(f"Saved! Total distance: **{metrics['total_distance_km']} km**")
                st.session_state["active_eq"]      = eq_name
                st.session_state["active_df"]      = df_new
                st.session_state["active_metrics"] = metrics
            except Exception as e:
                st.error(f"Error: {e}")
    elif uploaded and not eq_name.strip():
        st.warning("Enter an equipment name first.")

    st.divider()
    store_view = load_store()
    if store_view:
        st.subheader("📋 Saved Equipment")
        sel = st.selectbox("View equipment", list(store_view.keys()))
        if sel:
            sessions = store_view[sel]
            idx = st.selectbox("Session", range(len(sessions)),
                               format_func=lambda i: sessions[i]["file"])
            if st.button("📊 Load Session"):
                entry = sessions[idx]
                df_loaded = pd.DataFrame(entry["track"])
                df_loaded["lat"]    = df_loaded["lat"].astype(float)
                df_loaded["lng"]    = df_loaded["lng"].astype(float)
                df_loaded["course"] = df_loaded["course"].astype(int)
                df_loaded["tstamp"] = pd.to_datetime(df_loaded["tstamp"])
                st.session_state["active_eq"]      = sel
                st.session_state["active_df"]      = df_loaded
                st.session_state["active_metrics"] = entry["metrics"]

# ── Main panel ────────────────────────────────────────────────────────────────
if "active_df" not in st.session_state:
    st.info("👈 Upload a CSV file and give the equipment a name, then click **Process & Save**.")
    store_view2 = load_store()
    if store_view2:
        st.subheader("📈 All Equipment — Total Distance Summary")
        rows = []
        for name, sessions in store_view2.items():
            for s in sessions:
                mm = s["metrics"]
                rows.append({
                    "Equipment":            name,
                    "File":                 s["file"],
                    "Total Distance (km)":  mm["total_distance_km"],
                    "Operating Range (km)": mm["bbox_diagonal_km"],
                    "Duration (h)":         mm["duration_hours"],
                    "Avg Speed (km/h)":     mm["avg_speed_kmh"],
                    "Max Speed (km/h)":     mm["max_speed_kmh"],
                    "Idle %":               mm["idle_pct"],
                    "Data Points":          mm["data_points"],
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

else:
    df    = st.session_state["active_df"]
    m_    = st.session_state["active_metrics"]
    eq_id = st.session_state["active_eq"]

    st.subheader(f"📍 Equipment: {eq_id}")

    # KPI cards
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🛣️ Total Distance",    f"{m_['total_distance_km']} km")
    c2.metric("📐 Operating Range",   f"{m_['bbox_diagonal_km']} km")
    c3.metric("⏱️ Active Duration",   f"{m_['duration_hours']} h")
    c4.metric("⚡ Avg Speed",         f"{m_['avg_speed_kmh']} km/h")
    c5.metric("🏁 Max Speed",         f"{m_['max_speed_kmh']} km/h")

    c6, c7, c8, c9 = st.columns(4)
    c6.metric("😴 Idle Time",          f"{m_['idle_pct']} %")
    c7.metric("🧭 Dominant Direction", m_['dominant_direction'])
    c8.metric("📡 Data Points",        m_['data_points'])
    start_dt = pd.to_datetime(m_['start_time'])
    c9.metric("📅 Session Date",       start_dt.strftime("%d %b %Y"))

    st.divider()

    # ── Map tabs ──────────────────────────────────────────────────────────────
    tab1, tab2 = st.tabs(["🗺️ Animated Track (live)", "▶️ Step-by-step Playback"])

    with tab1:
        st.caption("Blue marching-ants line shows the direction of travel in real-time.")
        fmap = build_static_map(df)
        st_folium(fmap, width=None, height=480, use_container_width=True)

    with tab2:
        st.caption("Use Play / Pause / Reset and the speed selector to replay the route step by step.")
        html_src = build_playback_html(df)
        html_src = html_src.encode("utf-8", errors="ignore").decode("utf-8")
        st.components.v1.html(html_src, height=520, scrolling=False)

    st.divider()

    # Side-by-side charts
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("📊 Hourly Activity")
        hourly = m_["hourly_activity"]
        chart_df = pd.DataFrame({"Hour": list(range(24)),
                                  "GPS Pings": [hourly.get(h, 0) for h in range(24)]})
        st.bar_chart(chart_df.set_index("Hour"))

    with col_b:
        st.subheader("🧭 Heading Distribution")
        compass_counts = df["course"].apply(bearing_to_compass).value_counts().reset_index()
        compass_counts.columns = ["Direction", "Count"]
        st.bar_chart(compass_counts.set_index("Direction"))

    st.divider()

    # Speed profile
    st.subheader("📈 Speed Profile Over Time")
    lats  = df["lat"].tolist()
    lngs  = df["lng"].tolist()
    times = df["tstamp"].tolist()
    speed_rows = []
    for i in range(1, len(df)):
        d   = haversine_m(lats[i-1], lngs[i-1], lats[i], lngs[i])
        dt  = (times[i] - times[i-1]).total_seconds()
        spd = (d / dt * 3.6) if dt > 0 else 0
        speed_rows.append({"Time": times[i], "Speed (km/h)": round(spd, 2)})
    st.line_chart(pd.DataFrame(speed_rows).set_index("Time"))

    st.divider()

    with st.expander("🔍 Raw GPS Data"):
        st.dataframe(df, use_container_width=True)

    store_dl = load_store()
    st.download_button(
        "⬇️ Download Full JSON Store",
        data=json.dumps(store_dl, indent=2, default=str),
        file_name="equipment_gps_store.json",
        mime="application/json",
    )