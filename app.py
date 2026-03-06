import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import json, os, hashlib, time
import numpy as np

# ── Telemetry module ──────────────────────────────────────────────────────────
try:
    from telemetry import (
        fetch_lap_telemetry, make_channel_map,
        build_fastest_driver_map, make_telemetry_traces, extract_track_outline,
        make_linked_telemetry_figure,
    )
    TELEMETRY_AVAILABLE = True
except ImportError:
    TELEMETRY_AVAILABLE = False

# ─── CONFIG ───────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="F1 Analysis Dashboard",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_URL     = "https://api.openf1.org/v1"
CACHE_DIR    = os.path.join(os.path.dirname(__file__), ".f1_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

TEAM_COLORS = {
    "Red Bull Racing": "#3671C6", "Ferrari": "#E8002D",
    "Mercedes": "#27F4D2",        "McLaren": "#FF8000",
    "Aston Martin": "#229971",    "Alpine": "#FF87BC",
    "Williams": "#64C4FF",        "RB": "#6692FF",
    "Kick Sauber": "#52E252",     "Haas F1 Team": "#B6BABD",
}
DEFAULT_COLOR = "#FFFFFF"

COMPOUND_COLORS = {
    "SOFT": "#FF4444", "MEDIUM": "#FFCC00", "HARD": "#FFFFFF",
    "INTERMEDIATE": "#44AA44", "WET": "#4488FF", "UNKNOWN": "#888888",
}

SESSION_LAP_COUNTS = {
    "Race": "high", "Sprint": "medium",
    "Practice 1": "medium", "Practice 2": "medium", "Practice 3": "medium",
    "Qualifying": "low", "Sprint Qualifying": "low",
}

# ─── PERSISTENT DISK CACHE ────────────────────────────────────────────────────

def _cache_key(endpoint, params):
    def _to_native(obj):
        if hasattr(obj, "item"): return obj.item()
        if isinstance(obj, dict): return {k: _to_native(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [_to_native(v) for v in obj]
        return obj
    raw = json.dumps({"e": endpoint, "p": _to_native(params)}, sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()

def _cache_path(key):
    return os.path.join(CACHE_DIR, f"{key}.json")

def _cache_read(key):
    p = _cache_path(key)
    if os.path.exists(p):
        try:
            with open(p) as f:
                obj = json.load(f)
            # 24-hour TTL for historical data
            if time.time() - obj.get("ts", 0) < 86400:
                return obj["data"]
        except Exception:
            pass
    return None

def _cache_write(key, data):
    try:
        with open(_cache_path(key), "w") as f:
            json.dump({"ts": time.time(), "data": data}, f)
    except Exception:
        pass

# ─── STYLING ──────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Exo+2:wght@300;400;600;700;900&family=Share+Tech+Mono&display=swap');
  html, body, [class*="css"] { font-family:'Exo 2',sans-serif; background:#0d0d0f; color:#e8e8e8; }
  .stApp { background:#0d0d0f; }
  section[data-testid="stSidebar"] { background:#111115; border-right:1px solid #222230; }
  .metric-card {
    background:linear-gradient(135deg,#16161e,#1c1c28);
    border:1px solid #2a2a3d; border-radius:8px; padding:14px 18px; text-align:center;
  }
  .metric-label {
    font-size:11px; font-weight:600; letter-spacing:2px;
    text-transform:uppercase; color:#888; margin-bottom:5px;
  }
  .metric-value { font-family:'Share Tech Mono',monospace; font-size:20px; font-weight:700; color:#f0f0f0; }
  .section-header {
    font-size:11px; font-weight:700; letter-spacing:3px; text-transform:uppercase;
    color:#e10600; margin-bottom:10px; padding-bottom:5px; border-bottom:1px solid #2a2a3d;
  }
  .avail-badge {
    display:inline-block; padding:2px 8px; border-radius:3px;
    font-size:10px; font-weight:700; letter-spacing:1px; margin:2px;
  }
  .avail-high  { background:#1a4d1a; color:#4fff88; border:1px solid #4fff88; }
  .avail-med   { background:#4d3d00; color:#ffcc00; border:1px solid #ffcc00; }
  .avail-low   { background:#4d0000; color:#ff6666; border:1px solid #ff6666; }
  .stSelectbox label, .stMultiSelect label {
    font-size:11px; font-weight:600; letter-spacing:1.5px;
    text-transform:uppercase; color:#888 !important;
  }
</style>
""", unsafe_allow_html=True)

# ─── API HELPERS ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def fetch(endpoint, **params):
    key = _cache_key(endpoint, params)
    cached = _cache_read(key)
    if cached is not None:
        return cached
    try:
        r = requests.get(f"{BASE_URL}/{endpoint}", params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        _cache_write(key, data)
        return data
    except Exception as e:
        st.error(f"API error on /{endpoint}: {e}")
        return []

@st.cache_data(ttl=3600)
def get_meetings(year):
    data = fetch("meetings", year=year)
    if not data: return pd.DataFrame()
    return pd.DataFrame(data).sort_values("date_start").reset_index(drop=True)

@st.cache_data(ttl=3600)
def get_sessions(meeting_key):
    data = fetch("sessions", meeting_key=meeting_key)
    return pd.DataFrame(data) if data else pd.DataFrame()

@st.cache_data(ttl=3600)
def get_drivers(session_key):
    data = fetch("drivers", session_key=session_key)
    return pd.DataFrame(data) if data else pd.DataFrame()

@st.cache_data(ttl=3600)
def get_laps(session_key, driver_number=None):
    params = {"session_key": session_key}
    if driver_number: params["driver_number"] = driver_number
    data = fetch("laps", **params)
    return pd.DataFrame(data) if data else pd.DataFrame()

@st.cache_data(ttl=3600)
def get_stints(session_key):
    data = fetch("stints", session_key=session_key)
    return pd.DataFrame(data) if data else pd.DataFrame()

@st.cache_data(ttl=3600)
def get_position(session_key):
    data = fetch("position", session_key=session_key)
    return pd.DataFrame(data) if data else pd.DataFrame()

@st.cache_data(ttl=3600)
def get_weather(session_key):
    data = fetch("weather", session_key=session_key)
    return pd.DataFrame(data) if data else pd.DataFrame()

@st.cache_data(ttl=3600)
def get_team_radio(session_key, driver_number):
    data = fetch("team_radio", session_key=session_key, driver_number=driver_number)
    return pd.DataFrame(data) if data else pd.DataFrame()

# ─── UTILS ────────────────────────────────────────────────────────────────────

def fmt_time(seconds):
    try:
        s = float(seconds)
        if pd.isna(s) or s <= 0: return "—"
        m = int(s // 60)
        return f"{m}:{s - m*60:06.3f}"
    except: return "—"

def fmt_delta(delta):
    try:
        d = float(delta)
        return f"+{d:.3f}s" if d > 0 else f"{d:.3f}s"
    except: return "—"

def driver_color(team_name):
    for k, v in TEAM_COLORS.items():
        if k.lower() in str(team_name).lower(): return v
    return DEFAULT_COLOR

def hex_to_rgba(hex_color, alpha=0.3):
    h = hex_color.lstrip("#")
    if len(h) == 6:
        r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
        return f"rgba({r},{g},{b},{alpha})"
    return f"rgba(200,200,200,{alpha})"

# ─── HEADER ───────────────────────────────────────────────────────────────────

st.markdown("""
<h1 style='margin-bottom:0;color:#ffffff;font-size:2.2rem;'>
  🏎️ F1 <span style='color:#e10600;'>ANALYSIS</span> DASHBOARD
</h1>
<p style='color:#555;font-size:13px;margin-top:2px;letter-spacing:2px;'>
  POWERED BY OPENF1 API · 2025 / 2026 SEASONS
</p>
""", unsafe_allow_html=True)
st.divider()

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="section-header">🗓 Session Selection</div>', unsafe_allow_html=True)
    year = st.selectbox("Season", [2026, 2025], index=1)
    meetings_df = get_meetings(year)

    if meetings_df.empty:
        st.warning(f"No data for {year} yet.")
        st.stop()

    meeting_labels = meetings_df.apply(
        lambda r: f"R{r.name+1} · {r.get('meeting_name', r.get('country_name','?'))}", axis=1
    ).tolist()
    meeting_idx = st.selectbox("Grand Prix", range(len(meeting_labels)),
                               format_func=lambda i: meeting_labels[i],
                               index=len(meeting_labels)-1)
    selected_meeting = meetings_df.iloc[meeting_idx]
    meeting_key = selected_meeting["meeting_key"]

    sessions_df = get_sessions(meeting_key)
    if sessions_df.empty:
        st.warning("No sessions found.")
        st.stop()

    # ── Data availability badges ──────────────────────────────────────────
    st.markdown('<div class="section-header" style="margin-top:10px;">Session Availability</div>',
                unsafe_allow_html=True)
    badge_html = ""
    for _, s in sessions_df.iterrows():
        sname = s.get("session_name","?")
        level = SESSION_LAP_COUNTS.get(sname, "medium")
        badge_class = {"high":"avail-high","medium":"avail-med","low":"avail-low"}[level]
        badge_html += f'<span class="avail-badge {badge_class}">{sname}</span>'
    st.markdown(badge_html, unsafe_allow_html=True)
    st.caption("🟢 Rich data  🟡 Moderate  🔴 Sparse (telemetry may not load)")

    session_types = sessions_df["session_name"].tolist()
    session_idx = st.selectbox("Session", range(len(session_types)),
                               format_func=lambda i: session_types[i],
                               index=len(session_types)-1)
    selected_session = sessions_df.iloc[session_idx]
    session_key = int(selected_session["session_key"])
    session_type = selected_session.get("session_name","")

    # Warn if sparse session selected for telemetry
    if SESSION_LAP_COUNTS.get(session_type,"medium") == "low":
        st.warning("⚠️ Qualifying sessions have few laps — Track Telemetry tab may have no data. "
                   "Practice sessions work best.")

    st.markdown("---")
    st.markdown('<div class="section-header">🏁 Drivers</div>', unsafe_allow_html=True)
    drivers_df = get_drivers(session_key)
    if drivers_df.empty:
        st.warning("No driver data.")
        st.stop()

    driver_labels = drivers_df.apply(
        lambda r: f"{r.get('name_acronym','???')} — {r.get('full_name','?')}", axis=1
    ).tolist()
    selected_driver_indices = st.multiselect(
        "Compare Drivers (up to 5)",
        options=range(len(driver_labels)),
        format_func=lambda i: driver_labels[i],
        default=list(range(min(3, len(driver_labels)))),
        max_selections=5,
    )
    if not selected_driver_indices:
        st.info("Select at least one driver.")
        st.stop()
    selected_drivers = drivers_df.iloc[selected_driver_indices].reset_index(drop=True)

    st.markdown("---")
    st.markdown('<div class="section-header">⚙️ Options</div>', unsafe_allow_html=True)
    exclude_pits = st.toggle("Exclude pit-out / in-laps", value=True)
    show_tyre    = st.toggle("Show tyre compound",         value=True)

    st.markdown("---")
    st.markdown('<div class="section-header">💾 Cache</div>', unsafe_allow_html=True)
    cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".json")]
    st.caption(f"{len(cache_files)} items cached to disk")
    if st.button("🗑 Clear cache"):
        for f in cache_files:
            try: os.remove(os.path.join(CACHE_DIR, f))
            except: pass
        st.cache_data.clear()
        st.success("Cache cleared!")
        st.rerun()

# ─── LOAD LAP DATA ────────────────────────────────────────────────────────────

with st.spinner("Fetching lap data…"):
    all_laps = []
    prog = st.progress(0)
    for i, (_, row) in enumerate(selected_drivers.iterrows()):
        drv_num = int(row["driver_number"])
        laps = get_laps(session_key, drv_num)
        if not laps.empty:
            laps["driver_number"] = drv_num
            laps["acronym"]       = row.get("name_acronym", str(drv_num))
            laps["team_name"]     = row.get("team_name", "")
            laps["team_colour"]   = row.get("team_colour", driver_color(row.get("team_name","")))
            all_laps.append(laps)
        prog.progress((i+1)/len(selected_drivers))
    prog.empty()

if not all_laps:
    st.error("No lap data found.")
    st.stop()

laps_df = pd.concat(all_laps, ignore_index=True)
for col in ["duration_sector_1","duration_sector_2","duration_sector_3","lap_duration","lap_number"]:
    if col in laps_df.columns:
        laps_df[col] = pd.to_numeric(laps_df[col], errors="coerce")

if exclude_pits:
    laps_df = laps_df[laps_df.get("is_pit_out_lap", pd.Series(False, index=laps_df.index)) != True]
    laps_df = laps_df[laps_df["lap_duration"].notna() & (laps_df["lap_duration"] > 0)]

# Tyre compound
stints_df = get_stints(session_key) if show_tyre else pd.DataFrame()
if show_tyre and not stints_df.empty:
    stints_df["driver_number"] = pd.to_numeric(stints_df["driver_number"], errors="coerce")
    def _compound(row):
        ds = stints_df[stints_df["driver_number"] == row["driver_number"]]
        for _, st_ in ds.iterrows():
            if st_.get("lap_start",0) <= row["lap_number"] <= st_.get("lap_end",9999):
                return st_.get("compound","UNKNOWN")
        return "UNKNOWN"
    laps_df["compound"] = laps_df.apply(_compound, axis=1)
else:
    laps_df["compound"] = "UNKNOWN"

# ─── SUMMARY METRICS ──────────────────────────────────────────────────────────

st.markdown(f"""
<div class='section-header'>📊 {selected_meeting.get('meeting_name','').upper()} · {session_type.upper()}</div>
""", unsafe_allow_html=True)

mcols = st.columns(len(selected_drivers))
for i, (_, drv) in enumerate(selected_drivers.iterrows()):
    dl = laps_df[laps_df["acronym"] == drv.get("name_acronym")]
    color = f"#{str(drv.get('team_colour','ffffff')).lstrip('#')}"
    with mcols[i]:
        st.markdown(f"""
        <div class='metric-card' style='border-top:3px solid {color};'>
          <div style='font-size:17px;font-weight:900;color:{color};letter-spacing:2px;'>{drv.get('name_acronym','???')}</div>
          <div style='font-size:11px;color:#555;margin-bottom:8px;'>{drv.get('team_name','')}</div>
          <div class='metric-label'>Best Lap</div>
          <div class='metric-value'>{fmt_time(dl['lap_duration'].min() if not dl.empty else None)}</div>
          <div style='margin-top:8px;display:flex;justify-content:space-around;'>
            {''.join(f"<div><div class='metric-label'>S{j+1}</div><div style='font-family:monospace;font-size:12px;color:#ccc;'>{fmt_time(dl[f'duration_sector_{j+1}'].min() if f'duration_sector_{j+1}' in dl else None)}</div></div>" for j in range(3))}
          </div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── TABS ─────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📈 Lap Times",
    "🔵 Sector Breakdown",
    "⚡ Sector Delta",
    "🗺️ Track Telemetry",
    "🏁 Race Position",
    "🔧 Tyre Strategy",
    "📻 Team Radio",
    "📋 Lap Table",
])

# ── helpers shared across tabs ────────────────────────────────────────────────
sectors      = ["duration_sector_1","duration_sector_2","duration_sector_3"]
sector_labels = ["Sector 1","Sector 2","Sector 3"]

def drv_color_str(drv_row):
    return f"#{str(drv_row.get('team_colour','ffffff')).lstrip('#')}"

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LAP TIMES
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Lap Time Evolution</div>', unsafe_allow_html=True)

    # ── Weather overlay option ────────────────────────────────────────────
    weather_df = get_weather(session_key)
    show_weather = st.checkbox("Overlay track temperature", value=False) \
                  if not weather_df.empty else False

    fig = go.Figure()

    if show_weather and not weather_df.empty:
        weather_df["date"] = pd.to_datetime(weather_df["date"], utc=True)
        weather_df["track_temperature"] = pd.to_numeric(weather_df["track_temperature"], errors="coerce")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.75, 0.25], vertical_spacing=0.05)
        use_subplots = True
    else:
        use_subplots = False

    for _, drv in selected_drivers.iterrows():
        dl   = laps_df[laps_df["acronym"] == drv.get("name_acronym")].copy()
        if dl.empty: continue
        color   = drv_color_str(drv)
        acronym = drv.get("name_acronym","???")
        hover   = dl.apply(lambda r:
            f"<b>{acronym}</b> — Lap {int(r.get('lap_number',0))}<br>"
            f"Time: {fmt_time(r['lap_duration'])}<br>"
            f"S1:{fmt_time(r.get('duration_sector_1'))} S2:{fmt_time(r.get('duration_sector_2'))} S3:{fmt_time(r.get('duration_sector_3'))}<br>"
            f"Tyre: {r.get('compound','?')}", axis=1)

        scatter_kwargs = dict(
            x=dl["lap_number"], y=dl["lap_duration"],
            mode="lines+markers", name=acronym,
            line=dict(color=color, width=2),
            marker=dict(size=7,
                        color=[COMPOUND_COLORS.get(c,"#888") for c in dl.get("compound",["UNKNOWN"]*len(dl))],
                        line=dict(color=color, width=1.5)),
            hovertemplate=hover+"<extra></extra>",
        )
        if use_subplots:
            fig.add_trace(go.Scatter(**scatter_kwargs), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(**scatter_kwargs))

    if use_subplots and not weather_df.empty:
        fig.add_trace(go.Scatter(
            x=weather_df["date"], y=weather_df["track_temperature"],
            mode="lines", name="Track Temp (°C)",
            line=dict(color="#ff8800", width=1.5, dash="dot"),
            hovertemplate="Track: %{y:.1f}°C<extra></extra>",
        ), row=2, col=1)
        fig.update_yaxes(title_text="Lap Time (s)", row=1, col=1)
        fig.update_yaxes(title_text="°C", gridcolor="#222", row=2, col=1)

    _dark = dict(template="plotly_dark", paper_bgcolor="#0d0d0f", plot_bgcolor="#13131a",
                 font=dict(family="Exo 2",color="#ccc"),
                 legend=dict(bgcolor="#0d0d0f",bordercolor="#333",borderwidth=1),
                 hovermode="x unified", height=460, margin=dict(l=10,r=10,t=20,b=10))
    fig.update_layout(**_dark)
    fig.update_xaxes(gridcolor="#222", zeroline=False)
    fig.update_yaxes(gridcolor="#222", zeroline=False)
    st.plotly_chart(fig, use_container_width=True)

    # ── Consistency stats ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">Lap Time Consistency</div>', unsafe_allow_html=True)
    st.caption("Excludes outliers >107% of median. Lower std dev = more consistent.")
    stat_cols = st.columns(len(selected_drivers))
    for i, (_, drv) in enumerate(selected_drivers.iterrows()):
        dl = laps_df[laps_df["acronym"] == drv.get("name_acronym")].copy()
        color = drv_color_str(drv)
        if dl.empty or "lap_duration" not in dl.columns:
            continue
        times = dl["lap_duration"].dropna()
        median = times.median()
        clean  = times[times <= median * 1.07]
        std    = clean.std()
        best   = clean.min()
        worst  = clean.max()
        spread = worst - best
        # Consistency score: 100 = perfectly consistent, lower = more spread
        score  = max(0, 100 - (std / median * 1000)) if median > 0 else 0
        with stat_cols[i]:
            st.markdown(f"""
            <div class='metric-card' style='border-top:3px solid {color};'>
              <div style='font-size:15px;font-weight:900;color:{color};'>{drv.get('name_acronym','?')}</div>
              <div style='margin-top:8px;'>
                <div class='metric-label'>Consistency Score</div>
                <div class='metric-value' style='font-size:22px;color:{"#4fff88" if score>80 else "#ffcc00" if score>60 else "#ff6666"};'>{score:.0f}</div>
              </div>
              <div style='display:flex;justify-content:space-around;margin-top:8px;font-size:11px;color:#888;'>
                <div>σ <span style='color:#ccc;font-family:monospace;'>{std:.3f}s</span></div>
                <div>Spread <span style='color:#ccc;font-family:monospace;'>{spread:.3f}s</span></div>
                <div>N <span style='color:#ccc;font-family:monospace;'>{len(clean)}</span></div>
              </div>
            </div>""", unsafe_allow_html=True)

    # Tyre legend
    if show_tyre:
        st.markdown('<div class="section-header" style="margin-top:12px;">Tyre Legend</div>', unsafe_allow_html=True)
        lcols = st.columns(5)
        for ci, (compound, ccolor) in enumerate([c for c in COMPOUND_COLORS.items() if c[0] != "UNKNOWN"]):
            with lcols[ci]:
                st.markdown(f"<div style='text-align:center;'><span style='background:{ccolor};color:#000;padding:3px 10px;border-radius:3px;font-size:11px;font-weight:700;'>{compound}</span></div>", unsafe_allow_html=True)

    # ── Export ────────────────────────────────────────────────────────────
    export_cols = ["acronym","lap_number","lap_duration","duration_sector_1","duration_sector_2","duration_sector_3","compound"]
    export_df   = laps_df[[c for c in export_cols if c in laps_df.columns]].copy()
    st.download_button("⬇️ Export lap data (CSV)", export_df.to_csv(index=False),
                       file_name=f"f1_laps_{session_key}.csv", mime="text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SECTOR BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Best Sector Times Comparison</div>', unsafe_allow_html=True)
    fig2 = go.Figure()
    for _, drv in selected_drivers.iterrows():
        dl    = laps_df[laps_df["acronym"] == drv.get("name_acronym")]
        color = drv_color_str(drv)
        bests = [dl[s].min() if s in dl.columns else None for s in sectors]
        fig2.add_trace(go.Bar(name=drv.get("name_acronym","?"), x=sector_labels, y=bests,
                              marker_color=color, marker_line=dict(color="#000",width=1),
                              hovertemplate="<b>%{fullData.name}</b> — %{x}<br>%{y:.3f}s<extra></extra>"))
    fig2.update_layout(template="plotly_dark", paper_bgcolor="#0d0d0f", plot_bgcolor="#13131a",
                       barmode="group", height=400,
                       xaxis=dict(gridcolor="#222"), yaxis=dict(title="Best Sector Time (s)",gridcolor="#222",tickformat=".3f"),
                       legend=dict(bgcolor="#0d0d0f",bordercolor="#333",borderwidth=1),
                       margin=dict(l=10,r=10,t=20,b=10))
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">Sector Composition Per Lap</div>', unsafe_allow_html=True)
    drv_pick = st.selectbox("Driver", selected_drivers["name_acronym"].tolist(), key="stacked_drv")
    dl_pick  = laps_df[laps_df["acronym"] == drv_pick]
    if not dl_pick.empty:
        fig3 = go.Figure()
        for s, label, c in zip(sectors, sector_labels, ["#e10600","#ff8c00","#ffd700"]):
            if s in dl_pick.columns:
                fig3.add_trace(go.Bar(name=label, x=dl_pick["lap_number"], y=dl_pick[s],
                                      marker_color=c,
                                      hovertemplate=f"<b>{label}</b> Lap %{{x}}<br>%{{y:.3f}}s<extra></extra>"))
        fig3.update_layout(template="plotly_dark", paper_bgcolor="#0d0d0f", plot_bgcolor="#13131a",
                           barmode="stack", height=380,
                           xaxis=dict(title="Lap",gridcolor="#222"),
                           yaxis=dict(title="Time (s)",gridcolor="#222"),
                           margin=dict(l=10,r=10,t=20,b=10))
        st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SECTOR DELTA
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Sector Delta vs Reference Driver</div>', unsafe_allow_html=True)
    ref_driver = st.selectbox("Reference Driver", selected_drivers["name_acronym"].tolist(), key="ref_driver")
    ref_laps   = laps_df[laps_df["acronym"] == ref_driver].copy()

    if ref_laps.empty:
        st.warning("No laps for reference driver.")
    else:
        for sector, label in zip(sectors, sector_labels):
            if sector not in laps_df.columns: continue
            st.markdown(f"**{label}**")
            fig_d = go.Figure()
            ref_per_lap = ref_laps.set_index("lap_number")[sector]
            for _, drv in selected_drivers.iterrows():
                acr = drv.get("name_acronym")
                if acr == ref_driver: continue
                dl  = laps_df[laps_df["acronym"] == acr].copy()
                if dl.empty: continue
                color = drv_color_str(drv)
                dl    = dl.set_index("lap_number")
                common = dl.index.intersection(ref_per_lap.index)
                delta  = dl.loc[common, sector] - ref_per_lap.loc[common]
                fig_d.add_trace(go.Scatter(x=common, y=delta, mode="lines+markers",
                                           name=f"{acr} vs {ref_driver}",
                                           line=dict(color=color,width=2), marker=dict(size=5,color=color),
                                           hovertemplate=f"<b>{acr}</b> Lap %{{x}}<br>Delta: %{{y:.3f}}s<extra></extra>"))
            fig_d.add_hline(y=0, line_dash="dash", line_color="#555", line_width=1)
            fig_d.update_layout(template="plotly_dark", paper_bgcolor="#0d0d0f", plot_bgcolor="#13131a",
                                xaxis=dict(title="Lap",gridcolor="#222"),
                                yaxis=dict(title=f"Δ vs {ref_driver} (s)",gridcolor="#222",tickformat="+.3f"),
                                legend=dict(bgcolor="#0d0d0f",bordercolor="#333",borderwidth=1),
                                height=280, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig_d, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — TRACK TELEMETRY
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Track Map — Telemetry Overlay</div>', unsafe_allow_html=True)

    if not TELEMETRY_AVAILABLE:
        st.error("telemetry.py not found in the same folder as app.py.")
    elif SESSION_LAP_COUNTS.get(session_type,"medium") == "low":
        st.warning("⚠️ This is a qualifying session with very few laps. Location data is often absent. "
                   "Switch to a Practice session for best results.")
        if not st.checkbox("Try anyway"):
            st.stop()
    else:
        from telemetry import extract_track_outline, build_fastest_driver_map, make_channel_map, make_linked_telemetry_figure

        map_mode = st.radio("Map mode",
                            ["🏆 Fastest Driver","📡 Single Driver Channel"],
                            horizontal=True, key="map_mode")

        tc1, tc2 = st.columns([3,2])
        with tc1:
            all_lap_nums = sorted(laps_df["lap_number"].dropna().astype(int).unique().tolist())
            vl = laps_df[laps_df["lap_duration"].notna() & (laps_df["lap_duration"]>0)]
            best_overall_lap = int(vl.loc[vl["lap_duration"].idxmin(),"lap_number"]) if not vl.empty else (all_lap_nums[0] if all_lap_nums else 1)
            default_lap_idx = all_lap_nums.index(best_overall_lap) if best_overall_lap in all_lap_nums else 0
            selected_lap = st.selectbox("Lap", all_lap_nums, index=default_lap_idx,
                                        format_func=lambda n: f"Lap {n}" + (" ★ fastest" if n==best_overall_lap else ""),
                                        key="tel_lap")
        with tc2:
            ref_outline_driver = st.selectbox("Track outline reference driver",
                                              selected_drivers["name_acronym"].tolist(),
                                              key="ref_outline_driver")

        if map_mode == "📡 Single Driver Channel":
            sc1, sc2 = st.columns(2)
            with sc1:
                single_drv_name = st.selectbox("Driver", selected_drivers["name_acronym"].tolist(), key="single_drv")
            with sc2:
                channel_opts = {"Throttle %":"throttle","Brake":"brake","Speed":"speed","Gear":"n_gear"}
                channel_label = st.selectbox("Channel", list(channel_opts.keys()), key="channel")
                channel = channel_opts[channel_label]

        drv_colors_fetched = {r.get("name_acronym"): drv_color_str(r) for _,r in selected_drivers.iterrows()}
        ref_drv_row = selected_drivers[selected_drivers["name_acronym"]==ref_outline_driver].iloc[0]
        ref_drv_num = int(ref_drv_row["driver_number"])

        @st.cache_data(ttl=3600, show_spinner=False)
        def _cached_outline(s_key, d_num, _h):
            return extract_track_outline(s_key, d_num, laps_df)

        with st.spinner(f"Building track outline from {ref_outline_driver}'s best lap…"):
            track_x, track_y = _cached_outline(session_key, ref_drv_num, len(laps_df))

        if len(track_x) == 0:
            st.warning(f"No location data for {ref_outline_driver}. Try a different driver or a Practice session.")

        @st.cache_data(ttl=3600, show_spinner=False)
        def _cached_tel(s_key, d_num, lap_num, _h):
            return fetch_lap_telemetry(s_key, d_num, lap_num, laps_df)

        tel_dict = {}
        fp = st.progress(0, text="Fetching telemetry…")
        drv_list = selected_drivers["name_acronym"].tolist()
        for fi, (_, drv) in enumerate(selected_drivers.iterrows()):
            acr   = drv.get("name_acronym"); d_num = int(drv["driver_number"])
            lap_ex = not laps_df[(laps_df["driver_number"]==d_num)&(laps_df["lap_number"]==selected_lap)].empty
            if lap_ex:
                td = _cached_tel(session_key, d_num, selected_lap, len(laps_df))
                if not td.empty: tel_dict[acr] = td
            fp.progress((fi+1)/len(drv_list), text=f"Fetching {acr}…")
        fp.empty()

        if not tel_dict:
            st.warning("No telemetry data for this lap. Try another lap or a Practice session.")
        else:
            def _mini(label, val):
                return f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value' style='font-size:15px;'>{val}</div></div>"

            mc = st.columns(len(tel_dict)+1)
            mc[0].markdown(_mini("Lap", str(selected_lap)), unsafe_allow_html=True)
            for ci,(acr,td) in enumerate(tel_dict.items()):
                dlr = laps_df[(laps_df["acronym"]==acr)&(laps_df["lap_number"]==selected_lap)]
                lt  = fmt_time(dlr["lap_duration"].values[0]) if not dlr.empty else "—"
                c   = drv_colors_fetched.get(acr,"#fff")
                mc[ci+1].markdown(f"<div class='metric-card' style='border-top:3px solid {c};'><div class='metric-label' style='color:{c};'>{acr}</div><div class='metric-value' style='font-size:15px;'>{lt}</div><div style='font-size:10px;color:#555;'>{len(td)} samples</div></div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)


            if map_mode == "🏆 Fastest Driver":
                map_fig = build_fastest_driver_map(tel_dict, drv_colors_fetched, track_x, track_y)
            else:
                map_fig = make_channel_map(tel_dict.get(single_drv_name, pd.DataFrame()),
                                           channel, single_drv_name,
                                           drv_colors_fetched.get(single_drv_name,"#fff"),
                                           track_x, track_y)

            st.caption("🖱️ Click a point on the track map to pin a marker on the telemetry graphs below.")

            map_event = st.plotly_chart(
                map_fig,
                use_container_width=True,
                on_select="rerun",
                key="map_click",
            )

            # Resolve clicked map xy -> nearest sample index per driver
            pinned_samples = {}
            if map_event and map_event.selection and map_event.selection.get("points"):
                pt = map_event.selection["points"][0]
                click_x = pt.get("x")
                click_y = pt.get("y")
                if click_x is not None and click_y is not None:
                    for acr, td in tel_dict.items():
                        if "x" not in td.columns or "y" not in td.columns:
                            continue
                        dx = pd.to_numeric(td["x"], errors="coerce").values
                        dy = pd.to_numeric(td["y"], errors="coerce").values
                        valid = ~(np.isnan(dx) | np.isnan(dy))
                        if not valid.any():
                            continue
                        dists = (dx[valid] - click_x)**2 + (dy[valid] - click_y)**2
                        nearest = int(np.where(valid)[0][np.argmin(dists)])
                        pinned_samples[acr] = nearest

            st.markdown('<div class="section-header">Throttle · Brake · Speed · Gear</div>', unsafe_allow_html=True)
            traces_fig = make_telemetry_traces(tel_dict, drv_colors_fetched)

            # Inject per-driver vertical lines at their pinned sample index
            if pinned_samples:
                for acr, sample_idx in pinned_samples.items():
                    color = drv_colors_fetched.get(acr, "#ffffff")
                    traces_fig.add_vline(
                        x=sample_idx,
                        line=dict(color=color, width=1.5, dash="dash"),
                        annotation_text=acr,
                        annotation_position="top",
                        annotation_font=dict(color=color, size=10),
                    )

            st.plotly_chart(traces_fig, use_container_width=True)

# TAB 5 — RACE POSITION
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">Race Position Over Time</div>', unsafe_allow_html=True)

    if "race" not in session_type.lower():
        st.info("Position data is most meaningful for Race sessions. Switch to Race in the sidebar for the full chart.")

    pos_df = get_position(session_key)
    if pos_df.empty:
        st.warning("No position data available for this session.")
    else:
        pos_df["date"]     = pd.to_datetime(pos_df["date"], utc=True)
        pos_df["position"] = pd.to_numeric(pos_df["position"], errors="coerce")
        pos_df["driver_number"] = pd.to_numeric(pos_df["driver_number"], errors="coerce")

        # Map driver numbers to acronyms
        drv_map = {int(r["driver_number"]): r.get("name_acronym","?") for _,r in drivers_df.iterrows()}
        pos_df["acronym"] = pos_df["driver_number"].map(drv_map)

        # Filter to selected drivers only
        selected_acronyms = selected_drivers["name_acronym"].tolist()
        pos_filt = pos_df[pos_df["acronym"].isin(selected_acronyms)].copy()

        if pos_filt.empty:
            st.warning("No position data for selected drivers.")
        else:
            fig_pos = go.Figure()

            # Lap-number proxy: bin position changes by lap using laps_df date_start
            # Build a lap->time mapping for x-axis labels
            lap_times_map = {}
            if "date_start" in laps_df.columns:
                lt = laps_df[["lap_number","date_start","acronym"]].dropna()
                for _,r in lt.iterrows():
                    key = (r["acronym"], int(r["lap_number"]))
                    lap_times_map[key] = pd.to_datetime(r["date_start"], utc=True)

            for _, drv in selected_drivers.iterrows():
                acr   = drv.get("name_acronym")
                color = drv_color_str(drv)
                dp    = pos_filt[pos_filt["acronym"]==acr].sort_values("date")
                if dp.empty: continue

                fig_pos.add_trace(go.Scatter(
                    x=dp["date"], y=dp["position"],
                    mode="lines", name=acr,
                    line=dict(color=color, width=2.5, shape="hv"),
                    hovertemplate=f"<b>{acr}</b><br>P%{{y}}<extra></extra>",
                ))

            fig_pos.update_yaxes(autorange="reversed", title="Position",
                                 tickvals=list(range(1,21)), gridcolor="#222")
            fig_pos.update_xaxes(title="Time", gridcolor="#222")
            fig_pos.update_layout(
                template="plotly_dark", paper_bgcolor="#0d0d0f", plot_bgcolor="#13131a",
                font=dict(family="Exo 2",color="#ccc"),
                legend=dict(bgcolor="#0d0d0f",bordercolor="#333",borderwidth=1),
                height=480, margin=dict(l=10,r=10,t=20,b=10),
                hovermode="x unified",
            )
            st.plotly_chart(fig_pos, use_container_width=True)
            st.download_button("⬇️ Export position data (CSV)",
                               pos_filt.to_csv(index=False),
                               file_name=f"f1_position_{session_key}.csv", mime="text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — TYRE STRATEGY
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-header">Tyre Stint Strategy</div>', unsafe_allow_html=True)

    stints_full = get_stints(session_key)
    if stints_full.empty:
        st.warning("No stint data available for this session.")
    else:
        stints_full["driver_number"] = pd.to_numeric(stints_full["driver_number"], errors="coerce")
        stints_full["lap_start"]     = pd.to_numeric(stints_full["lap_start"], errors="coerce")
        stints_full["lap_end"]       = pd.to_numeric(stints_full["lap_end"], errors="coerce")

        drv_num_map = {int(r["driver_number"]): r.get("name_acronym","?") for _,r in drivers_df.iterrows()}
        stints_full["acronym"] = stints_full["driver_number"].map(drv_num_map)

        selected_acronyms = selected_drivers["name_acronym"].tolist()
        st_filt = stints_full[stints_full["acronym"].isin(selected_acronyms)].copy()

        if st_filt.empty:
            st.warning("No stint data for selected drivers.")
        else:
            fig_st = go.Figure()

            for yi, acr in enumerate(selected_acronyms):
                drv_stints = st_filt[st_filt["acronym"]==acr].sort_values("lap_start")
                if drv_stints.empty: continue

                for _, stint in drv_stints.iterrows():
                    compound = stint.get("compound","UNKNOWN")
                    lap_s    = float(stint.get("lap_start",0) or 0)
                    lap_e    = float(stint.get("lap_end",   lap_s+1) or lap_s+1)
                    age      = int(stint.get("tyre_age_at_start",0) or 0)
                    color    = COMPOUND_COLORS.get(compound,"#888")
                    bar_color = color if compound != "HARD" else "#dddddd"

                    fig_st.add_trace(go.Bar(
                        x=[lap_e - lap_s],
                        base=[lap_s],
                        y=[acr],
                        orientation="h",
                        marker_color=bar_color,
                        marker_line=dict(color="#000",width=1),
                        name=compound,
                        legendgroup=compound,
                        showlegend=compound not in [t.name for t in fig_st.data],
                        hovertemplate=(
                            f"<b>{acr}</b> — {compound}<br>"
                            f"Laps {int(lap_s)}–{int(lap_e)}<br>"
                            f"Stint length: {int(lap_e-lap_s)} laps<br>"
                            f"Tyre age at start: {age} laps<extra></extra>"
                        ),
                        text=compound[:1],
                        textposition="inside",
                        textfont=dict(color="#000",size=10,family="Exo 2"),
                        insidetextanchor="middle",
                    ))

            fig_st.update_layout(
                template="plotly_dark", paper_bgcolor="#0d0d0f", plot_bgcolor="#13131a",
                font=dict(family="Exo 2",color="#ccc"),
                barmode="overlay",
                xaxis=dict(title="Lap Number", gridcolor="#222"),
                yaxis=dict(title="Driver", categoryorder="array",
                           categoryarray=list(reversed(selected_acronyms))),
                legend=dict(bgcolor="#0d0d0f",bordercolor="#333",borderwidth=1,
                            title="Compound"),
                height=max(280, len(selected_acronyms)*80),
                margin=dict(l=10,r=10,t=20,b=10),
            )
            st.plotly_chart(fig_st, use_container_width=True)

            # Pit stop summary
            pit_data = fetch("pit", session_key=session_key)
            if pit_data:
                pit_df = pd.DataFrame(pit_data)
                pit_df["driver_number"] = pd.to_numeric(pit_df["driver_number"], errors="coerce")
                pit_df["acronym"] = pit_df["driver_number"].map(drv_num_map)
                pit_filt = pit_df[pit_df["acronym"].isin(selected_acronyms)]
                if not pit_filt.empty:
                    st.markdown('<div class="section-header">Pit Stop Times</div>', unsafe_allow_html=True)
                    pit_display = pit_filt[["acronym","lap_number","stop_duration","lane_duration"]].copy()
                    pit_display.columns = ["Driver","Lap","Stop (s)","Lane (s)"]
                    pit_display = pit_display.sort_values(["Driver","Lap"])
                    st.dataframe(pit_display, use_container_width=True, hide_index=True)

            st.download_button("⬇️ Export stint data (CSV)",
                               st_filt.to_csv(index=False),
                               file_name=f"f1_stints_{session_key}.csv", mime="text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — TEAM RADIO
# ══════════════════════════════════════════════════════════════════════════════
with tab7:
    st.markdown('<div class="section-header">Team Radio Browser</div>', unsafe_allow_html=True)
    st.caption("Audio clips from selected drivers. Click ▶ to play in browser.")

    radio_driver = st.selectbox("Driver", selected_drivers["name_acronym"].tolist(), key="radio_drv")
    radio_drv_row = selected_drivers[selected_drivers["name_acronym"]==radio_driver].iloc[0]
    radio_drv_num = int(radio_drv_row["driver_number"])
    radio_color   = drv_color_str(radio_drv_row)

    radio_df = get_team_radio(session_key, radio_drv_num)

    if radio_df.empty:
        st.info(f"No team radio clips found for {radio_driver} in this session.")
    else:
        radio_df["date"] = pd.to_datetime(radio_df["date"], utc=True)
        radio_df = radio_df.sort_values("date").reset_index(drop=True)
        st.markdown(f"**{len(radio_df)} clips found for {radio_driver}**")

        # Try to match radio clip to lap number
        if "date_start" in laps_df.columns:
            drv_laps_radio = laps_df[laps_df["driver_number"]==radio_drv_num].copy()
            drv_laps_radio["date_start"] = pd.to_datetime(drv_laps_radio["date_start"], utc=True)
            drv_laps_radio = drv_laps_radio.sort_values("date_start")

        for _, clip in radio_df.iterrows():
            # Find approximate lap
            lap_label = ""
            if "date_start" in laps_df.columns and not drv_laps_radio.empty:
                past = drv_laps_radio[drv_laps_radio["date_start"] <= clip["date"]]
                if not past.empty:
                    lap_label = f"  ·  Lap {int(past.iloc[-1]['lap_number'])}"

            clip_time = clip["date"].strftime("%H:%M:%S UTC")
            url = clip.get("recording_url","")

            with st.container():
                st.markdown(f"""
                <div style='background:#16161e;border:1px solid #2a2a3d;border-left:3px solid {radio_color};
                            border-radius:6px;padding:10px 14px;margin-bottom:8px;'>
                  <span style='color:{radio_color};font-weight:700;font-size:13px;'>{radio_driver}</span>
                  <span style='color:#555;font-size:11px;margin-left:8px;'>{clip_time}{lap_label}</span>
                </div>
                """, unsafe_allow_html=True)
                if url:
                    st.audio(url, format="audio/mp3")
                else:
                    st.caption("_(no audio URL)_")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 — LAP TABLE
# ══════════════════════════════════════════════════════════════════════════════
with tab8:
    st.markdown('<div class="section-header">Full Lap Data</div>', unsafe_allow_html=True)
    display_cols = ["acronym","lap_number","lap_duration","duration_sector_1",
                    "duration_sector_2","duration_sector_3","compound"]
    avail_cols = [c for c in display_cols if c in laps_df.columns]
    table_df   = laps_df[avail_cols].copy().rename(columns={
        "acronym":"Driver","lap_number":"Lap","lap_duration":"Lap Time",
        "duration_sector_1":"S1","duration_sector_2":"S2",
        "duration_sector_3":"S3","compound":"Tyre",
    })
    for tc in ["Lap Time","S1","S2","S3"]:
        if tc in table_df.columns:
            table_df[tc] = table_df[tc].apply(fmt_time)

    fc, sc = st.columns(2)
    with fc:
        fd = st.multiselect("Filter driver", table_df["Driver"].unique().tolist(),
                            default=table_df["Driver"].unique().tolist())
    with sc:
        st.selectbox("Sort by", ["Lap","Lap Time","S1","S2","S3"], index=0, key="sort_by")

    table_df = table_df[table_df["Driver"].isin(fd)]
    st.dataframe(table_df, use_container_width=True, height=500, hide_index=True)
    st.download_button("⬇️ Export table (CSV)", table_df.to_csv(index=False),
                       file_name=f"f1_table_{session_key}.csv", mime="text/csv")

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;color:#333;font-size:11px;letter-spacing:2px;padding:20px 0 10px 0;'>
  DATA FROM OPENF1.ORG · NOT AFFILIATED WITH FORMULA ONE · FOR PERSONAL USE
</div>
""", unsafe_allow_html=True)
