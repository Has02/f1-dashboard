"""
telemetry.py  –  Track map + throttle/brake overlay helpers for the F1 dashboard.

Key design notes:
  - /location gives x,y,z in an arbitrary Cartesian system at ~3.7 Hz
  - /car_data gives throttle,brake,speed,rpm,n_gear,drs at ~3.7 Hz
  - Both are time-series; we scope to a lap using date_start + lap_duration
  - The track outline is drawn from the driver's own fastest-lap location data
  - "Fastest driver" overlay: for each track point, colour by whoever had the
    highest speed at the nearest matching xy position
"""

import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Optional, Dict, Tuple

BASE_URL = "https://api.openf1.org/v1"


# ─────────────────────────────────────────────────────────────────────────────
# LOW-LEVEL FETCH
# ─────────────────────────────────────────────────────────────────────────────

def _get(endpoint, **params):
    try:
        r = requests.get(f"{BASE_URL}/{endpoint}", params=params, timeout=25)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


def _hex_to_rgba(hex_color, alpha=0.25):
    h = hex_color.lstrip("#")
    if len(h) == 6:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"
    return f"rgba(200,200,200,{alpha})"


# ─────────────────────────────────────────────────────────────────────────────
# LAP TIME WINDOW
# ─────────────────────────────────────────────────────────────────────────────

def _lap_window(laps_df, driver_number, lap_number):
    rows = laps_df[
        (laps_df["driver_number"] == driver_number) &
        (laps_df["lap_number"] == lap_number)
    ]
    if rows.empty:
        return None, None
    row = rows.iloc[0]
    t_start = pd.to_datetime(row["date_start"], format="ISO8601", utc=True)
    dur = float(row.get("lap_duration") or 120)
    t_end = t_start + pd.Timedelta(seconds=dur + 2)
    return t_start.isoformat(), t_end.isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# FETCH ONE LAP'S TELEMETRY
# ─────────────────────────────────────────────────────────────────────────────

def fetch_lap_telemetry(session_key, driver_number, lap_number, laps_df):
    """Returns merged DataFrame: x,y,z,throttle,brake,speed,rpm,n_gear,drs,date"""
    t0, t1 = _lap_window(laps_df, driver_number, lap_number)
    if t0 is None:
        return pd.DataFrame()

    loc_raw = _get("location", session_key=session_key,
                   driver_number=driver_number, **{"date>": t0, "date<": t1})
    if not loc_raw:
        return pd.DataFrame()

    loc_df = pd.DataFrame(loc_raw)
    loc_df["date"] = pd.to_datetime(loc_df["date"], format="ISO8601", utc=True)
    loc_df = loc_df.sort_values("date").reset_index(drop=True)

    car_raw = _get("car_data", session_key=session_key,
                   driver_number=driver_number, **{"date>": t0, "date<": t1})
    if not car_raw:
        return loc_df[["date", "x", "y", "z"]]

    car_df = pd.DataFrame(car_raw)
    car_df["date"] = pd.to_datetime(car_df["date"], format="ISO8601", utc=True)
    car_df = car_df.sort_values("date").reset_index(drop=True)

    merged = pd.merge_asof(
        loc_df.sort_values("date"),
        car_df[["date", "throttle", "brake", "speed", "rpm", "n_gear", "drs"]].sort_values("date"),
        on="date", direction="nearest", tolerance=pd.Timedelta("500ms"),
    )
    for col in ["throttle", "brake", "speed", "rpm", "n_gear", "drs"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    return merged.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# TRACK OUTLINE FROM A SINGLE CLEAN LAP
# ─────────────────────────────────────────────────────────────────────────────

def extract_track_outline(session_key, driver_number, laps_df):
    """
    Fetch location data for the driver's best lap and return
    smoothed (x, y) np arrays representing the track centerline.
    """
    drv_laps = laps_df[laps_df["driver_number"] == driver_number].copy()
    drv_laps = drv_laps[drv_laps["lap_duration"].notna() & (drv_laps["lap_duration"] > 0)]
    if drv_laps.empty:
        return np.array([]), np.array([])

    best_lap = int(drv_laps.loc[drv_laps["lap_duration"].idxmin(), "lap_number"])
    t0, t1 = _lap_window(laps_df, driver_number, best_lap)
    if t0 is None:
        return np.array([]), np.array([])

    raw = _get("location", session_key=session_key,
               driver_number=driver_number, **{"date>": t0, "date<": t1})
    if not raw:
        return np.array([]), np.array([])

    df = pd.DataFrame(raw).sort_values("date")
    x = pd.to_numeric(df["x"], errors="coerce").values.astype(float)
    y = pd.to_numeric(df["y"], errors="coerce").values.astype(float)

    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 5:
        return np.array([]), np.array([])

    win = min(9, len(x))
    x = pd.Series(x).rolling(win, min_periods=1, center=True).mean().values
    y = pd.Series(y).rolling(win, min_periods=1, center=True).mean().values
    return x, y


# ─────────────────────────────────────────────────────────────────────────────
# FASTEST-DRIVER MAP
# ─────────────────────────────────────────────────────────────────────────────

def build_fastest_driver_map(telemetry_dict, driver_colors, track_x, track_y):
    """
    Colour each segment of the track by whichever driver had the highest
    speed at that location across the chosen lap.

    telemetry_dict: { acronym: DataFrame with x, y, speed columns }
    driver_colors:  { acronym: "#RRGGBB" }
    track_x, track_y: reference centerline arrays
    """
    fig = go.Figure()

    no_track = track_x is None or len(track_x) == 0

    if no_track:
        fig.add_annotation(
            text="⚠️ No track outline — check session / driver selection",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(color="#888", size=14),
        )
        fig.update_layout(**_map_layout())
        return fig

    # ── Thick grey background ──────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=track_x, y=track_y,
        mode="lines",
        line=dict(color="#252530", width=18),
        hoverinfo="skip", showlegend=False,
    ))
    # Thin white center line
    fig.add_trace(go.Scatter(
        x=track_x, y=track_y,
        mode="lines",
        line=dict(color="#3a3a4a", width=2),
        hoverinfo="skip", showlegend=False,
    ))

    if not telemetry_dict:
        fig.update_layout(**_map_layout())
        return fig

    n = len(track_x)

    # Build arrays: best_speed[i], best_driver[i]
    best_speed  = np.full(n, -1.0)
    best_driver = np.full(n, "", dtype=object)

    for acronym, tel_df in telemetry_dict.items():
        if tel_df.empty or "x" not in tel_df.columns:
            continue
        dx  = pd.to_numeric(tel_df["x"],     errors="coerce").values
        dy  = pd.to_numeric(tel_df["y"],     errors="coerce").values
        spd = pd.to_numeric(tel_df.get("speed", pd.Series(dtype=float)),
                            errors="coerce").fillna(0).values
        valid = ~(np.isnan(dx) | np.isnan(dy))
        dx, dy, spd = dx[valid], dy[valid], spd[valid]
        if len(dx) == 0:
            continue

        # Vectorised nearest-neighbour for all track points at once
        # Shape: (n_track, n_tel)
        diff_x = track_x[:, None] - dx[None, :]   # (n, m)
        diff_y = track_y[:, None] - dy[None, :]
        dists2  = diff_x ** 2 + diff_y ** 2        # (n, m)
        nn_idx  = np.argmin(dists2, axis=1)         # (n,)
        drv_spd = spd[nn_idx]                       # speed at nearest point

        mask = drv_spd > best_speed
        best_speed[mask]  = drv_spd[mask]
        best_driver[mask] = acronym

    # ── Draw coloured segments by contiguous driver runs ──────────────────
    added_to_legend = set()
    i = 0
    while i < n:
        drv = best_driver[i]
        if drv == "":
            i += 1
            continue

        # Find end of contiguous run
        j = i + 1
        while j < n and best_driver[j] == drv:
            j += 1

        color = driver_colors.get(drv, "#ffffff")
        seg_x = np.append(track_x[i:j], track_x[j] if j < n else track_x[j - 1])
        seg_y = np.append(track_y[i:j], track_y[j] if j < n else track_y[j - 1])
        avg_spd = float(np.mean(best_speed[i:j]))

        show_leg = drv not in added_to_legend
        if show_leg:
            added_to_legend.add(drv)

        fig.add_trace(go.Scatter(
            x=seg_x, y=seg_y,
            mode="lines",
            line=dict(color=color, width=5),
            name=drv,
            legendgroup=drv,
            showlegend=show_leg,
            hovertemplate=(
                f"<b>{drv}</b> fastest<br>"
                f"Avg speed: {avg_spd:.0f} km/h<extra></extra>"
            ),
        ))
        i = j

    # S/F marker
    fig.add_trace(go.Scatter(
        x=[track_x[0]], y=[track_y[0]],
        mode="markers+text",
        marker=dict(symbol="diamond", size=13, color="#ffffff",
                    line=dict(color="#000", width=1)),
        text=["S/F"], textposition="top center",
        textfont=dict(color="#ffffff", size=10),
        hoverinfo="skip", showlegend=False,
    ))

    fig.update_layout(**_map_layout())
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-DRIVER CHANNEL MAP
# ─────────────────────────────────────────────────────────────────────────────

_CHANNEL_META = {
    "throttle": {
        "label": "Throttle %",
        "colorscale": [[0.0, "#1a1a1a"], [0.3, "#1a6600"], [0.7, "#33cc00"], [1.0, "#00ff44"]],
        "range": (0, 100), "unit": "%",
    },
    "brake": {
        "label": "Brake",
        "colorscale": [[0.0, "#1a1a1a"], [0.01, "#1a1a1a"], [0.05, "#ff4400"], [1.0, "#ff0000"]],
        "range": (0, 100), "unit": "",
    },
    "speed": {
        "label": "Speed", "colorscale": "Viridis",
        "range": None, "unit": " km/h",
    },
    "n_gear": {
        "label": "Gear", "colorscale": "Plasma",
        "range": (1, 8), "unit": "",
    },
}


def make_channel_map(telemetry_df, channel, driver_acronym, team_color, track_x, track_y):
    """Single-driver map coloured by throttle/brake/speed/gear."""
    fig = go.Figure()

    if track_x is not None and len(track_x) > 0:
        fig.add_trace(go.Scatter(
            x=track_x, y=track_y,
            mode="lines",
            line=dict(color="#252530", width=18),
            hoverinfo="skip", showlegend=False,
        ))

    if telemetry_df.empty or "x" not in telemetry_df.columns:
        fig.add_annotation(
            text="No telemetry data for this lap",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(color="#888", size=14),
        )
        fig.update_layout(**_map_layout())
        return fig

    meta = _CHANNEL_META.get(channel, {"label": channel, "colorscale": "Viridis",
                                        "range": None, "unit": ""})
    x   = pd.to_numeric(telemetry_df["x"], errors="coerce").values
    y   = pd.to_numeric(telemetry_df["y"], errors="coerce").values
    val = telemetry_df[channel].fillna(0).values if channel in telemetry_df.columns \
          else np.zeros(len(x))

    valid = ~(np.isnan(x) | np.isnan(y))
    x, y, val = x[valid], y[valid], val[valid]

    cmin, cmax = meta["range"] if meta["range"] else (float(np.nanmin(val)), float(np.nanmax(val)))

    spd_col  = telemetry_df["speed"].fillna(0).values[valid]  if "speed"  in telemetry_df.columns else np.zeros(len(x))
    gear_col = telemetry_df["n_gear"].fillna(0).values[valid] if "n_gear" in telemetry_df.columns else np.zeros(len(x))

    hover = [
        f"<b>{driver_acronym}</b><br>"
        f"{meta['label']}: {val[i]:.0f}{meta['unit']}<br>"
        f"Speed: {spd_col[i]:.0f} km/h  Gear: {int(gear_col[i])}"
        for i in range(len(x))
    ]

    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(
            color=val, colorscale=meta["colorscale"],
            cmin=cmin, cmax=cmax, size=5,
            colorbar=dict(
                title=dict(text=meta["label"], side="right",
                           font=dict(color="#aaa", size=11)),
                thickness=12, len=0.7,
                tickfont=dict(color="#aaa", size=10),
            ),
            line=dict(width=0),
        ),
        text=hover,
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    ))

    if len(x):
        fig.add_trace(go.Scatter(
            x=[x[0]], y=[y[0]],
            mode="markers+text",
            marker=dict(symbol="diamond", size=12, color=team_color,
                        line=dict(color="#fff", width=2)),
            text=["S/F"], textposition="top center",
            textfont=dict(color="#fff", size=10),
            hoverinfo="skip", showlegend=False,
        ))

    fig.update_layout(**_map_layout())
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# TELEMETRY TRACES (subplots)
# ─────────────────────────────────────────────────────────────────────────────

def make_telemetry_traces(telemetry_dict, driver_colors):
    from plotly.subplots import make_subplots

    # Detect if we have real distance data (FastF1) or sample index (OpenF1)
    use_distance = any(
        "distance" in df.columns and df["distance"].notna().any()
        for df in telemetry_dict.values() if not df.empty
    )
    sources = set(
        df["source"].iloc[0] for df in telemetry_dict.values()
        if not df.empty and "source" in df.columns
    )
    x_label = "Distance (m)" if use_distance else "Sample index (≈ distance proxy)"
    x_label += "  ·  FastF1" if "fastf1" in sources else "  ·  OpenF1"

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        subplot_titles=["Throttle (%)", "Brake", "Speed (km/h)", "Gear"],
        row_heights=[0.32, 0.18, 0.32, 0.18],
    )
    channels = ["throttle", "brake", "speed", "n_gear"]
    row_map  = {c: i + 1 for i, c in enumerate(channels)}

    for acronym, tel_df in telemetry_dict.items():
        if tel_df.empty:
            continue
        color      = driver_colors.get(acronym, "#ffffff")
        fill_color = _hex_to_rgba(color, 0.25)

        # X axis: real distance if available, else sample index
        if use_distance and "distance" in tel_df.columns:
            x_vals = tel_df["distance"].fillna(method="ffill").values
        else:
            x_vals = np.arange(len(tel_df))

        for ch in channels:
            if ch not in tel_df.columns:
                continue
            vals     = tel_df[ch].fillna(0).values
            row      = row_map[ch]
            show_leg = (ch == "throttle")
            ht = f"<b>{acronym}</b> %{{x:.0f}}{'m' if use_distance else ''} → {ch}: %{{y:.1f}}<extra></extra>"

            if ch == "brake":
                fig.add_trace(go.Scatter(
                    x=x_vals, y=vals, mode="lines",
                    line=dict(color=color, width=1.5),
                    fill="tozeroy", fillcolor=fill_color,
                    name=acronym, legendgroup=acronym, showlegend=show_leg,
                    hovertemplate=ht,
                ), row=row, col=1)
            else:
                fig.add_trace(go.Scatter(
                    x=x_vals, y=vals, mode="lines",
                    line=dict(color=color, width=1.8),
                    name=acronym, legendgroup=acronym, showlegend=show_leg,
                    hovertemplate=ht,
                ), row=row, col=1)

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0d0d0f", plot_bgcolor="#13131a",
        font=dict(family="Exo 2, sans-serif", color="#ccc", size=11),
        height=520, margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        legend=dict(bgcolor="#0d0d0f", bordercolor="#333", borderwidth=1),
    )
    for i in range(1, 5):
        fig.update_xaxes(
            gridcolor="#1e1e2a", zeroline=False,
            showspikes=True, spikemode="across", spikesnap="cursor",
            spikecolor="#aaaaaa", spikethickness=1, spikedash="solid",
            row=i, col=1,
        )
        fig.update_yaxes(gridcolor="#1e1e2a", zeroline=False, row=i, col=1)
    fig.update_xaxes(title_text=x_label, row=4, col=1)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SHARED MAP LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

def _map_layout():
    return dict(
        template="plotly_dark",
        paper_bgcolor="#0d0d0f",
        plot_bgcolor="#0d0d0f",
        font=dict(family="Exo 2, sans-serif", color="#ccc"),
        xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False),
        height=560,
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(
            bgcolor="rgba(13,13,15,0.85)",
            bordercolor="#333", borderwidth=1,
            font=dict(size=13),
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# COMBINED MAP + TRACES FIGURE (hover-linked)
# ─────────────────────────────────────────────────────────────────────────────

def make_linked_telemetry_figure(
    telemetry_dict,
    driver_colors,
    track_x,
    track_y,
    map_mode="channel",    # "channel" | "fastest"
    channel="speed",       # used only when map_mode=="channel"
    single_driver=None,    # used only when map_mode=="channel"
):
    """
    Single Plotly figure with:
      Left column  (col 1): track map  — markers carry customdata=[sample_index]
      Right column (col 2): 4 subplots — throttle / brake / speed / gear
                            x-axis = sample index, shared across all 4 rows

    Hovering a map point fires a vertical spike on the matching sample index
    in all four telemetry traces via Plotly's spike/crosshair mechanism.
    """
    from plotly.subplots import make_subplots

    # ── Build subplot grid: 4 rows × 2 cols, map spans all 4 rows on left ─
    fig = make_subplots(
        rows=4, cols=2,
        column_widths=[0.42, 0.58],
        shared_xaxes=False,
        vertical_spacing=0.04,
        horizontal_spacing=0.06,
        specs=[
            [{"rowspan": 4, "type": "xy"}, {"type": "xy"}],
            [None,                          {"type": "xy"}],
            [None,                          {"type": "xy"}],
            [None,                          {"type": "xy"}],
        ],
        subplot_titles=["", "Throttle (%)", "", "Brake", "", "Speed (km/h)", "", "Gear"],
    )

    # ── Track background (col 1) ───────────────────────────────────────────
    if track_x is not None and len(track_x) > 0:
        fig.add_trace(go.Scatter(
            x=track_x, y=track_y, mode="lines",
            line=dict(color="#252530", width=18),
            hoverinfo="skip", showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=track_x, y=track_y, mode="lines",
            line=dict(color="#3a3a4a", width=2),
            hoverinfo="skip", showlegend=False,
        ), row=1, col=1)

    # ── Map traces with customdata = sample index ─────────────────────────
    if map_mode == "fastest" and len(telemetry_dict) > 1:
        _add_fastest_map_traces(fig, telemetry_dict, driver_colors, track_x, track_y)
    else:
        # Single driver channel map
        drv_name = single_driver or (list(telemetry_dict.keys())[0] if telemetry_dict else None)
        if drv_name and drv_name in telemetry_dict:
            tel = telemetry_dict[drv_name]
            color = driver_colors.get(drv_name, "#ffffff")
            _add_channel_map_trace(fig, tel, channel, drv_name, color)

    # S/F marker
    if track_x is not None and len(track_x) > 0:
        fig.add_trace(go.Scatter(
            x=[track_x[0]], y=[track_y[0]],
            mode="markers+text",
            marker=dict(symbol="diamond", size=12, color="#ffffff",
                        line=dict(color="#000", width=1)),
            text=["S/F"], textposition="top center",
            textfont=dict(color="#fff", size=10),
            hoverinfo="skip", showlegend=False,
        ), row=1, col=1)

    # ── Telemetry traces (col 2, rows 1-4) ────────────────────────────────
    channels  = ["throttle", "brake", "speed", "n_gear"]
    row_map   = {c: i + 1 for i, c in enumerate(channels)}
    trace_row_labels = {1: "Throttle (%)", 2: "Brake", 3: "Speed (km/h)", 4: "Gear"}

    for acronym, tel_df in telemetry_dict.items():
        if tel_df.empty:
            continue
        color      = driver_colors.get(acronym, "#ffffff")
        fill_color = _hex_to_rgba(color, 0.25)
        x_idx      = list(range(len(tel_df)))

        for ch in channels:
            if ch not in tel_df.columns:
                continue
            vals     = tel_df[ch].fillna(0).values
            row      = row_map[ch]
            show_leg = (ch == "throttle")

            # Attach full telemetry as customdata so hover shows all channels
            throttle_v = tel_df["throttle"].fillna(0).values if "throttle" in tel_df.columns else np.zeros(len(x_idx))
            brake_v    = tel_df["brake"].fillna(0).values    if "brake"    in tel_df.columns else np.zeros(len(x_idx))
            speed_v    = tel_df["speed"].fillna(0).values    if "speed"    in tel_df.columns else np.zeros(len(x_idx))
            gear_v     = tel_df["n_gear"].fillna(0).values   if "n_gear"   in tel_df.columns else np.zeros(len(x_idx))
            cdata      = np.stack([throttle_v, brake_v, speed_v, gear_v], axis=1)

            ht = (
                f"<b>{acronym}</b> · sample %{{x}}<br>"
                f"Throttle: %{{customdata[0]:.0f}}%<br>"
                f"Brake: %{{customdata[1]:.0f}}<br>"
                f"Speed: %{{customdata[2]:.0f}} km/h<br>"
                f"Gear: %{{customdata[3]:.0f}}<extra></extra>"
            )

            if ch == "brake":
                fig.add_trace(go.Scatter(
                    x=x_idx, y=vals, mode="lines",
                    line=dict(color=color, width=1.5),
                    fill="tozeroy", fillcolor=fill_color,
                    customdata=cdata, hovertemplate=ht,
                    name=acronym, legendgroup=acronym, showlegend=show_leg,
                ), row=row, col=2)
            else:
                fig.add_trace(go.Scatter(
                    x=x_idx, y=vals, mode="lines",
                    line=dict(color=color, width=1.8),
                    customdata=cdata, hovertemplate=ht,
                    name=acronym, legendgroup=acronym, showlegend=show_leg,
                ), row=row, col=2)

    # ── Shared x-axis spikes on col 2 (the crosshair that links all rows) ─
    for row in range(1, 5):
        fig.update_xaxes(
            showspikes=True, spikemode="across", spikesnap="cursor",
            spikecolor="#888", spikethickness=1, spikedash="dot",
            gridcolor="#1e1e2a", zeroline=False,
            row=row, col=2,
        )
        fig.update_yaxes(
            gridcolor="#1e1e2a", zeroline=False,
            showspikes=True, spikecolor="#555", spikethickness=1,
            row=row, col=2,
        )

    fig.update_xaxes(title_text="Sample index", row=4, col=2)
    fig.update_xaxes(visible=False, scaleanchor="y", scaleratio=1, row=1, col=1)
    fig.update_yaxes(visible=False, row=1, col=1)

    # Remove the blank subplot_title strings that appear for col 1 rows
    for ann in fig.layout.annotations:
        if ann.text == "":
            ann.visible = False

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d0d0f",
        plot_bgcolor="#0d0d0f",
        font=dict(family="Exo 2, sans-serif", color="#ccc", size=11),
        height=620,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        legend=dict(
            bgcolor="rgba(13,13,15,0.85)",
            bordercolor="#333", borderwidth=1,
            font=dict(size=12),
            x=1.01, y=1, xanchor="left",
        ),
    )

    # Override plot_bgcolor for map subplot to match dark background
    fig.update_layout(**{
        "xaxis":  dict(visible=False, scaleanchor="y", scaleratio=1),
        "yaxis":  dict(visible=False),
    })

    return fig


def _add_channel_map_trace(fig, tel_df, channel, driver_acronym, team_color):
    """Add colour-coded map markers to row=1, col=1 with sample index as customdata."""
    if tel_df.empty or "x" not in tel_df.columns:
        return

    meta = _CHANNEL_META.get(channel, {"label": channel, "colorscale": "Viridis",
                                        "range": None, "unit": ""})
    x   = pd.to_numeric(tel_df["x"], errors="coerce").values
    y   = pd.to_numeric(tel_df["y"], errors="coerce").values
    val = tel_df[channel].fillna(0).values if channel in tel_df.columns else np.zeros(len(x))

    valid = ~(np.isnan(x) | np.isnan(y))
    x, y, val = x[valid], y[valid], val[valid]
    sample_idx = np.where(valid)[0]   # original indices → used as crosshair x

    throttle_v = tel_df["throttle"].fillna(0).values[valid] if "throttle" in tel_df.columns else np.zeros(len(x))
    brake_v    = tel_df["brake"].fillna(0).values[valid]    if "brake"    in tel_df.columns else np.zeros(len(x))
    speed_v    = tel_df["speed"].fillna(0).values[valid]    if "speed"    in tel_df.columns else np.zeros(len(x))
    gear_v     = tel_df["n_gear"].fillna(0).values[valid]   if "n_gear"   in tel_df.columns else np.zeros(len(x))

    # customdata columns: [sample_idx, throttle, brake, speed, gear, channel_val]
    cdata = np.stack([sample_idx, throttle_v, brake_v, speed_v, gear_v, val], axis=1)
    cmin, cmax = meta["range"] if meta["range"] else (float(np.nanmin(val)), float(np.nanmax(val)))

    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(
            color=val, colorscale=meta["colorscale"],
            cmin=cmin, cmax=cmax, size=5,
            colorbar=dict(
                title=dict(text=meta["label"], side="right",
                           font=dict(color="#aaa", size=10)),
                thickness=10, len=0.5, y=0.5,
                tickfont=dict(color="#aaa", size=9),
            ),
            line=dict(width=0),
        ),
        customdata=cdata,
        hovertemplate=(
            f"<b>{driver_acronym}</b> · sample %{{customdata[0]:.0f}}<br>"
            f"Throttle: %{{customdata[1]:.0f}}%<br>"
            f"Brake: %{{customdata[2]:.0f}}<br>"
            f"Speed: %{{customdata[3]:.0f}} km/h<br>"
            f"Gear: %{{customdata[4]:.0f}}<br>"
            f"{meta['label']}: %{{customdata[5]:.0f}}{meta['unit']}"
            "<extra></extra>"
        ),
        showlegend=False,
    ), row=1, col=1)


def _add_fastest_map_traces(fig, telemetry_dict, driver_colors, track_x, track_y):
    """Add fastest-driver coloured segments to row=1, col=1."""
    if track_x is None or len(track_x) == 0:
        return
    n = len(track_x)
    best_speed  = np.full(n, -1.0)
    best_driver = np.full(n, "", dtype=object)
    best_idx    = np.full(n, -1, dtype=int)   # nearest sample index per track point

    for acronym, tel_df in telemetry_dict.items():
        if tel_df.empty or "x" not in tel_df.columns:
            continue
        dx  = pd.to_numeric(tel_df["x"], errors="coerce").values
        dy  = pd.to_numeric(tel_df["y"], errors="coerce").values
        spd = pd.to_numeric(tel_df.get("speed", pd.Series(dtype=float)),
                            errors="coerce").fillna(0).values
        valid = ~(np.isnan(dx) | np.isnan(dy))
        dx, dy, spd = dx[valid], dy[valid], spd[valid]
        orig_idx = np.where(valid)[0]
        if len(dx) == 0:
            continue

        diff_x = track_x[:, None] - dx[None, :]
        diff_y = track_y[:, None] - dy[None, :]
        dists2 = diff_x**2 + diff_y**2
        nn     = np.argmin(dists2, axis=1)
        drv_spd = spd[nn]
        mask = drv_spd > best_speed
        best_speed[mask]  = drv_spd[mask]
        best_driver[mask] = acronym
        best_idx[mask]    = orig_idx[nn[mask]]

    added_to_legend = set()
    i = 0
    while i < n:
        drv = best_driver[i]
        if drv == "" or drv not in telemetry_dict:
            i += 1
            continue
        j = i + 1
        while j < n and best_driver[j] == drv:
            j += 1

        color   = driver_colors.get(drv, "#ffffff")
        tel_df  = telemetry_dict[drv]
        seg_x   = np.append(track_x[i:j], track_x[j] if j < n else track_x[j-1])
        seg_y   = np.append(track_y[i:j], track_y[j] if j < n else track_y[j-1])
        avg_spd = float(np.mean(best_speed[i:j]))
        seg_idx = best_idx[i]   # representative sample index for this segment

        # Pull telemetry at seg_idx for hover
        def _safe(col):
            if col in tel_df.columns and seg_idx < len(tel_df):
                v = tel_df[col].iloc[seg_idx]
                return float(v) if pd.notna(v) else 0.0
            return 0.0

        cdata_seg = [[seg_idx, _safe("throttle"), _safe("brake"), avg_spd, _safe("n_gear")]] * len(seg_x)

        show_leg = drv not in added_to_legend
        if show_leg:
            added_to_legend.add(drv)

        fig.add_trace(go.Scatter(
            x=seg_x, y=seg_y, mode="lines",
            line=dict(color=color, width=5),
            name=drv, legendgroup=drv, showlegend=show_leg,
            customdata=cdata_seg,
            hovertemplate=(
                f"<b>{drv}</b> fastest · sample %{{customdata[0]:.0f}}<br>"
                f"Throttle: %{{customdata[1]:.0f}}%<br>"
                f"Brake: %{{customdata[2]:.0f}}<br>"
                f"Speed: %{{customdata[3]:.0f}} km/h<br>"
                f"Gear: %{{customdata[4]:.0f}}<extra></extra>"
            ),
        ), row=1, col=1)
        i = j
