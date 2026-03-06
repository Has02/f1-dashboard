"""
data_layer.py  —  Unified telemetry + corner analysis layer.

Sources:
  FastF1  — distance-aligned 240Hz, clean GPS, Q1/Q2/Q3 best lap support
  OpenF1  — fallback for telemetry; primary for radio/position/stints

Unified output columns:
  distance    m from lap start   speed  km/h     throttle  0-100%
  brake       0-100%             n_gear 1-8       rpm       RPM
  x, y        Cartesian coords   drs    0/8/10/12/14
  source      "fastf1"|"openf1"
"""

import os, warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_URL = "https://api.openf1.org/v1"

# ── FastF1 availability ───────────────────────────────────────────────────────
try:
    import fastf1
    import fastf1.utils
    _FF1_AVAILABLE = True
    _FF1_CACHE = os.path.join(os.path.dirname(__file__), ".fastf1_cache")
    os.makedirs(_FF1_CACHE, exist_ok=True)
    fastf1.Cache.enable_cache(_FF1_CACHE)
except ImportError:
    _FF1_AVAILABLE = False

def fastf1_available() -> bool:
    return _FF1_AVAILABLE

_FF1_SESSION_MAP = {
    "Practice 1": "Practice 1", "Practice 2": "Practice 2",
    "Practice 3": "Practice 3", "Qualifying": "Qualifying",
    "Sprint": "Sprint", "Sprint Qualifying": "Sprint Qualifying",
    "Race": "Race",
}

_session_cache = {}

def _load_ff1_session(year, gp, session_name):
    if not _FF1_AVAILABLE:
        return None
    key = (year, gp, session_name)
    if key in _session_cache:
        return _session_cache[key]
    try:
        ff1_name = _FF1_SESSION_MAP.get(session_name, session_name)
        sess = fastf1.get_session(year, gp, ff1_name)
        sess.load(telemetry=True, weather=False, messages=False, laps=True)
        _session_cache[key] = sess
        return sess
    except Exception as e:
        print(f"[FastF1] session load failed: {e}")
        return None

def _lap_to_telemetry(lap) -> pd.DataFrame:
    try:
        tel = lap.get_telemetry().add_distance()
        df = pd.DataFrame({
            "distance": tel["Distance"].values.astype(float),
            "x":        tel["X"].values.astype(float),
            "y":        tel["Y"].values.astype(float),
            "speed":    tel["Speed"].values.astype(float),
            "throttle": tel["Throttle"].values.astype(float),
            "brake":    tel["Brake"].values.astype(float) * 100.0,
            "n_gear":   tel["nGear"].values.astype(float),
            "rpm":      tel["RPM"].values.astype(float),
            "drs":      tel["DRS"].values.astype(float),
            "source":   "fastf1",
        })
        return df.dropna(subset=["distance","x","y"]).reset_index(drop=True)
    except Exception as e:
        print(f"[FastF1] _lap_to_telemetry failed: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# QUALIFYING SEGMENT SUPPORT
# ─────────────────────────────────────────────────────────────────────────────

def get_qual_segments(year, gp) -> list:
    """Returns which Q segments are available, e.g. ['Q1','Q2','Q3']."""
    if not _FF1_AVAILABLE:
        return ["Best"]
    try:
        sess = _load_ff1_session(year, gp, "Qualifying")
        if sess is None:
            return ["Best"]
        segments = [q for q in ["Q1","Q2","Q3"]
                    if q in sess.laps.columns and sess.laps[q].notna().any()]
        return segments if segments else ["Best"]
    except Exception:
        return ["Best"]


def _best_lap_in_segment(sess, driver_acronym, segment):
    """Return the fastest FastF1 Lap in the given Q segment for a driver."""
    drv_laps = sess.laps.pick_drivers(driver_acronym)
    drv_laps = drv_laps[drv_laps["LapTime"].notna()]
    if drv_laps.empty:
        return None
    if segment != "Best" and segment in drv_laps.columns:
        seg_laps = drv_laps[drv_laps[segment].notna()]
        if not seg_laps.empty:
            return seg_laps.loc[seg_laps["LapTime"].idxmin()]
    # Fallback: fastest overall
    return drv_laps.loc[drv_laps["LapTime"].idxmin()]


# ─────────────────────────────────────────────────────────────────────────────
# TRACK OUTLINE
# ─────────────────────────────────────────────────────────────────────────────

def get_track_outline(year, gp, session_name,
                      session_key, driver_number, laps_df):
    """Returns (x, y, source_str)."""
    if _FF1_AVAILABLE:
        try:
            sess = _load_ff1_session(year, gp, session_name)
            if sess is not None:
                tel = _lap_to_telemetry(sess.laps.pick_fastest())
                if not tel.empty:
                    x = pd.Series(tel["x"].values).rolling(5, min_periods=1, center=True).mean().values
                    y = pd.Series(tel["y"].values).rolling(5, min_periods=1, center=True).mean().values
                    return x, y, "fastf1"
        except Exception as e:
            print(f"[FastF1] track outline failed: {e}")
    from telemetry import extract_track_outline
    x, y = extract_track_outline(session_key, driver_number, laps_df)
    return x, y, "openf1"


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE LAP TELEMETRY (by lap number)
# ─────────────────────────────────────────────────────────────────────────────

def _openf1_lap_tel(session_key, driver_number, lap_number, laps_df):
    from telemetry import fetch_lap_telemetry
    df = fetch_lap_telemetry(session_key, driver_number, lap_number, laps_df)
    if df.empty:
        return df
    if "x" in df.columns and "y" in df.columns:
        dx = np.diff(pd.to_numeric(df["x"], errors="coerce").fillna(0).values, prepend=0)
        dy = np.diff(pd.to_numeric(df["y"], errors="coerce").fillna(0).values, prepend=0)
        df["distance"] = np.cumsum(np.sqrt(dx**2 + dy**2))
    else:
        df["distance"] = np.arange(len(df), dtype=float)
    df["source"] = "openf1"
    return df.reset_index(drop=True)


def get_lap_telemetry(driver_acronym, lap_number,
                      year, gp, session_name,
                      session_key, driver_number, laps_df) -> pd.DataFrame:
    if _FF1_AVAILABLE:
        try:
            sess = _load_ff1_session(year, gp, session_name)
            if sess is not None:
                drv_laps = sess.laps.pick_drivers(driver_acronym)
                match = drv_laps[drv_laps["LapNumber"] == lap_number]
                if not match.empty:
                    return _lap_to_telemetry(match.iloc[0])
        except Exception as e:
            print(f"[FastF1] lap tel failed {driver_acronym} #{lap_number}: {e}")
    return _openf1_lap_tel(session_key, driver_number, lap_number, laps_df)


# ─────────────────────────────────────────────────────────────────────────────
# QUALIFYING BEST-LAP TELEMETRY
# ─────────────────────────────────────────────────────────────────────────────

def get_qual_best_telemetry(driver_acronym, segment,
                            year, gp,
                            session_key, driver_number, laps_df) -> tuple:
    """
    Returns (telemetry_df, lap_time_seconds, lap_number).
    Uses FastF1 best lap for the Q segment when available.
    """
    if _FF1_AVAILABLE:
        try:
            sess = _load_ff1_session(year, gp, "Qualifying")
            if sess is not None:
                lap = _best_lap_in_segment(sess, driver_acronym, segment)
                if lap is not None:
                    tel     = _lap_to_telemetry(lap)
                    lap_t   = lap["LapTime"].total_seconds() if pd.notna(lap["LapTime"]) else None
                    lap_num = int(lap["LapNumber"]) if pd.notna(lap["LapNumber"]) else None
                    return tel, lap_t, lap_num
        except Exception as e:
            print(f"[FastF1] qual best tel failed {driver_acronym} {segment}: {e}")

    # OpenF1 fallback — fastest lap in laps_df
    drv_laps = laps_df[laps_df["driver_number"] == driver_number].copy()
    drv_laps = drv_laps[drv_laps["lap_duration"].notna() & (drv_laps["lap_duration"] > 0)]
    if drv_laps.empty:
        return pd.DataFrame(), None, None
    best    = drv_laps.loc[drv_laps["lap_duration"].idxmin()]
    lap_num = int(best["lap_number"])
    tel     = _openf1_lap_tel(session_key, driver_number, lap_num, laps_df)
    return tel, float(best["lap_duration"]), lap_num


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-DRIVER TELEMETRY
# ─────────────────────────────────────────────────────────────────────────────

def get_multi_driver_telemetry(drivers, lap_number,
                               year, gp, session_name,
                               session_key, laps_df) -> dict:
    result = {}
    for drv in drivers:
        tel = get_lap_telemetry(
            drv["acronym"], lap_number,
            year, gp, session_name,
            session_key, drv["driver_number"], laps_df,
        )
        if not tel.empty:
            result[drv["acronym"]] = tel
    return result


# ─────────────────────────────────────────────────────────────────────────────
# DELTA TIME  (FastF1 preferred, speed-integral fallback)
# ─────────────────────────────────────────────────────────────────────────────

def get_delta_to_ref(ref_acronym, comp_acronym,
                     ref_tel: pd.DataFrame, comp_tel: pd.DataFrame) -> pd.DataFrame:
    """
    Returns DataFrame {distance, delta} where delta>0 means comp is behind ref.
    """
    if ref_tel.empty or comp_tel.empty:
        return pd.DataFrame()

    # FastF1 path
    if (_FF1_AVAILABLE
            and not ref_tel.empty and not comp_tel.empty
            and ref_tel.get("source", pd.Series(["openf1"])).iloc[0] == "fastf1"
            and comp_tel.get("source", pd.Series(["openf1"])).iloc[0] == "fastf1"):
        try:
            ref_ff = ref_tel.rename(columns={"distance":"Distance","speed":"Speed"})
            cmp_ff = comp_tel.rename(columns={"distance":"Distance","speed":"Speed"})
            delta, ref_out, _ = fastf1.utils.delta_time(ref_ff, cmp_ff)
            return pd.DataFrame({"distance": ref_out["Distance"].values, "delta": delta.values})
        except Exception as e:
            print(f"[FastF1] delta_time failed: {e}")

    # Speed-integral fallback
    try:
        def _cumtime(tel):
            d   = tel["distance"].values
            spd = np.maximum(tel["speed"].values, 1.0)
            dd  = np.diff(d, prepend=d[0])
            return d, np.cumsum(dd / (spd / 3.6))

        r_d, r_t = _cumtime(ref_tel)
        c_d, c_t = _cumtime(comp_tel)
        max_d    = min(r_d[-1], c_d[-1])
        common   = r_d[r_d <= max_d]
        r_tc     = r_t[r_d <= max_d]
        c_ti     = np.interp(common, c_d, c_t)
        return pd.DataFrame({"distance": common, "delta": c_ti - r_tc})
    except Exception as e:
        print(f"[data_layer] fallback delta failed: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# CORNER DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_corners(tel_df: pd.DataFrame,
                   min_corner_len_m: float = 50.0,
                   curvature_threshold: float = 0.0003,
                   merge_gap_m: float = 80.0) -> pd.DataFrame:
    """
    Detect corners via track curvature (dheading/ddistance).
    Returns DataFrame: corner_num, start_dist, end_dist, apex_dist,
    apex_speed, length_m, view_start, view_end.
    """
    if tel_df.empty or "x" not in tel_df.columns or "distance" not in tel_df.columns:
        return pd.DataFrame()

    x = tel_df["x"].values.astype(float)
    y = tel_df["y"].values.astype(float)
    d = tel_df["distance"].values.astype(float)
    s = tel_df["speed"].values.astype(float) if "speed" in tel_df.columns else np.full(len(x), 200.0)

    # Smooth to reduce GPS noise
    win = max(5, min(31, len(x) // 20 * 2 + 1))
    xs  = pd.Series(x).rolling(win, min_periods=1, center=True).mean().values
    ys  = pd.Series(y).rolling(win, min_periods=1, center=True).mean().values

    heading   = np.arctan2(np.gradient(ys), np.gradient(xs))
    h_unwrap  = np.unwrap(heading)
    dd        = np.where(np.abs(np.gradient(d)) < 1e-6, 1e-6, np.gradient(d))
    curvature = np.abs(np.gradient(h_unwrap) / dd)
    curvature = pd.Series(curvature).rolling(win, min_periods=1, center=True).mean().values

    in_corner = curvature > curvature_threshold

    # Extract contiguous corner segments
    raw_corners = []
    i = 0
    while i < len(d):
        if in_corner[i]:
            j = i + 1
            while j < len(d) and in_corner[j]:
                j += 1
            length = d[j-1] - d[i]
            if length >= min_corner_len_m:
                raw_corners.append({"start": int(i), "end": int(j-1)})
            i = j
        else:
            i += 1

    if not raw_corners:
        return pd.DataFrame()

    # Merge close corners (chicanes)
    merged = [raw_corners[0].copy()]
    for c in raw_corners[1:]:
        gap = d[c["start"]] - d[merged[-1]["end"]]
        if gap < merge_gap_m:
            merged[-1]["end"] = c["end"]
        else:
            merged.append(c.copy())

    # Build output DataFrame
    rows = []
    for ci, c in enumerate(merged):
        seg_d = d[c["start"]:c["end"]+1]
        seg_s = s[c["start"]:c["end"]+1]
        if len(seg_d) < 3:
            continue
        apex_i = int(np.argmin(seg_s))
        rows.append({
            "corner_num": ci + 1,
            "start_dist": float(seg_d[0]),
            "end_dist":   float(seg_d[-1]),
            "apex_dist":  float(seg_d[apex_i]),
            "apex_speed": float(seg_s[apex_i]),
            "length_m":   float(seg_d[-1] - seg_d[0]),
            "view_start": float(max(0.0, seg_d[0] - 120.0)),
            "view_end":   float(seg_d[-1] + 80.0),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# CORNER STATS  — per-driver per-corner summary table
# ─────────────────────────────────────────────────────────────────────────────

def corner_stats(tel_dict: dict, corners_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns tidy DataFrame:
      corner_num, driver, apex_speed, brake_dist_before_apex,
      throttle_dist_after_apex, time_in_corner_s
    """
    if corners_df.empty or not tel_dict:
        return pd.DataFrame()

    rows = []
    for _, corner in corners_df.iterrows():
        cnum = int(corner["corner_num"])
        v_s  = corner["view_start"]
        v_e  = corner["view_end"]

        for driver, tel in tel_dict.items():
            if tel.empty or "distance" not in tel.columns:
                continue
            d   = tel["distance"].values
            spd = tel["speed"].values   if "speed"    in tel.columns else np.zeros(len(d))
            brk = tel["brake"].values   if "brake"    in tel.columns else np.zeros(len(d))
            thr = tel["throttle"].values if "throttle" in tel.columns else np.full(len(d), 100.0)

            mask = (d >= v_s) & (d <= v_e)
            if mask.sum() < 3:
                continue

            seg_d   = d[mask]
            seg_spd = spd[mask]
            seg_brk = brk[mask]
            seg_thr = thr[mask]

            apex_i   = int(np.argmin(seg_spd))
            apex_spd = float(seg_spd[apex_i])
            apex_d   = float(seg_d[apex_i])

            # Braking: first brake>20% before apex
            pre = seg_d <= apex_d
            bp  = seg_d[pre & (seg_brk > 20)]
            brake_dist = float(apex_d - bp[0]) if len(bp) > 0 else 0.0

            # Throttle: first throttle>30% after apex
            post = seg_d > apex_d
            tp   = seg_d[post & (seg_thr > 30)]
            thr_dist = float(tp[0] - apex_d) if len(tp) > 0 else 0.0

            # Time in corner window
            seg_spd_safe = np.maximum(seg_spd, 1.0)
            dd = np.diff(seg_d, prepend=seg_d[0])
            t_corner = float(np.sum(dd / (seg_spd_safe / 3.6)))

            rows.append({
                "corner_num":               cnum,
                "driver":                   driver,
                "apex_speed_kmh":           round(apex_spd, 1),
                "brake_dist_before_apex_m": round(brake_dist, 1),
                "throttle_dist_after_apex_m": round(thr_dist, 1),
                "time_in_corner_s":         round(t_corner, 3),
            })

    return pd.DataFrame(rows)
