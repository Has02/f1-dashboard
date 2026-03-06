# 🏎️ F1 Sector Analysis Dashboard

A Streamlit dashboard for Formula 1 sector & split analysis, powered by the [OpenF1 API](https://openf1.org).

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Features (MVP)

- **Session Browser** — Select season (2025/2026), Grand Prix, and session type (FP1–3, Quali, Race)
- **Driver Comparison** — Pick up to 5 drivers side-by-side
- **Lap Time Evolution** — Line chart with tyre compound markers
- **Sector Breakdown** — Best S1/S2/S3 per driver, grouped bar chart
- **Stacked Sector View** — Per-lap sector composition for any driver
- **Sector Delta** — Per-sector gap vs a reference driver, lap by lap
- **Lap Table** — Full sortable/filterable raw lap data

## Planned Future Features

- [ ] Telemetry overlays (speed, throttle, brake, DRS)
- [ ] Mini-sector / speed trap splits
- [ ] Tyre stint strategy timeline
- [ ] Position change chart (race)
- [ ] Weather overlay
- [ ] Championship standings tracker
- [ ] Live session mode (requires OpenF1 sponsor tier)

## Data

OpenF1 is free for historical data (no API key needed). Rate limit: 3 req/s, 30 req/min.
Data is cached locally per session to avoid redundant requests.
