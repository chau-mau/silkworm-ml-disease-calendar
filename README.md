# Tasar Silkworm ML Disease Calendar (Ranchi)

Round-the-year Virosis & Bacteriosis risk calendar for Ranchi, driven by Random Forest
models trained on CTRTI Ranchi 2025–2026 field data (weather-only features: Tmax, Tmin,
Humidity, THI (NRC 1971), Wind Speed). Replaces the earlier Humidity/Temperature ratio rules.

**Live:** https://chau-mau.github.io/silkworm-ml-disease-calendar/

## How it works
1. `generate_calendar.py` downloads 5 years of daily NASA POWER climate for Ranchi.
2. The trained models (`model/models.pkl`) predict disease probability for each day.
3. Probabilities are averaged per calendar date, smoothed over 7 days, and graded by
   rank within the year: bottom 40% Low, 40–70% Moderate, 70–90% High, top 10% Very High.
4. The trees are exported to `docs/rf_models.json` so the page's custom calculator runs
   the same model in the browser.

## Regenerate
```bash
pip install -r requirements.txt
python generate_calendar.py
```

Risk is relative (small, season-limited training set) — a decision-support tool, not a definitive forecast.
Developed by CSB-Central Tasar Research and Training Institute, Ranchi.
