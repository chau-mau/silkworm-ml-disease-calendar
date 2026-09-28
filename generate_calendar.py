"""
Generate the ML-based Tasar Silkworm Disease Calendar (Ranchi).

Fetches the last 5 full years of daily climate from NASA POWER for Ranchi,
runs the trained Random Forest models (model/models.pkl, trained on CTRTI
Ranchi 2025-2026 field data) on every day, and averages the predicted
disease probability by day-of-year to build a typical-year calendar.

Features (weather only): Tmax, Tmin, Humidity, THI (NRC 1971), Wind_Speed
Targets: Virosis, Bacteriosis

Risk levels rank each day's model probability (7-day centred rolling mean)
against the rest of the year, per disease:
    bottom 40% -> Low | 40-70% -> Moderate | 70-90% -> High | top 10% -> Very High
The probability cut-offs are saved so the browser calculator uses the same scale.

Also exports the Random Forest trees to docs/rf_models.json so the web page's
custom calculator runs the same model in the browser.
"""

import json
import pickle
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
MODEL_PKL = ROOT / "model" / "models.pkl"
OUT_JSON = ROOT / "docs" / "calendar_data.json"
OUT_MODEL_JSON = ROOT / "docs" / "rf_models.json"

LAT, LON = 23.3441, 85.3096  # Ranchi, Jharkhand
END_YEAR = date.today().year - 1
START_YEAR = END_YEAR - 4

DISEASES = ["Virosis", "Bacteriosis"]
RISK_ORDER = {"Low": 0, "Moderate": 1, "High": 2, "Very High": 3}


def thi_nrc(tmax, tmin, rh):
    """NRC (1971) THI, T = mean of Tmax/Tmin (same formula used in training)."""
    t = (tmax + tmin) / 2.0
    return (1.8 * t + 32) - (0.55 - 0.0055 * rh) * (1.8 * t - 58)


PERCENTILES = (40, 70, 90)


def risk_from_prob(p, cuts):
    if p < cuts[0]:
        return "Low"
    if p < cuts[1]:
        return "Moderate"
    if p < cuts[2]:
        return "High"
    return "Very High"


def fetch_nasa_historical(lat, lon, start_year, end_year):
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": "T2M_MAX,T2M_MIN,RH2M,WS10M,PRECTOTCORR",
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": f"{start_year}0101",
        "end": f"{end_year}1231",
        "format": "JSON",
    }
    print(f"Fetching NASA POWER climate for Ranchi ({start_year}-{end_year})...")
    r = requests.get(url, params=params, timeout=180)
    r.raise_for_status()
    p = r.json()["properties"]["parameter"]
    rows = []
    for k in sorted(p["T2M_MAX"]):
        if -999 in (p["T2M_MAX"][k], p["T2M_MIN"][k], p["RH2M"][k], p["WS10M"][k]):
            continue
        rows.append({
            "date": f"{k[:4]}-{k[4:6]}-{k[6:]}",
            "tmax": p["T2M_MAX"][k],
            "tmin": p["T2M_MIN"][k],
            "humidity": p["RH2M"][k],
            "wind_speed": p["WS10M"][k],
            "rainfall": max(p["PRECTOTCORR"][k], 0),
        })
    return pd.DataFrame(rows)


def export_forest(rf):
    """Compact JSON of each tree: parallel arrays, leaf value = P(class 1)."""
    trees = []
    for est in rf.estimators_:
        t = est.tree_
        vals = t.value[:, 0, :]
        p1 = (vals[:, 1] / vals.sum(axis=1)).round(4).tolist()
        trees.append({
            "l": t.children_left.tolist(),
            "r": t.children_right.tolist(),
            "f": t.feature.tolist(),
            "t": np.round(t.threshold, 4).tolist(),
            "p": p1,
        })
    return trees


def main():
    bundle = pickle.load(open(MODEL_PKL, "rb"))
    feats = bundle["feature_cols"]
    models = {d: bundle["models"][d]["random_forest"] for d in DISEASES}

    df = fetch_nasa_historical(LAT, LON, START_YEAR, END_YEAR)
    print(f"Retrieved {len(df)} daily records.")
    X = pd.DataFrame({
        "Tmax": df.tmax, "Tmin": df.tmin, "Humidity": df.humidity,
        "THI": thi_nrc(df.tmax, df.tmin, df.humidity), "Wind_Speed": df.wind_speed,
    })[feats]
    df["thi"] = X["THI"]
    for d in DISEASES:
        df[f"p_{d}"] = models[d].predict_proba(X)[:, 1]
    base = {d: bundle["models"][d]["n_positive"] / bundle["models"][d]["n_samples"]
            for d in DISEASES}

    df["doy"] = df["date"].str[5:]
    by_doy = defaultdict(list)
    for row in df.to_dict("records"):
        by_doy[row["doy"]].append(row)

    typical_year = 2026
    records = []
    for m in range(1, 13):
        for dd in range(1, 32):
            try:
                day = date(typical_year, m, dd)
            except ValueError:
                continue
            rows = by_doy.get(f"{m:02d}-{dd:02d}", [])
            if not rows:
                continue
            avg = lambda k: float(np.mean([r[k] for r in rows]))
            rec = {
                "date": day.isoformat(),
                "tmax": round(avg("tmax"), 1),
                "tmin": round(avg("tmin"), 1),
                "humidity": round(avg("humidity"), 1),
                "wind_speed": round(avg("wind_speed"), 2),
                "rainfall": round(avg("rainfall"), 2),
                "thi": round(avg("thi"), 1),
                "years_sampled": len(rows),
            }
            for d in DISEASES:
                rec[d.lower()] = {"probability": avg(f"p_{d}")}
            records.append(rec)

    # 7-day centred (circular) rolling mean of probability, then grade by percentile
    cuts = {}
    for d in DISEASES:
        k = d.lower()
        p = np.array([r[k]["probability"] for r in records])
        sm = np.convolve(np.concatenate([p[-3:], p, p[:3]]), np.ones(7) / 7, "valid")
        cuts[d] = [round(float(c), 4) for c in np.percentile(sm, PERCENTILES)]
        for r, v in zip(records, sm):
            r[k] = {"probability": round(float(v), 3),
                    "risk": risk_from_prob(v, cuts[d])}
    for rec in records:
        v, b = rec["virosis"]["risk"], rec["bacteriosis"]["risk"]
        rec["combined_risk"] = v if RISK_ORDER[v] >= RISK_ORDER[b] else b

    info = {d: {k: bundle["models"][d][k] for k in
                ("rf_accuracy", "rf_accuracy_2026_holdout", "feature_importance",
                 "n_samples", "n_positive")} for d in DISEASES}
    summary = {
        "total_days": len(records),
        "historical_days": len(df),
        "year_range": f"{START_YEAR}-{END_YEAR}",
        "location": {"name": "Ranchi, Jharkhand", "lat": LAT, "lon": LON},
        "model": info,
        "base_rate": {d: round(base[d], 3) for d in DISEASES},
        "prob_cuts": cuts,
        "note": (
            "Calendar colours come from Random Forest models trained on CTRTI Ranchi "
            "2025-2026 field observations (features: Tmax, Tmin, Humidity, THI, Wind Speed). "
            f"Each day's risk is the model's disease probability averaged over {START_YEAR}-{END_YEAR} "
            "NASA POWER daily climate for the same calendar date."
        ),
        "disclaimer": (
            "Disease incidence is subject to rearing season, crop stage, and management practices. "
            "This calendar indicates climate-based risk only and should be used as a "
            "decision-support tool, not a definitive forecast."
        ),
    }
    OUT_JSON.write_text(json.dumps({"summary": summary, "records": records}, indent=1))
    OUT_MODEL_JSON.write_text(json.dumps(
        {"features": feats, "base_rate": base, "prob_cuts": cuts, "models": {d: export_forest(models[d]) for d in DISEASES}},
        separators=(",", ":")))
    print(f"Wrote {OUT_JSON} ({len(records)} days) and {OUT_MODEL_JSON}")
    for d in DISEASES:
        c = pd.Series([r[d.lower()]["risk"] for r in records]).value_counts().to_dict()
        print(d, c)


if __name__ == "__main__":
    main()
