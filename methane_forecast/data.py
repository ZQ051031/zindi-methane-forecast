from pathlib import Path
import numpy as np
import pandas as pd
from .synthetic import WEATHER


def load_table(path, kind, columns=None):
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    df = df.rename(columns=(columns or {}).get(kind, {}))
    required = {"stations": ["station_id", "network", "continent", "lat", "lon", "elevation", "continental"],
                "measurements": ["station_id", "date", "ch4"],
                "weather": ["station_id", "issue_date", "valid_date", "lead_days", *WEATHER]}[kind]
    if missing := set(required) - set(df):
        raise ValueError(f"{kind}: missing columns {sorted(missing)}")
    df = df[required].copy()
    if df.station_id.isna().any():
        raise ValueError("Missing station ID")
    df["station_id"] = df.station_id.astype(str)
    for col in [c for c in required if c.endswith("date")]:
        df[col] = pd.to_datetime(df[col], errors="raise", utc=True).dt.tz_convert(None).dt.normalize()
        if df[col].isna().any():
            raise ValueError(f"Missing {col}")
    numeric = [c for c in required if c in ["ch4", "lat", "lon", "elevation", "continental", "lead_days", *WEATHER]]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="raise")
    if np.isinf(df[numeric].to_numpy()).any():
        raise ValueError("Infinite numeric value")
    if kind == "measurements":
        return df.groupby(["station_id", "date"], as_index=False).ch4.mean()
    keys = ["station_id"] if kind == "stations" else ["station_id", "issue_date", "valid_date"]
    if df.duplicated(keys).any() or df.isna().any().any():
        raise ValueError(f"{kind}: duplicate keys or missing values")
    if kind == "weather" and not (((df.valid_date - df.issue_date).dt.days == df.lead_days) & df.lead_days.between(0, 5)).all():
        raise ValueError("Invalid forecast lead/date alignment")
    return df


def load_data(config):
    frames = tuple(load_table(Path(config.data_dir) / name, kind, config.columns)
                   for kind, name in [("stations", config.stations_file), ("measurements", config.measurements_file), ("weather", config.weather_file)])
    for df in frames[1:]:
        if not set(df.station_id) <= set(frames[0].station_id):
            raise ValueError("Unknown station ID")
    return frames
