from pathlib import Path
import numpy as np
import pandas as pd


WEATHER = ["t2m", "u10", "v10", "wind_speed", "blh", "precip", "surface_pressure", "rh"]


def generate(days=540, n_stations=59, seed=42):
    if days < 30 or n_stations < 1:
        raise ValueError("Require at least 30 days and one station")
    rng = np.random.default_rng(seed)
    continents = ["Europe", "Asia", "North America", "South America", "Africa", "Oceania"]
    centers = [(50, 10), (35, 100), (42, -100), (-20, -60), (0, 20), (-30, 140)]
    stations, measurements, forecasts = [], [], []
    dates = pd.date_range("2023-01-01", periods=days + 5)
    for i in range(n_stations):
        lat, lon = centers[i % 6]
        lat, lon = lat + rng.normal(0, 7), lon + rng.normal(0, 8)
        sid = f"S{i:03d}"
        continental = int(i % 4 != 0)
        stations.append((sid, ["ICOS", "AGAGE", "SNO", "NOAA", "JMA"][i % 5],
                         continents[i % 6], lat, lon, rng.uniform(0, 2000), continental))
        n = len(dates)
        u, v = rng.normal(0, 3, (2, n))
        wind = np.hypot(u, v)
        blh = rng.uniform(180, 1800, n)
        season = np.cos(2 * np.pi * dates.dayofyear.to_numpy() / 365.25) * (1 if lat >= 0 else -1)
        temp = 288 - abs(lat) * .35 - 10 * season + rng.normal(0, 2, n)
        weather = np.column_stack([temp, u, v, wind, blh, rng.exponential(1, n),
                                   rng.normal(101325, 700, n), rng.uniform(30, 95, n)])
        noise = np.zeros(n)
        for j in range(1, n):
            noise[j] = .65 * noise[j - 1] + rng.normal(0, 1.2)
        ch4 = (1930 + np.arange(n) * 10 / 365.25 + .65 * lat + 14 * season
               + continental * (65 / (wind + 1) + 15000 / blh) + noise)
        missing = rng.random(days) < .025
        for _ in range(2):
            start = rng.integers(0, days - 4)
            missing[start:start + 3] = True
        for j in range(days):
            measurements.append((sid, dates[j], np.nan if missing[j] else ch4[j]))
        for lead in range(6):
            values = weather[lead:lead + days].copy()
            values += rng.normal(size=values.shape) * (lead * np.array([.35, .12, .12, .15, 12, .1, 30, 1]))
            values[:, 3:6] = np.maximum(values[:, 3:6], 0)
            frame = pd.DataFrame(values, columns=WEATHER)
            frame.insert(0, "lead_days", lead)
            frame.insert(0, "valid_date", dates[lead:lead + days])
            frame.insert(0, "issue_date", dates[:days])
            frame.insert(0, "station_id", sid)
            forecasts.append(frame)
    return (pd.DataFrame(stations, columns=["station_id", "network", "continent", "lat", "lon", "elevation", "continental"]),
            pd.DataFrame(measurements, columns=["station_id", "date", "ch4"]),
            pd.concat(forecasts, ignore_index=True))


def write_synthetic(config, days=540):
    path = Path(config.data_dir)
    path.mkdir(parents=True, exist_ok=True)
    for frame, name in zip(generate(days=days, seed=config.seed),
                           [config.stations_file, config.measurements_file, config.weather_file]):
        frame.to_csv(path / name, index=False)
