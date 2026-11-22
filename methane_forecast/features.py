import numpy as np
import pandas as pd
from .synthetic import WEATHER


def build_features(stations, measurements, weather, horizon, end=None):
    """Origin is end-of-day UTC; lag_1 is that day's last available daily mean.

    Calendar reindexing prevents missing days from compressing time. Expanding
    station encoding only includes observations available at each row's origin.
    """
    end = pd.Timestamp(end) if end is not None else measurements.date.max()
    dates = pd.date_range(measurements.date.min(), end)
    idx = pd.MultiIndex.from_product([stations.station_id, dates], names=["station_id", "origin"])
    raw = measurements.set_index(["station_id", "date"]).ch4.reindex(idx)
    group = raw.groupby(level=0)
    known = group.ffill()
    df = idx.to_frame(index=False)
    df["last"] = known.to_numpy()
    for lag in range(1, 15):
        df[f"lag_{lag}"] = known.groupby(level=0).shift(lag - 1).to_numpy()
    for window in [3, 7, 14]:
        for stat in ["mean", "std", "min", "max"]:
            df[f"roll_{window}_{stat}"] = known.groupby(level=0).transform(
                lambda s: getattr(s.rolling(window, min_periods=1), stat)()).to_numpy()
    df["diff_1"] = df.lag_1 - df.lag_2
    df["diff_7"] = df.lag_1 - df.lag_8
    df["station_encoding"] = group.transform(lambda s: s.expanding().mean()).to_numpy()
    observed_date = pd.Series(idx.get_level_values(1).where(raw.notna()), index=idx).groupby(level=0).ffill()
    df["days_since_observed"] = (idx.get_level_values(1) - pd.DatetimeIndex(observed_date)).days
    df["target_date"] = df.origin + pd.Timedelta(days=horizon)
    phase = 2 * np.pi * df.target_date.dt.dayofyear / 365.25
    df["doy_sin"], df["doy_cos"] = np.sin(phase), np.cos(phase)
    df["trend_days"] = (df.origin - pd.Timestamp("2020-01-01")).dt.days
    df = df.merge(stations, on="station_id", validate="many_to_one")
    for col in ["network", "continent"]:
        df[col] = pd.Categorical(df[col], categories=sorted(stations[col].unique())).codes
    forecasts = weather.loc[weather.lead_days == horizon, ["station_id", "issue_date", "valid_date", *WEATHER]]
    df = df.merge(forecasts, left_on=["station_id", "origin", "target_date"],
                  right_on=["station_id", "issue_date", "valid_date"], how="left", validate="one_to_one")
    truth = measurements.rename(columns={"date": "target_date", "ch4": "target"})
    df = df.merge(truth, on=["station_id", "target_date"], how="left", validate="one_to_one")
    return df.drop(columns=["issue_date", "valid_date"])


def feature_columns(frame):
    return [c for c in frame if c not in ["station_id", "origin", "target_date", "target"]]
