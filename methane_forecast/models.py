import numpy as np
from lightgbm import LGBMRegressor
from .features import feature_columns


class DirectModel:
    def __init__(self, seed=42):
        self.seed = seed
        self.models = {}
        self.columns = {}

    def fit(self, frames, cutoff):
        for h, frame in frames.items():
            train = frame.loc[(frame.target_date <= cutoff) & frame.target.notna() & frame["last"].notna()]
            if train.empty:
                raise ValueError("No training observations before cutoff")
            cols = feature_columns(frame)
            model = LGBMRegressor(n_estimators=100, num_leaves=23, learning_rate=.07,
                                  verbosity=-1, n_jobs=2, random_state=self.seed)
            model.fit(train[cols], train.target - train["last"])
            self.models[h], self.columns[h] = model, cols
        return self

    def predict(self, frame, horizon):
        if frame.empty or frame["last"].isna().any():
            raise ValueError("Prediction requires station observation history")
        return frame["last"].to_numpy() + self.models[horizon].predict(frame[self.columns[horizon]])


class SeasonalClimatology:
    """Station-specific annual harmonic regression, fitted only before cutoff."""
    def fit(self, measurements, cutoff):
        self.coefficients = {}
        for station, group in measurements.loc[measurements.date <= cutoff].dropna().groupby("station_id"):
            self.coefficients[station] = np.linalg.lstsq(self.design(group.date), group.ch4, rcond=None)[0]
        return self

    @staticmethod
    def design(dates):
        phase = 2 * np.pi * dates.dt.dayofyear.to_numpy() / 365.25
        trend = (dates - np.datetime64("2023-01-01")).dt.days.to_numpy() / 365.25
        return np.column_stack([np.ones(len(dates)), np.sin(phase), np.cos(phase), trend])

    def predict(self, frame):
        return np.array([x @ self.coefficients[s] for x, s in zip(self.design(frame.target_date), frame.station_id)])
