import numpy as np
import pandas as pd
from .features import build_features
from .models import DirectModel, SeasonalClimatology


def backtest(stations, measurements, weather, horizons=(1, 2, 3, 4, 5), folds=3, window=14, seed=42):
    if folds < 1 or window < 1:
        raise ValueError("folds and window must be positive")
    end = measurements.date.max()
    first = end - pd.Timedelta(days=folds * window + max(horizons))
    if (first - measurements.date.min()).days < 60:
        raise ValueError("Insufficient history for requested backtest")
    frames = {h: build_features(stations, measurements, weather, h) for h in horizons}
    records = []
    for fold in range(folds):
        cutoff = first + pd.Timedelta(days=fold * window)
        model = DirectModel(seed).fit(frames, cutoff)
        seasonal = SeasonalClimatology().fit(measurements, cutoff)
        for h, frame in frames.items():
            test = frame.loc[(frame.origin > cutoff) & (frame.origin <= cutoff + pd.Timedelta(days=window))
                             & frame.target.notna() & frame["last"].notna()]
            if test.empty:
                raise ValueError("No scored measurements in fold")
            for name, pred in [("direct", model.predict(test, h)), ("persistence", test["last"].to_numpy()),
                               ("seasonal", seasonal.predict(test))]:
                records.extend((fold, h, name, float(err)) for err in pred - test.target.to_numpy())
    errors = pd.DataFrame(records, columns=["fold", "horizon", "model", "error"])
    rows = []
    for (h, name), group in errors.groupby(["horizon", "model"]):
        rows.append((str(h), name, np.sqrt(np.mean(group.error**2)), np.mean(abs(group.error)), len(group)))
    for name, group in errors.groupby("model"):
        rows.append(("overall", name, np.sqrt(np.mean(group.error**2)), np.mean(abs(group.error)), len(group)))
    return pd.DataFrame(rows, columns=["horizon", "model", "RMSE", "MAE", "n"])
