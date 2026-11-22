import numpy as np
import pandas as pd
import pytest
from methane_forecast.config import Config
from methane_forecast.synthetic import generate, WEATHER
from methane_forecast.data import load_table
from methane_forecast.features import build_features, feature_columns
from methane_forecast.models import DirectModel
from methane_forecast.submission import build_submission
from methane_forecast.validation import backtest


@pytest.fixture(scope="module")
def dataset():
    return generate(days=360, n_stations=12)


def test_synthetic_schema():
    stations, measurements, weather = generate(days=60)
    assert len(stations) == 59
    assert stations.continent.nunique() == 6
    assert stations.network.nunique() == 5
    assert len(measurements) == 59 * 60
    assert len(weather) == 59 * 60 * 6
    assert set(WEATHER) <= set(weather)
    assert set(weather.lead_days) == set(range(6))
    assert measurements.ch4.isna().any()
    assert measurements.ch4.dropna().between(1750, 2250).all()
    assert ((weather.valid_date - weather.issue_date).dt.days == weather.lead_days).all()


@pytest.mark.parametrize("horizon", [1, 2, 3, 4, 5])
def test_no_future_leakage(dataset, horizon):
    stations, measurements, weather = dataset
    origin = measurements.date.min() + pd.Timedelta(days=200)
    original = build_features(*dataset, horizon)
    changed = measurements.copy()
    changed.loc[changed.date > origin, "ch4"] = 100000
    new_weather = weather.copy()
    new_weather.loc[new_weather.issue_date > origin, WEATHER] = -9999
    mutated = build_features(stations, changed, new_weather, horizon)
    cols = feature_columns(original)
    pd.testing.assert_frame_equal(original.loc[original.origin <= origin, cols],
                                  mutated.loc[mutated.origin <= origin, cols])
    # The weather at the target date is legitimately available when issued at origin.
    row = original.loc[(original.origin == origin) & (original.station_id == "S000")].iloc[0]
    expected = weather.loc[(weather.station_id == "S000") & (weather.issue_date == origin)
                           & (weather.lead_days == horizon), "blh"].iloc[0]
    assert row.blh == expected


def test_direct_beats_persistence(dataset):
    metrics = backtest(*dataset, folds=2, window=10)
    table = metrics.pivot(index="horizon", columns="model", values="RMSE")
    assert (table.direct < table.persistence).all(), table


def test_mapping_and_daily_aggregation(tmp_path):
    path = tmp_path / "obs.csv"
    pd.DataFrame({"site": ["A", "A"], "timestamp": ["2024-01-01 01:00", "2024-01-01 12:00"],
                  "methane": [1900, 1920]}).to_csv(path, index=False)
    mapping = {"measurements": {"site": "station_id", "timestamp": "date", "methane": "ch4"}}
    result = load_table(path, "measurements", mapping)
    assert len(result) == 1 and result.ch4.iloc[0] == 1910
    with pytest.raises(ValueError, match="missing columns"):
        load_table(path, "measurements")
    parquet = tmp_path / "obs.parquet"
    result.to_parquet(parquet)
    pd.testing.assert_frame_equal(result, load_table(parquet, "measurements"))


def test_gap_calendar_and_model_empty(dataset):
    stations, measurements, weather = dataset
    origin = measurements.date.min() + pd.Timedelta(days=200)
    modified = measurements.loc[measurements.date != origin].copy()
    frame = build_features(stations, modified, weather, 1)
    at_origin = frame.loc[frame.origin == origin]
    assert len(at_origin) == len(stations)
    assert (at_origin.days_since_observed >= 1).all()
    with pytest.raises(ValueError, match="No training"):
        DirectModel().fit({1: frame}, measurements.date.min())
    with pytest.raises(ValueError, match="Insufficient history"):
        backtest(stations, measurements.head(10), weather)


def test_submission(tmp_path):
    config = Config()
    predictions = pd.DataFrame([(f"S{s:03d}", d, 1930.) for s in range(59)
                                for d in pd.date_range("2025-01-02", periods=5)],
                               columns=["station_id", "target_date", "prediction"])
    result = build_submission(predictions, config)
    assert list(result) == ["ID", "Target"]
    assert len(result) == 59 * 5 and not result.isna().any().any()
    assert result.ID.iloc[0] == "S000_2025-01-02"
    sample = tmp_path / "sample.csv"
    result.iloc[::-1].to_csv(sample, index=False)
    aligned = build_submission(predictions, config, sample)
    assert aligned.ID.tolist() == result.ID.iloc[::-1].tolist()
    result.iloc[:-1].to_csv(sample, index=False)
    with pytest.raises(ValueError, match="cover"):
        build_submission(predictions, config, sample)
    with pytest.raises(ValueError, match="Empty"):
        build_submission(predictions.iloc[:0], config)
    predictions.loc[0, "prediction"] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        build_submission(predictions, config)
