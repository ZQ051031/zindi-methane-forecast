# Methane forecasting: synthetic-first Zindi starter

A complete Python pipeline for the **Green Rising Methane Innovation Challenge**:
daily CH4 (ppb), 59 stations, six continents, five future days. The challenge is
scheduled to open December 11, 2026. Data and submission specifications are not
published ("More info coming soon"). **All formats here are assumptions, not
official competition specifications.** No Zindi login or registration is needed.

## Quickstart

Python 3.10+ on Linux (LightGBM may require `libgomp1`):

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
python -m methane_forecast.cli run --config config.yaml
```

`run` generates synthetic CSV files, backtests, trains, reloads the saved model,
and writes `outputs/submission.csv`. It overwrites synthetic inputs, so **never
use `run` against a directory containing real data**. On the development machine
the default 540-day pipeline took **15.50 seconds** (excluding installation).

Individual commands:

```sh
python -m methane_forecast.cli make-synthetic --config config.yaml --days 540
python -m methane_forecast.cli backtest --config config.yaml
python -m methane_forecast.cli train --config config.yaml --origin 2024-06-23
python -m methane_forecast.cli predict --config config.yaml --origin 2024-06-23
# Optional official template alignment:
python -m methane_forecast.cli predict --config config.yaml --origin 2024-06-23 --sample data/SampleSubmission.csv
```

Outputs: `backtest.csv`, `model.pkl`, `submission.csv`. Only unpickle artifacts
you created/trust. Data, outputs, and the virtual environment are git-ignored.

## Pipeline and availability contract

- `config.py`: YAML-backed dataclass, input/output paths, per-table column
  mappings, forecast horizons, ID/date templates, deterministic seed.
- `synthetic.py`: 59 synthetic stations spanning Europe, Asia, North America,
  South America, Africa and Oceania, with ICOS/AGAGE/SNO/NOAA/JMA network labels
  (not actual network site metadata). CH4 includes a ~1930 ppb reference,
  +10 ppb/year trend, north/south latitude gradient, hemisphere-phased annual
  cycle, continental low-wind/low-BLH enhancements, AR(1) noise and missing gaps.
  Archived forecasts contain issue/valid dates, leads 0..5, temperature, wind
  components/speed, BLH, precipitation, pressure and humidity. Forecast error
  grows with lead. Units: K, m/s, m, mm/day, Pa, percent respectively.
- `data.py`: CSV/Parquet readers, source-to-canonical renaming, numeric/schema
  and duplicate-key checks, UTC daily means for sub-daily observations. Metadata
  and weather must be complete; methane NaNs are allowed. Unknown station IDs
  and inconsistent forecast lead dates fail loudly.
- `features.py`: calendar-day lags 1..14, rolling mean/std/min/max over 3/7/14
  days, 1/7-day differences, annual sine/cosine, trend, metadata, expanding
  station target mean and time since last observation. Calendar reindexing and
  within-station forward fill preserve gaps; no backward fill is used.
- `models.py`: persistence, station-specific harmonic seasonal-climatology
  regression with trend, and **direct LightGBM**, a separate delta-to-last-value
  regressor for each horizon. Recursive forecasting is not enabled.
- `validation.py`: three rolling training cutoffs, each scoring 14 subsequent
  daily origins. Models train only on rows whose **target date <= cutoff**;
  this naturally purges horizon-overlapping labels. Actual missing targets are
  omitted, never imputed for scoring. Each origin uses its newly available
  observations even though models remain fixed for the fold. RMSE/MAE pool all
  observed station/date pairs, including all folds, with counts reported.
- `submission.py`: default `ID,Target` format, station × origin+1..5, unique
  IDs and finite values; a sample file must cover exactly the same ID set and
  is preserved in its exact row order. A coverage mismatch is an error rather
  than a silent fallback.

**Origin means end of the UTC day**: that day's daily mean must already be
available. `lag_1` is the observation available at origin, `lag_2` the previous
calendar day's carried-forward value. Station encoding is expanding through
each origin, not a full-dataset mean: even training rows cannot see later labels.
Weather joins use `(station_id, issue_date=origin, valid_date=origin+h)` and never
realized future weather or a later forecast issue. Historical backtests require
archived forecasts, not weather reanalysis. Tests perturb future observations
and later-issued forecasts at every horizon and verify identical origin features.
If real observations arrive with delay, shift their availability timestamps in
your ingestion step before using this contract. No arbitrary long-gap exclusion
is imposed: stale values are carried forward with an explicit age feature.

## Actual synthetic backtest results

Seed 42, 540 days, 59 stations, three 14-origin folds; units ppb. These synthetic
scores demonstrate functionality, **not expected competition performance**.
The simulator deliberately has strong weather-driven daily variability, so its
archived weather forecasts are unusually valuable relative to persistence.

| Horizon | Direct RMSE | Direct MAE | Persistence RMSE | Persistence MAE | Seasonal RMSE | Seasonal MAE | Scored pairs |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.546 | 3.304 | 20.218 | 13.260 | 14.581 | 10.132 | 2391 |
| 2 | 4.935 | 3.588 | 20.369 | 13.418 | 14.615 | 10.143 | 2393 |
| 3 | 5.596 | 3.975 | 20.737 | 13.570 | 14.637 | 10.166 | 2395 |
| 4 | 5.821 | 4.156 | 20.168 | 13.310 | 14.590 | 10.160 | 2392 |
| 5 | 6.145 | 4.324 | 20.763 | 13.763 | 14.512 | 10.107 | 2389 |
| Overall | 5.440 | 3.869 | 20.453 | 13.464 | 14.587 | 10.142 | 11960 |

## Swapping in the real Zindi data

1. **Wait for official documentation.** Verify target units, forecast origin
   timing, station coverage, permitted external data, and submission rules.
   This project neither downloads restricted data nor submits automatically.
2. Put files under `data/` (or change `data_dir` and filenames in `config.yaml`).
   Supported canonical schemas:
   - Stations: `station_id,network,continent,lat,lon,elevation,continental`.
     `continental` is a numeric inland-site indicator (0/1); derive it from
     documented site metadata. Coordinates are degrees; elevation is meters.
   - Measurements: `station_id,date,ch4`. Sub-daily timestamps are normalized
     to UTC days and averaged. Convert methane to ppb before loading.
   - Weather: `station_id,issue_date,valid_date,lead_days,t2m,u10,v10,wind_speed,blh,precip,surface_pressure,rh`.
     Provide one row per station/issue/valid date, leads 0..5; aggregate/choose
     a consistent daily forecast issuance upstream. Do not substitute a forecast
     issued after the origin. Derive wind speed from components upstream if needed.
3. Edit source-to-canonical mappings independently in YAML, for example:

   ```yaml
   columns:
     stations: {site: station_id, latitude: lat, longitude: lon}
     measurements: {site: station_id, timestamp: date, methane_ppb: ch4}
     weather: {site: station_id, initialization: issue_date, forecast_date: valid_date}
   ```

   Unmapped canonical columns keep their names. Changing physical units or
   deriving missing fields requires an explicit ingestion transform, not just
   renaming. Parquet filenames are accepted (pyarrow included).
4. Configure `id_format` (placeholders `{station_id}` and `{date}`) and
   `date_format` (strftime syntax) to match published IDs. Default:
   `"{station_id}_{date}"` and `"%Y-%m-%d"`. Pass `--sample
   data/SampleSubmission.csv` to fill IDs exactly, check full coverage, and
   preserve template order. If the published template represents a different
   station/date grid or uses opaque IDs, adapt the builder explicitly; it
   intentionally refuses to guess or silently drop stations/dates.
5. Run `backtest` on real historical observations and genuine archived weather
   forecasts; inspect scores by horizon and observation coverage. Adjust folds
   via the Python `backtest(..., folds=..., window=...)` API if needed.
6. For **daily recurring forecasts**, ingest only newly released observations
   and the weather issue available that day, append to the historical archive,
   retrain with `train --origin YYYY-MM-DD`, then `predict --origin YYYY-MM-DD
   --sample ...`. Each prediction covers the next five dates, never the origin.
   A previously trained model can be reused at a later origin with unchanged
   config/metadata; retraining daily is the simplest workflow. Save a dated copy
   of each output outside `outputs/submission.csv` before the next run, retain
   issue archives, and score once future measurements arrive. Prediction fails
   if required origin-issued weather is absent or a station has no history.
   Submit manually according to the competition's eventual rules/deadlines.

## Tests and CI

`python -m pytest -q` covers synthetic schemas, missing calendar days, all five
no-leakage horizons, direct-model improvement over persistence on deterministic
synthetic folds, loader column mapping and Parquet, error paths, ID shape/count,
finite targets and sample-file alignment. GitHub Actions runs the suite on push
and pull requests. Synthetic improvement is a regression test, not a guarantee
on unseen real stations or a substitute for a real time-based evaluation.
