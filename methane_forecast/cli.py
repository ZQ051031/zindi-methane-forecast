import argparse
import pickle
from pathlib import Path
import pandas as pd
from .config import Config
from .data import load_data
from .synthetic import write_synthetic
from .features import build_features
from .models import DirectModel
from .validation import backtest
from .submission import build_submission


def main():
    parser = argparse.ArgumentParser(description="Synthetic-first methane forecasting")
    parser.add_argument("command", choices=["make-synthetic", "backtest", "train", "predict", "run"])
    parser.add_argument("--config")
    parser.add_argument("--days", type=int, default=540)
    parser.add_argument("--origin", help="UTC origin date (default latest measurement date)")
    parser.add_argument("--sample", help="SampleSubmission.csv; must exactly cover prediction grid")
    args = parser.parse_args()
    config = Config.load(args.config)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.command in ["make-synthetic", "run"]:
        write_synthetic(config, args.days)
        print(f"Synthetic data saved to {config.data_dir}")
    if args.command == "make-synthetic":
        return
    stations, measurements, weather = load_data(config)
    origin = pd.Timestamp(args.origin) if args.origin else measurements.date.max()
    if args.command in ["backtest", "run"]:
        metrics = backtest(stations, measurements, weather, config.horizons, seed=config.seed)
        metrics.to_csv(output / "backtest.csv", index=False)
        print(metrics.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if args.command == "backtest":
        return
    if args.command in ["train", "run"]:
        frames = {h: build_features(stations, measurements, weather, h, end=origin) for h in config.horizons}
        model = DirectModel(config.seed).fit(frames, origin)
        with (output / "model.pkl").open("wb") as handle:
            pickle.dump({"model": model, "cutoff": origin, "stations": stations, "config": config}, handle)
        print(f"Trained through {origin.date()}")
    if args.command in ["predict", "run"]:
        # Only load model artifacts you trust: pickle can execute code.
        with (output / "model.pkl").open("rb") as handle:
            artifact = pickle.load(handle)
        if artifact["cutoff"] > origin:
            raise ValueError("Model training cutoff is later than forecast origin")
        if artifact["config"] != config or not artifact["stations"].equals(stations):
            raise ValueError("Config or station metadata changed; retrain model")
        predictions = []
        for h in config.horizons:
            frame = build_features(stations, measurements, weather, h, end=origin)
            frame = frame.loc[frame.origin == origin].copy()
            if frame[["t2m", "blh"]].isna().any().any():
                raise ValueError("Missing origin-issued weather forecasts; fetch forecasts before predicting")
            frame["prediction"] = artifact["model"].predict(frame, h)
            predictions.append(frame[["station_id", "target_date", "prediction"]])
        submission = build_submission(pd.concat(predictions), config, args.sample)
        submission.to_csv(output / "submission.csv", index=False)
        print(f"Wrote {len(submission)} predictions to {output / 'submission.csv'}")


if __name__ == "__main__":
    main()
