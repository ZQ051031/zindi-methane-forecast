import numpy as np
import pandas as pd


def build_submission(predictions, config, sample=None):
    if predictions.empty or predictions.duplicated(["station_id", "target_date"]).any():
        raise ValueError("Empty or duplicate predictions")
    ids = [config.id_format.format(station_id=r.station_id, date=r.target_date.strftime(config.date_format))
           for r in predictions.itertuples()]
    result = pd.DataFrame({"ID": ids, "Target": predictions.prediction.to_numpy()})
    if result.ID.duplicated().any() or not np.isfinite(result.Target).all():
        raise ValueError("Duplicate IDs or nonfinite predictions")
    if sample is not None:
        template = pd.read_csv(sample, dtype={"ID": str})
        if "ID" not in template or template.ID.isna().any() or template.ID.duplicated().any():
            raise ValueError("Sample needs unique nonempty ID column")
        if set(template.ID) != set(result.ID):
            raise ValueError("Sample IDs do not exactly cover forecast station/date grid")
        result = template[["ID"]].merge(result, on="ID", how="left", validate="one_to_one")
    return result
