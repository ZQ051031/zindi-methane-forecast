from dataclasses import dataclass, field, asdict
from pathlib import Path
import yaml


@dataclass
class Config:
    data_dir: str = "data"
    output_dir: str = "outputs"
    stations_file: str = "stations.csv"
    measurements_file: str = "measurements.csv"
    weather_file: str = "weather.csv"
    horizons: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    id_format: str = "{station_id}_{date}"
    date_format: str = "%Y-%m-%d"
    columns: dict = field(default_factory=dict)
    seed: int = 42

    def __post_init__(self):
        if not self.horizons or any(h not in range(1, 6) for h in self.horizons):
            raise ValueError("horizons must be a nonempty subset of 1..5")

    @classmethod
    def load(cls, path=None):
        return cls(**(yaml.safe_load(Path(path).read_text()) or {})) if path else cls()

    def save(self, path):
        Path(path).write_text(yaml.safe_dump(asdict(self), sort_keys=False))
