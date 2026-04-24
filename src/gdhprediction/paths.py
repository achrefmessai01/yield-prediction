import os
from pathlib import Path
from typing import Final, Literal

XdgPaths = Literal["XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME"]
DefPaths = Literal[".cache", ".config", ".local/share"]


class GdhPredictionPaths:
    """Class to hold paths for gdh prediction project."""

    name: str
    cache: Path
    config: Path
    data: Path

    def __init__(self, name: str = "gdhprediction") -> None:
        self.name = name
        self.cache = self._make_dir("XDG_CACHE_HOME", ".cache")
        self.config = self._make_dir("XDG_CONFIG_HOME", ".config")
        self.data = self._make_dir("XDG_DATA_HOME", ".local/share")

    def _make_dir(self, xdg: XdgPaths, default: DefPaths) -> Path:
        dir_path = Path(os.getenv(xdg, Path.home() / default)) / self.name
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    @property
    def forecast_cache(self) -> Path:
        return self.cache / "forecast.json"

    def annual_data(self, hashkey: str) -> Path:
        return self.data / f"annual_{hashkey}.npy"

    def annual_data_checksum(self, hashkey: str) -> Path:
        return self.data / f"annual_{hashkey}.sha256"

    @property
    def model_config(self) -> Path:
        return self.config / f"{self.name}.yaml"


APP_PATHS: Final = GdhPredictionPaths()
