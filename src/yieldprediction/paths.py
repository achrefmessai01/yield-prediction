import os
from pathlib import Path
from typing import Final, Literal

XdgPaths = Literal["XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME"]
DefPaths = Literal[".cache", ".config", ".local/share"]


class YieldPredictionPaths:
    """Class to hold paths for yield prediction project."""

    name: str
    cache: Path
    config: Path
    data: Path

    def __init__(self, name: str = "yieldprediction") -> None:
        self.name = name
        self.cache = self._make_dir("XDG_CACHE_HOME", ".cache")
        self.config = self._make_dir("XDG_CONFIG_HOME", ".config")
        self.data = self._make_dir("XDG_DATA_HOME", ".local/share")

    def _make_dir(self, xdg: XdgPaths, default: DefPaths) -> Path:
        dir_path = Path(os.getenv(xdg, Path.home() / default)) / self.name
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    @property
    def model_config(self) -> Path:
        return self.config / f"{self.name}.yaml"


APP_PATHS: Final = YieldPredictionPaths()
