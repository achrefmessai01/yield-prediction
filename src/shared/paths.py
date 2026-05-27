import os
from pathlib import Path
from typing import Final, Literal

XdgPaths = Literal["XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME"]
DefPaths = Literal[".cache", ".config", ".local/share"]


class FruitloadPredictionPaths:
    """Class to hold paths for fruitlaod prediction project."""

    name: str
    tenant: str
    batch: str
    cache: Path
    config: Path
    data: Path

    def __init__(self, tenant:str, batch: str, name: str = "fruitloadprediction") -> None:
        if tenant == "":
            raise ValueError("Tenant cannot be empty")
        if batch == "":
            raise ValueError("Batch cannot be empty")
        if name == "":
            raise ValueError("Name cannot be empty")
        self.tenant = tenant
        self.batch = batch
        self.name = name
        self.cache = self._make_dir("XDG_CACHE_HOME", ".cache")
        self.config = self._make_dir("XDG_CONFIG_HOME", ".config")
        self.data = self._make_dir("XDG_DATA_HOME", ".local/share")

    def _make_dir(self, xdg: XdgPaths, default: DefPaths) -> Path:
        dir_path = Path(os.getenv(xdg, Path.home() / default)) / self.name / self.tenant / self.batch
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    @property
    def model_config(self) -> Path:
        return self.config / f"{self.name}-config.yaml"


class YieldPredictionPaths:
    """Class to hold paths for yield prediction project."""

    name: str
    tenant: str
    batch: str
    cache: Path
    config: Path
    data: Path

    def __init__(self, tenant:str, batch: str, name: str = "yieldprediction") -> None:
        if tenant == "":
            raise ValueError("Tenant cannot be empty")
        if batch == "":
            raise ValueError("Batch cannot be empty")
        if name == "":
            raise ValueError("Name cannot be empty")
        self.tenant = tenant
        self.batch = batch
        self.name = name
        self.cache = self._make_dir("XDG_CACHE_HOME", ".cache")
        self.config = self._make_dir("XDG_CONFIG_HOME", ".config")
        self.data = self._make_dir("XDG_DATA_HOME", ".local/share")

    def _make_dir(self, xdg: XdgPaths, default: DefPaths) -> Path:
        dir_path = Path(os.getenv(xdg, Path.home() / default)) / self.name / self.tenant / self.batch
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    @property
    def model_config(self) -> Path:
        return self.config / f"{self.name}-config.yaml"
    


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

    def data_file(self, hashkey: str) -> Path:
        return self.data / f"{hashkey}.npy"

    def data_file_checksum(self, hashkey: str) -> Path:
        return self.data / f"{hashkey}.sha256"

    def cache_file(self, hashkey: str) -> Path:
        return self.cache / f"{hashkey}.npy"

    def cache_file_checksum(self, hashkey: str) -> Path:
        return self.cache / f"{hashkey}.sha256"

    @property
    def model_config(self) -> Path:
        return self.config / f"{self.name}.yaml"


APP_PATHS: Final = GdhPredictionPaths()