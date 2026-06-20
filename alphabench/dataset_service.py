"""
dataset_service.py
------------------
Read-only access to versioned training data and asset metadata.
Enforces the training cutoff defined in TaskDefinition.
Never exposes hidden OOS data.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .contracts import AssetMetadata, TaskDefinition


class DatasetService:
    """
    Provides read-only access to training data for a given TaskDefinition.

    The service reads from ``data_root / task.dataset_version /``.
    Any request for data beyond ``task.train_end`` raises PermissionError.
    """

    def __init__(self, data_root: Path, task: TaskDefinition) -> None:
        self._task = task
        self._root = Path(data_root) / task.dataset_version
        manifest_path = self._root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        self._manifest: dict = json.loads(manifest_path.read_text())
        
        # Validate task cutoff does not exceed manifest cutoff
        manifest_cutoff = date.fromisoformat(self._manifest["training_end"])
        self._cutoff: date = date.fromisoformat(task.train_end)
        if self._cutoff > manifest_cutoff:
            raise ValueError(
                f"Task train_end ({task.train_end}) exceeds dataset training_end ({self._manifest['training_end']}) in manifest.json"
            )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def list_assets(self) -> list[str]:
        """Return the list of available asset IDs."""
        return [a["asset_id"] for a in self._manifest["assets"]]

    def get_metadata(self, asset_id: str) -> AssetMetadata:
        """Return metadata for the given asset_id."""
        for a in self._manifest["assets"]:
            if a["asset_id"] == asset_id:
                return AssetMetadata(**a)
        raise KeyError(f"Unknown asset: {asset_id!r}")

    def load(
        self,
        asset_id: str,
        start: date,
        end: date,
        fields: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Load OHLCV data for *asset_id* between *start* and *end* (inclusive).

        Parameters
        ----------
        asset_id:
            Asset to load, e.g. "BTC-USDT".
        start:
            First date to include (inclusive).
        end:
            Last date to include (inclusive). Must not exceed task.train_end.
        fields:
            Optional list of columns to load. If None, all columns are loaded.
            The timestamp index is always included regardless.

        Returns
        -------
        pd.DataFrame
            DatetimeIndex (UTC), columns per *fields* (or all if None).

        Raises
        ------
        PermissionError
            If *end* exceeds the training cutoff.
        KeyError
            If *asset_id* is not in the manifest.
        FileNotFoundError
            If the parquet file does not exist.
        """
        if end > self._cutoff:
            raise PermissionError(
                f"Requested end date {end} exceeds training cutoff {self._cutoff}. "
                "Hidden evaluation data is not accessible through DatasetService."
            )

        # Validate asset exists (raises KeyError if not)
        self.get_metadata(asset_id)

        path = self._root / f"{asset_id}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Parquet file not found: {path}")

        # Read with optional column selection (always include timestamp)
        columns = None
        if fields is not None:
            columns = ["timestamp"] + [f for f in fields if f != "timestamp"]

        table = pq.read_table(path, columns=columns)
        df = table.to_pandas()

        # Set timestamp as index
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()

        # Slice by date range
        return df.loc[str(start): str(end)]

    def load_full(self, asset_id: str) -> pd.DataFrame:
        """
        Convenience: load all training data for *asset_id* using task date range.
        Equivalent to load(asset_id, task.train_start, task.train_end).
        """
        return self.load(
            asset_id,
            start=date.fromisoformat(self._task.train_start),
            end=date.fromisoformat(self._task.train_end),
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def version(self) -> str:
        return self._manifest["version"]

    @property
    def cutoff(self) -> date:
        return self._cutoff
