# AlphaBench Dataset Specification

**Version:** v1  
**Status:** Active  
**Date:** June 2026

---

## 1. Purpose

This document defines the exact format, naming conventions, column schema, and access rules for all AlphaBench benchmark datasets. `DatasetService` is implemented against this spec. Any dataset that does not conform to this spec will be rejected at load time.

---

## 2. Directory Layout

```text
data/
  v1/                        ← training data (visible to agent)
    manifest.json
    BTC-USDT.parquet
    ETH-USDT.parquet         ← additional assets (optional)
    SOL-USDT.parquet         ← additional assets (optional)

data_hidden/                 ← OOS data (never exposed to agent; .gitignored)
  v1/
    BTC-USDT.parquet
    ETH-USDT.parquet
    SOL-USDT.parquet
```

- `data/` is the **training root** — passed to `DatasetService` and `AgentRuntime`.
- `data_hidden/` is the **OOS root** — passed only to `EvaluationEngine`. It must never appear in any agent-facing code path.
- Both roots share the same version subdirectory structure and file format.

---

## 3. Versioning

Dataset versions are directory-based. The version string (e.g. `v1`) maps to a subdirectory under the data root.

- MVP uses version `v1`.
- Future versions increment the version string (`v2`, `v3`, ...).
- `DatasetService` is initialized with the data root path and a `TaskDefinition`, from which it reads the version string to access that subdirectory.

---

## 4. Manifest Schema

Each version directory must contain a `manifest.json` at its root.

### 4.1 Format

```json
{
  "version": "v1",
  "created_at": "2026-06-06T11:36:34Z",
  "training_end": "2024-12-31",
  "assets": [
    {
      "asset_id": "BTC-USDT",
      "exchange": "binance",
      "instrument_type": "spot",
      "base_currency": "USDT",
      "available_from": "2021-01-01",
      "available_to": "2024-12-31",
      "fields": ["open", "high", "low", "close", "volume"]
    },
    {
      "asset_id": "ETH-USDT",
      "exchange": "binance",
      "instrument_type": "spot",
      "base_currency": "USDT",
      "available_from": "2021-01-01",
      "available_to": "2024-12-31",
      "fields": ["open", "high", "low", "close", "volume"]
    },
    {
      "asset_id": "SOL-USDT",
      "exchange": "binance",
      "instrument_type": "spot",
      "base_currency": "USDT",
      "available_from": "2021-01-01",
      "available_to": "2024-12-31",
      "fields": ["open", "high", "low", "close", "volume"]
    }
  ]
}
```

### 4.2 Field Definitions

| Field | Type | Description |
|-------|------|-------------|
| `version` | string | Must match the parent directory name |
| `created_at` | ISO 8601 datetime string | When this dataset version was created |
| `training_end` | ISO date string (`YYYY-MM-DD`) | Hard cutoff enforced by `DatasetService.load()` |
| `assets[].asset_id` | string | Unique asset identifier; must match the parquet filename (see §5) |
| `assets[].exchange` | string | Exchange the data originates from (e.g. `"binance"`) |
| `assets[].instrument_type` | string (optional) | The type of instrument (e.g. `"spot"`, `"perpetual"`, `"future"`, `"option"`) |
| `assets[].base_currency` | string | Settlement currency (e.g. `"USDT"`) |
| `assets[].available_from` | ISO date string | Earliest date available in the parquet file |
| `assets[].available_to` | ISO date string | Latest date available in the parquet file |
| `assets[].fields` | list of strings | Column names present in the parquet file (must include all required columns) |

---

## 5. Parquet File Naming

Each asset maps to exactly one parquet file:

```
{asset_id}.parquet
```

Examples:
- `BTC-USDT.parquet`
- `ETH-USDT.parquet`

The filename (without extension) must exactly match the `asset_id` in `manifest.json`.

---

## 6. Parquet File Schema

### 6.1 Index Column

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | `datetime64[ns, UTC]` or `datetime64[ns]` | Date or datetime of the bar. **Must be the index column** of the DataFrame after loading. |

- For **daily data**, `timestamp` should be midnight UTC (e.g. `2021-01-01 00:00:00`).
- The index must be **monotonically increasing** and **unique**.
- When stored in parquet, `timestamp` is stored as a regular column named `timestamp`. `DatasetService.load()` sets it as the index after reading.

### 6.2 Required Columns

All assets must provide these columns:

| Column | Parquet Type | Description |
|--------|-------------|-------------|
| `open` | `float64` | Bar open price |
| `high` | `float64` | Bar high price |
| `low` | `float64` | Bar low price |
| `close` | `float64` | Bar close price |
| `volume` | `float64` | Bar volume in base currency units |

### 6.3 Optional Columns

Assets may include these additional columns. If present, they must be listed in `manifest.json` under `assets[].fields`.

| Column | Parquet Type | Description |
|--------|-------------|-------------|
| `funding_rate` | `float64` | Perpetual futures 8h funding rate (annualized or raw — must be documented per asset) |
| `open_interest` | `float64` | Open interest in base currency units |
| `basis` | `float64` | Futures basis (spot-adjusted) |
| `spread` | `float64` | Bid-ask spread |

### 6.4 Null / NaN Policy

- Required columns must have **no null values** in the training parquet files.
- Optional columns may have nulls; consuming code must handle them.
- `DatasetService.load()` does **not** fill or interpolate nulls — that is the agent's responsibility in EDA.

---

## 7. Temporal Conventions

| Convention | Value |
|------------|-------|
| Granularity (MVP) | **Daily** (`1D` bars) |
| Timezone | **UTC** |
| Bar timestamp semantics | **Open of bar** (i.e. `2021-01-01` = data for the day starting Jan 1 2021) |
| Date format in manifest | `YYYY-MM-DD` (ISO 8601 date) |
| Datetime format in manifest | `YYYY-MM-DDTHH:MM:SSZ` (ISO 8601 UTC) |

---

## 8. Training / OOS Split Rule

- `manifest.json` declares `training_end`.
- `DatasetService` enforces that the requested date range does not exceed `TaskDefinition.train_end` (which must not exceed `training_end` in the manifest). It raises `PermissionError` if `end > task.train_end`.
- The OOS period begins the **day after** the training period.
- OOS data lives in `data_hidden/` and is **never accessible** through `DatasetService`.
- `EvaluationEngine` loads OOS data directly from `data_hidden/` using the same parquet format — no `DatasetService` wrapper.

### Example Split (Active v1 Dataset)

```
training_end = "2024-12-31"

Training data:  2021-01-01 → 2024-12-31  (accessible to agent via DatasetService)
OOS data:       2025-01-01 → 2025-12-31  (accessible only to EvaluationEngine)
```

---

## 9. Access Rules Summary

| Who | Can access training data | Can access OOS data |
|-----|--------------------------|---------------------|
| `DatasetService` | ✅ Yes | ❌ No (raises `PermissionError` if requested) |
| `SandboxExecutor` (EDA) | ✅ Yes (via injected `df`) | ❌ No |
| `BacktestEngine` | ✅ Yes | ❌ No |
| `EvaluationEngine` | ✅ Yes (for in-sample validation) | ✅ Yes (for OOS scoring) |
| Agent code | ✅ Yes (via `run_eda` tool) | ❌ Never |

---

## 10. Example: Reading a Parquet File

```python
import pyarrow.parquet as pq
import pandas as pd

table = pq.read_table("data/v1/BTC-USDT.parquet")
df = table.to_pandas()
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.set_index("timestamp").sort_index()

# df.index is now DatetimeIndex (UTC)
# df.columns = ["open", "high", "low", "close", "volume"]
```

---

## 11. Validation Checklist (for data providers)

Before adding a new asset parquet file, verify:

- [ ] Filename matches `asset_id` in manifest exactly
- [ ] `timestamp` column is present and parseable as datetime
- [ ] Index is monotonically increasing with no duplicates
- [ ] All required columns (`open`, `high`, `low`, `close`, `volume`) are present
- [ ] No nulls in required columns
- [ ] Optional columns listed in `manifest.json` under `fields`
- [ ] Date range matches `available_from` / `available_to` in manifest
- [ ] `available_to` does not exceed `training_end`

---

## 12. Future Extensions

- Multi-timeframe support (hourly bars) — would add a `timeframe` field to manifest
- Multi-market tasks — same format, multiple asset IDs in `asset_universe`
- Dataset registry tooling — formal versioning beyond directory-based scheme
