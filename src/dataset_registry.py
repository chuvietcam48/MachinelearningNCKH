"""
src/dataset_registry.py
=======================
Dataset Registry — Plugin-based loader system.

Centralises all dataset-specific knowledge (file path, loader function,
snapshot logic, display name) in one place.  Adding a new dataset requires:

  1. Write `src/data_loader_<name>.py` with a `load()` function.
  2. Call `register_dataset(...)` below — no changes to `main.py` or `app.py`.

Registry contract
-----------------
Each entry in the registry must provide:
  * name        (str)  : canonical key, e.g. "uci", "tafeng", "cdnow"
  * display     (str)  : human-readable label for UI / logs
  * data_path   (str)  : absolute path to the raw data file
  * loader_fn   (callable) : fn(path: str) -> pd.DataFrame
                             Output must have standard schema:
                             [CustomerID, InvoiceNo, InvoiceDate, TotalSpend]
  * snapshot_fn (callable, optional) : fn(df: pd.DataFrame) -> pd.Timestamp
                             Defaults to  max(InvoiceDate) + 1 day.

Usage
-----
    from src.dataset_registry import get_dataset, list_datasets

    info   = get_dataset("uci")          # raises KeyError if not found
    df     = info.loader_fn(info.data_path)
    snap   = info.snapshot_fn(df)
    datasets = list_datasets()           # [("uci", "UCI Online Retail"), ...]
"""

import os
import logging
import pandas as pd
from dataclasses import dataclass, field
from typing import Callable, Optional, Dict

logger = logging.getLogger(__name__)

# ── Root path helper ──────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_currency_symbol() -> str:
    """
    Read the currency symbol from config/simulation_params.yaml.

    Returns
    -------
    str
        E.g. '£' for GBP, '$' for USD.  Falls back to '£' if config missing.
    """
    cfg_path = os.path.join(_ROOT, "config", "simulation_params.yaml")
    try:
        import yaml
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("economics", {}).get("currency_symbol", "£")
    except Exception:
        return "£"


def get_currency_code() -> str:
    """Read the ISO currency code from config (e.g. 'GBP', 'USD')."""
    cfg_path = os.path.join(_ROOT, "config", "simulation_params.yaml")
    try:
        import yaml
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("economics", {}).get("currency", "GBP")
    except Exception:
        return "GBP"




def _default_snapshot(df: pd.DataFrame) -> pd.Timestamp:
    """Standard snapshot = last invoice date + 1 day."""
    snap = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    logger.info(f"Snapshot date: {snap.date()}")
    return snap


# ── Registry entry ────────────────────────────────────────────────────────────
@dataclass
class DatasetInfo:
    name: str
    display: str
    data_path: str
    loader_fn: Callable
    snapshot_fn: Callable = field(default=_default_snapshot)


# ── Registry store ────────────────────────────────────────────────────────────
_REGISTRY: Dict[str, DatasetInfo] = {}


def register_dataset(
    name: str,
    display: str,
    data_path: str,
    loader_fn: Callable,
    snapshot_fn: Optional[Callable] = None,
) -> None:
    """
    Register a dataset into the global registry.

    Parameters
    ----------
    name : str
        Canonical key (lowercase, no spaces). Used as CLI --dataset value.
    display : str
        Human-readable label shown in logs and Streamlit sidebar.
    data_path : str
        Absolute path to the raw data file.
    loader_fn : Callable
        Function (path: str) -> pd.DataFrame with standard schema.
    snapshot_fn : Callable, optional
        Function (df: pd.DataFrame) -> pd.Timestamp.
        Defaults to max(InvoiceDate) + 1 day.
    """
    if name in _REGISTRY:
        logger.warning(f"[Registry] Overwriting existing dataset registration: '{name}'")
    _REGISTRY[name] = DatasetInfo(
        name=name,
        display=display,
        data_path=data_path,
        loader_fn=loader_fn,
        snapshot_fn=snapshot_fn if snapshot_fn is not None else _default_snapshot,
    )
    logger.debug(f"[Registry] Registered dataset: '{name}' ({display})")


def get_dataset(name: str) -> DatasetInfo:
    """
    Retrieve a registered dataset by name.

    Raises
    ------
    KeyError
        If name is not registered. Lists available datasets in the message.
    """
    if name not in _REGISTRY:
        available = ", ".join(f"'{k}'" for k in _REGISTRY)
        raise KeyError(
            f"Unknown dataset '{name}'. "
            f"Available: [{available}]. "
            f"Register it via src.dataset_registry.register_dataset(...)."
        )
    return _REGISTRY[name]


def list_datasets() -> list:
    """
    Return a list of (name, display) tuples for all registered datasets.
    Useful for populating CLI choices and Streamlit sidebars.
    """
    return [(k, v.display) for k, v in _REGISTRY.items()]


# =============================================================================
# Built-in Dataset Registrations
# =============================================================================
# Import loaders lazily inside lambda to avoid circular imports at module load.
# All data paths use the project root as the base.

def _load_uci(path: str) -> pd.DataFrame:
    from src.data_loader import load_and_clean
    return load_and_clean(path)

def _snapshot_uci(df: pd.DataFrame) -> pd.Timestamp:
    from src.data_loader import get_snapshot_date
    return get_snapshot_date(df)

def _load_tafeng(path: str) -> pd.DataFrame:
    from src.data_loader_tafeng import load_and_clean_tafeng
    return load_and_clean_tafeng(path)

def _snapshot_tafeng(df: pd.DataFrame) -> pd.Timestamp:
    from src.data_loader_tafeng import get_snapshot_date_tafeng
    return get_snapshot_date_tafeng(df)

def _load_cdnow(path: str) -> pd.DataFrame:
    from src.data_loader_cdnow import load_data
    return load_data(path)


register_dataset(
    name="uci",
    display="UCI Online Retail",
    data_path=os.path.join(_ROOT, "data", "raw", "Online Retail.xlsx"),
    loader_fn=_load_uci,
    snapshot_fn=_snapshot_uci,
)

register_dataset(
    name="tafeng",
    display="Ta Feng Grocery",
    data_path=os.path.join(_ROOT, "data", "raw", "ta_feng_all_months_merged.csv"),
    loader_fn=_load_tafeng,
    snapshot_fn=_snapshot_tafeng,
)

register_dataset(
    name="cdnow",
    display="CDNOW Music",
    data_path=os.path.join(_ROOT, "data", "raw", "cdnow.csv"),
    loader_fn=_load_cdnow,
    snapshot_fn=_default_snapshot,   # max + 1 day
)
