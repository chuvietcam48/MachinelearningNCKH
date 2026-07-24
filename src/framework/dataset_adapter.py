import os
import yaml
import logging
from pathlib import Path
import pandas as pd
from src.dataset_registry import get_dataset

logger = logging.getLogger(__name__)

class DatasetAdapter:
    """
    Layer 1: Dataset Adapter
    Responsible for loading datasets and enforcing a standard schema.
    It reads from config/framework_config.yaml to determine capabilities.
    """
    def __init__(self, dataset_name: str, config_path: str = None):
        self.dataset_name = dataset_name.lower()
        if config_path is None:
            repo_root = Path(__file__).resolve().parent.parent.parent
            config_path = repo_root / "config" / "framework_config.yaml"
            
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
            
        if self.dataset_name not in self.config["datasets"]:
            raise ValueError(f"Dataset {self.dataset_name} not found in framework_config.yaml")
            
        self.dataset_config = self.config["datasets"][self.dataset_name]
        
    def has_semantic_data(self) -> bool:
        return self.dataset_config.get("has_semantic", False)
        
    def get_display_name(self) -> str:
        return self.dataset_config.get("name", self.dataset_name)
        
    def load_data(self) -> pd.DataFrame:
        """
        Loads the raw behavioral dataset using the existing dataset_registry.
        Enforces standard columns: CustomerID (Standardized from dataset_registry).
        """
        logger.info(f"Loading Dataset: {self.get_display_name()}")
        
        # We use the old registry to actually fetch the CSV and standardise basic columns
        registry_info = get_dataset(self.dataset_name)
        df = registry_info.loader_fn(registry_info.data_path)
        
        # Enforce standard behavioral schema checks
        required_cols = ['CustomerID']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Dataset {self.dataset_name} is missing standard behavioral columns: {missing}")
            
        return df
