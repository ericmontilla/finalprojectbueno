from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import polars as pl

@dataclass
class DataLoader:
    """Class for loading and basic processing of real estate data."""
    data_path: Path
   
    def load_data_from_csv(self) -> List[Dict[str, Any]]:
        """Load data from CSV file into a list of dictionaries."""
        self.data = pl.read_csv(self.data_path, null_values=["NA"])
        return self.data.to_dicts()

    def validate_columns(self, required_columns: List[str]) -> bool:
        """Validate that all required columns are present in the dataset."""
        data = pl.read_csv(self.data_path, null_values=["NA"])
        existing_columns = set(data.columns)
        return all(col in existing_columns for col in required_columns)

