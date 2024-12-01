from dataclasses import dataclass
from typing import Any, Dict, List
import re

@dataclass
class Cleaner:
    """Class for cleaning real estate data."""
    data: List[Dict[str, Any]]

    def rename_with_best_practices(self) -> None:
        """Rename columns with best practices (e.g., snake_case)."""
        if self.data:
            first_row = self.data[0]
            new_keys = {key: re.sub(r'(?<!^)(?=[A-Z])', '_', key).lower()for key in first_row.keys()}
            for row in self.data:
                for old_key, new_key in new_keys.items():
                    row[new_key] = row.pop(old_key)

    def na_to_none(self) -> List[Dict[str, Any]]:
        """Replace 'NA' with None in the dataset."""
        for row in self.data:
            for key, value in row.items():
                if value == "NA":
                    row[key] = None
        return self.data