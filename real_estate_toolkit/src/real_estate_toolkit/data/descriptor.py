from dataclasses import dataclass
from typing import Any, Dict, List, Union, Tuple, Literal, Optional

@dataclass
class Descriptor:
    """Class for describing real estate data without NumPy or statistics."""
    data: List[Dict[str, Any]]

    def none_ratio(self, columns: Union[List[str], Literal["all"]] = "all") -> Dict[str, float]:
        """Compute the ratio of None values per column."""
        if columns == "all":
            columns = list(self.data[0].keys())
        result: Dict[str, float] = {}
        for col in columns:
            total = len(self.data)
            none_count = sum(1 for row in self.data if row.get(col) is None)
            result[col] = none_count / total if total > 0 else 0
        return result

    def average(self, columns: Union[List[str], Literal["all"]] = "all") -> Dict[str, Optional[float]]:
        """Compute the average for numeric variables."""
        if columns == "all":
            columns = [col for col in self.data[0].keys() if isinstance(self.data[0][col], (int, float))]
        result: Dict[str, Optional[float]] = {}
        for col in columns:
            values = [row[col] for row in self.data if isinstance(row.get(col), (int, float))]
            result[col] = sum(values) / len(values) if values else None
        return result

    def median(self, columns: Union[List[str], Literal["all"]] = "all") -> Dict[str, Optional[float]]:
        """Compute the median for numeric variables."""
        if columns == "all":
            columns = [col for col in self.data[0].keys() if isinstance(self.data[0][col], (int, float))]
        result: Dict[str, Optional[float]] = {}
        for col in columns:
            values = sorted(row[col] for row in self.data if isinstance(row.get(col), (int, float)))
            n = len(values)
            if n == 0:
                result[col] = None
            elif n % 2 == 1:
                result[col] = values[n // 2]
            else:
                result[col] = (values[n // 2 - 1] + values[n // 2]) / 2
        return result

    def percentile(self, columns: Union[List[str], Literal["all"]] = "all", percentile: int = 50) -> Dict[str, Optional[float]]:
        """Compute a specific percentile for numeric variables."""
        if columns == "all":
            columns = [col for col in self.data[0].keys() if isinstance(self.data[0][col], (int, float))]
        result: Dict[str, Optional[float]] = {}
        for col in columns:
            values = sorted(row[col] for row in self.data if isinstance(row.get(col), (int, float)))
            if not values:
                result[col] = None
                continue
            k = (len(values) - 1) * (percentile / 100)
            lower_index = int(k)
            upper_index = min(lower_index + 1, len(values) - 1)
            weight = k - lower_index
            result[col] = (1 - weight) * values[lower_index] + weight * values[upper_index]
        return result

    def type_and_mode(self, columns: Union[List[str], Literal["all"]] = "all") -> Dict[str, Union[Tuple[str, Any], None]]:
        """Compute the mode and type for variables."""
        if columns == "all":
            columns = list(self.data[0].keys())
        result: Dict[str, Union[Tuple[str, Any], None]] = {}
        for col in columns:
            values = [row[col] for row in self.data if row.get(col) is not None]
            if not values:
                result[col] = None
                continue
            if isinstance(values[0], (int, float)):
                mode_value = max(set(values), key=values.count)
                result[col] = ("numeric", mode_value)
            else:
                mode_value = max(set(values), key=values.count)
                result[col] = ("categorical", mode_value)
        return result



from dataclasses import dataclass
from typing import Any, Dict, List, Union, Tuple, Literal
import numpy as np

@dataclass
class DescriptorNumpy:
    """Class for describing real estate data using NumPy."""
    data: List[Dict[str, Any]]

    def none_ratio(self, columns: Union[List[str], Literal["all"]] = "all") -> Dict[str, float]:
        """Compute the ratio of None values per column."""
        if columns == "all":
            columns = list(self.data[0].keys())
        result: Dict[str, float] = {}
        for col in columns:
            total = len(self.data)
            none_count = sum(1 for row in self.data if row.get(col) is None)
            result[col] = none_count / total if total > 0 else 0
        return result

    def average(self, columns: Union[List[str], Literal["all"]] = "all") -> Dict[str, Union[float, None]]:
        """Compute the average for numeric variables."""
        if columns == "all":
            columns = [col for col in self.data[0].keys() if isinstance(self.data[0][col], (int, float))]
        result: Dict[str, Union[float, None]] = {}
        for col in columns:
            values = np.array([row[col] for row in self.data if isinstance(row.get(col), (int, float))])
            result[col] = values.mean() if len(values) > 0 else None
        return result

    def median(self, columns: Union[List[str], Literal["all"]] = "all") -> Dict[str, Union[float, None]]:
        """Compute the median for numeric variables."""
        if columns == "all":
            columns = [col for col in self.data[0].keys() if isinstance(self.data[0][col], (int, float))]
        result: Dict[str, Union[float, None]] = {}
        for col in columns:
            values = np.array([row[col] for row in self.data if isinstance(row.get(col), (int, float))])
            result[col] = np.median(values).item() if len(values) > 0 else None
        return result

    def percentile(self, columns: Union[List[str], Literal["all"]] = "all", percentile: int = 50) -> Dict[str, Union[float, None]]:
        """Compute a specific percentile for numeric variables."""
        if columns == "all":
            columns = [col for col in self.data[0].keys() if isinstance(self.data[0][col], (int, float))]
        result: Dict[str, Union[float, None]] = {}
        for col in columns:
            values = np.array([row[col] for row in self.data if isinstance(row.get(col), (int, float))])
            result[col] = np.percentile(values, percentile).item() if len(values) > 0 else None
        return result

    def type_and_mode(self, columns: Union[List[str], Literal["all"]] = "all") -> Dict[str, Union[Tuple[str, Any], None]]:
        """Compute the mode and type for variables."""
        if columns == "all":
            columns = list(self.data[0].keys())
        result: Dict[str, Union[Tuple[str, Any], None]] = {}
        for col in columns:
            values = [row[col] for row in self.data if row.get(col) is not None]
            if not values:
                result[col] = None
                continue
            if isinstance(values[0], (int, float)):
                mode_value = max(set(values), key=values.count)
                result[col] = ("numeric", mode_value)
            else:
                mode_value = max(set(values), key=values.count)
                result[col] = ("categorical", mode_value)
        return result






