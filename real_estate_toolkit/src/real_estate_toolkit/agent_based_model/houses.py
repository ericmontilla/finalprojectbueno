from enum import Enum
from dataclasses import dataclass
from typing import Optional


class QualityScore(Enum):
    EXCELLENT = 5
    GOOD = 4
    AVERAGE = 3
    FAIR = 2
    POOR = 1


@dataclass
class House:
    id: int
    price: float
    area: float
    bedrooms: int
    year_built: int
    quality_score: Optional[QualityScore] = None
    available: bool = True

    def calculate_price_per_square_foot(self) -> float:
        """
        Calculate and return the price per square foot.

        Returns:
            float: The price per square foot, rounded to 2 decimal places.
        """
        return round(self.price / self.area, 2) if self.area > 0 else 0.0

    def is_new_construction(self, current_year: int = 2024) -> bool:
        """
        Determine if the house is considered new construction (< 5 years old).

        Args:
            current_year (int): The current year. Defaults to 2024.

        Returns:
            bool: True if the house is new construction, False otherwise.
        """
        return (current_year - self.year_built) < 5

    def get_quality_score(self) -> None:
        """
        Generate a quality score based on house attributes.
        If the quality score is not provided, assign a score based on attributes.

        Implementation:
        - Assign scores based on house's age, size, and number of bedrooms.
        """
        if self.quality_score is None:
            age = 2024 - self.year_built
            if age < 5 and self.area > 1500 and self.bedrooms >= 3:
                self.quality_score = QualityScore.EXCELLENT
            elif age < 10 and self.area > 1200:
                self.quality_score = QualityScore.GOOD
            elif age < 20:
                self.quality_score = QualityScore.AVERAGE
            else:
                self.quality_score = QualityScore.FAIR

    def sell_house(self) -> None:
        """
        Mark the house as sold by updating the 'available' status.
        """
        self.available = False
