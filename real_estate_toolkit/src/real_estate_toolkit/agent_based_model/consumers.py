from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, List, Dict
from .houses import House
from .house_market import HousingMarket


class Segment(Enum):
    FANCY = auto()  # Prefers new construction with high quality scores
    OPTIMIZER = auto()  # Focuses on price per square foot value
    AVERAGE = auto()  # Considers average market prices


@dataclass
class Consumer:
    id: int
    annual_income: float
    children_number: int
    segment: Segment
    house: Optional[House] = None
    savings: float = 0.0
    saving_rate: float = 0.3
    interest_rate: float = 0.05

    def compute_savings(self, years: int) -> None:
        """
        Calculate accumulated savings over time using compound interest.

        Args:
            years (int): Number of years to calculate savings for.
        """
        annual_savings = self.annual_income * self.saving_rate
        self.savings = annual_savings * ((1 + self.interest_rate) ** years - 1) / self.interest_rate

    def buy_a_house(self, housing_market: HousingMarket) -> None:
        """
        Attempt to purchase a suitable house based on segment preferences and savings.
        """
        # Step 1: Filter houses based on segment preferences
        if self.segment == Segment.FANCY:
            suitable_houses = [
                house for house in housing_market.houses
                if house.is_new_construction() and house.quality_score and house.quality_score.value >= 4 and house.available
            ]
        elif self.segment == Segment.OPTIMIZER:
            suitable_houses = [
                house for house in housing_market.houses
                if house.calculate_price_per_square_foot() <= (self.annual_income / 12) and house.available
            ]
        elif self.segment == Segment.AVERAGE:
            average_price = housing_market.calculate_average_price()
            suitable_houses = [
                house for house in housing_market.houses
                if house.price <= average_price and house.available
            ]
        else:
            suitable_houses = []

        # Debugging: print number of suitable houses
        print(f"Consumer {self.id}: Found {len(suitable_houses)} suitable houses based on segment preferences.")

        # Step 2: Filter further based on affordability
        suitable_houses = [
            house for house in suitable_houses if self.savings >= house.price
        ]

        # Debugging: print number of affordable houses
        print(f"Consumer {self.id}: Found {len(suitable_houses)} affordable houses based on savings.")

        # Step 3: Attempt to purchase the best house
        if suitable_houses:
            house_to_buy = min(suitable_houses, key=lambda house: house.price)
            self.house = house_to_buy
            self.savings -= house_to_buy.price  # Deduct the house price from savings
            house_to_buy.available = False  # Mark the house as unavailable after purchase
            print(f"Consumer {self.id} purchased house {house_to_buy.id} for ${house_to_buy.price}!")
        else:
            self.house = None  # No house was purchased
            print(f"Consumer {self.id} could not find a suitable house.")
