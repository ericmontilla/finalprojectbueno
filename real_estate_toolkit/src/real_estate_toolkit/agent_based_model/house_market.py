from typing import List, Optional
from .houses import House
from enum import Enum

class HousingMarket:
    def __init__(self, houses: List[House]):
        """
        Initializes the HousingMarket with a list of houses.

        Args:
            houses (List[House]): A list of House objects to initialize the market.
        """
        self.houses: List[House] = houses

    def get_house_by_id(self, house_id: int) -> Optional[House]:
        """
        Retrieve a specific house by its ID.

        Args:
            house_id (int): The ID of the house to retrieve.

        Returns:
            Optional[House]: The house with the given ID, or None if not found.
        """
        for house in self.houses:
            if house.id == house_id:
                return house
        return None

    def calculate_average_price(self, bedrooms: Optional[int] = None) -> float:
        """
        Calculate the average price of houses, optionally filtered by the number of bedrooms.

        Args:
            bedrooms (Optional[int]): Number of bedrooms to filter by (if specified).

        Returns:
            float: The average price of the filtered houses. Returns 0.0 if no houses match.
        """
        filtered_houses = (
            [house for house in self.houses if house.bedrooms == bedrooms]
            if bedrooms is not None
            else self.houses
        )

        if not filtered_houses:
            return 0.0
        
        total_price = sum(house.price for house in filtered_houses)
        return total_price / len(filtered_houses)

    def get_houses_that_meet_requirements(self, max_price: int, segment: str) -> List[House]:
        """
        Filter houses based on buyer requirements, such as price and segment.

        Args:
            max_price (int): Maximum price the buyer is willing to pay.
            segment (str): The target segment of the buyer (e.g., 'FANCY', 'OPTIMIZER', 'AVERAGE').

        Returns:
            List[House]: A list of houses that meet the buyer's requirements, or an empty list if no matches.
        """
        print(f"Filtering houses for max_price={max_price} and segment={segment}")

        filtered_houses = []
        for house in self.houses:
            print(f"Checking House ID {house.id}: Price {house.price}, Available {house.available}, Quality: {house.quality_score}")

            # Price and availability filter
            if house.price <= max_price and house.available:
                print(f"House ID {house.id} passes price and availability filters.")

                # Check segment's quality score requirement
                quality_score_threshold = self._get_quality_score_threshold(segment)
                print(f"Quality score threshold for {segment} segment: {quality_score_threshold}")
                if house.quality_score:
                    print(f"House ID {house.id}: Quality Score {house.quality_score.value}")
                    if house.quality_score.value >= quality_score_threshold:
                        print(f"House ID {house.id} meets the quality score requirement for segment {segment}.")
                        filtered_houses.append(house)
                    else:
                        print(f"House ID {house.id} does not meet the quality score requirement for {segment}.")
                else:
                    print(f"House ID {house.id} does not have a quality score, assuming it meets the requirement.")
                    filtered_houses.append(house)
            else:
                print(f"House ID {house.id} does not meet price or availability criteria.")

        print(f"Found {len(filtered_houses)} houses that meet the criteria.")
        
        # Debugging: Print details of the filtered houses
        for house in filtered_houses:
            print(f"House ID: {house.id}, Price: {house.price}, Quality: {house.quality_score}, Available: {house.available}")

        return filtered_houses


    def _get_quality_score_threshold(self, segment: str) -> int:
           """
           Map the buyer's segment to the minimum required quality score.


           Args:
               segment (str): The buyer's segment (e.g., 'FANCY', 'OPTIMIZER', 'AVERAGE').


           Returns:
               int: The quality score threshold for the segment.
           """
           quality_score_threshold = {
               'FANCY': 4,  # FANCY buyers require high-quality houses (score >= 4)
               'OPTIMIZER': 3,  # OPTIMIZER buyers focus on price per square foot, not quality
               'AVERAGE': 2  # AVERAGE buyers consider average houses
           }


           return quality_score_threshold.get(segment, 2)  # Default to 2 (AVERAGE) if the segment is unknown


