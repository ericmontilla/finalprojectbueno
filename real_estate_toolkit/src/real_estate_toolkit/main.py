from pathlib import Path
import numpy as np
import polars as pl
from typing import List, Dict, Any
import pytest

from real_estate_toolkit.data.loader import DataLoader
from real_estate_toolkit.data.cleaner import Cleaner
from real_estate_toolkit.data.descriptor import Descriptor, DescriptorNumpy
from real_estate_toolkit.agent_based_model.houses import House, QualityScore
from real_estate_toolkit.agent_based_model.house_market import HousingMarket
from real_estate_toolkit.agent_based_model.consumers import Consumer, Segment
from real_estate_toolkit.agent_based_model.simulation import (
    Simulation, 
    CleaningMarketMechanism, 
    AnnualIncomeStatistics,
    ChildrenRange
)
#from real_estate_toolkit.ml_models.predictor import HousePricePredictor
#from real_estate_toolkit.analytics.exploratory import MarketAnalyzer

def test_data_loading_and_cleaning():
    """Test data loading and cleaning functionality"""
    # Test data loading
    data_path = Path('C:/Users/emb69/OneDrive/Escritorio/real_estate_toolkit/src/real_estate_toolkit/data/input/train.csv')
    loader = DataLoader(data_path)
    
    # Test column validation
    required_columns = ["Id", "SalePrice", "LotArea", "YearBuilt", "BedroomAbvGr"]
    assert loader.validate_columns(required_columns), "Required columns missing from dataset"
    
    # Load and test data format
    data = loader.load_data_from_csv()
    assert isinstance(data, list), "Data should be returned as a list"
    assert all(isinstance(row, dict) for row in data), "Each row should be a dictionary"
    
    # Test data cleaning
    cleaner = Cleaner(data)
    cleaner.rename_with_best_practices()
    cleaned_data = cleaner.na_to_none()
    
    # Verify cleaning results
    assert all((key.islower() and "_") or (key.islower()) in key for key in cleaned_data[0].keys()), "Column names should be in snake_case"
    assert all(val is None or isinstance(val, (str, int, float)) for row in cleaned_data for val in row.values()), \
        "Values should be None or basic types"
    
    return cleaned_data

def test_descriptive_statistics(cleaned_data: List[Dict[str, Any]]):
    """Test descriptive statistics functionality"""
    descriptor = Descriptor(cleaned_data)
    descriptor_numpy = DescriptorNumpy(cleaned_data)
    
    # Test none ratio calculation
    none_ratios = descriptor.none_ratio()
    none_ratios_numpy = descriptor_numpy.none_ratio()
    assert isinstance(none_ratios, dict), "None ratios should be returned as dictionary"
    assert set(none_ratios.keys()) == set(none_ratios_numpy.keys()), "Both implementations should handle same columns"
    
    # Test numeric calculations
    numeric_columns = ["sale_price", "lot_area"]  # Assuming these are the cleaned names
    averages = descriptor.average(numeric_columns)
    medians = descriptor.median(numeric_columns)
    percentiles = descriptor.percentile(numeric_columns, 75)
    
    # Test numpy implementation
    averages_numpy = descriptor_numpy.average(numeric_columns)
    medians_numpy = descriptor_numpy.median(numeric_columns)
    percentiles_numpy = descriptor_numpy.percentile(numeric_columns, 75)
    
    # Compare results
    for col in numeric_columns:
        assert abs(averages[col] - averages_numpy[col]) < 1e-6, f"Average calculations differ for {col}"
        assert abs(medians[col] - medians_numpy[col]) < 1e-6, f"Median calculations differ for {col}"
        assert abs(percentiles[col] - percentiles_numpy[col]) < 1e-6, f"Percentile calculations differ for {col}"
    
    # Test type and mode
    type_modes = descriptor.type_and_mode()
    type_modes_numpy = descriptor_numpy.type_and_mode()
    assert set(type_modes.keys()) == set(type_modes_numpy.keys()), "Both implementations should handle same columns"
    
    return numeric_columns

def test_house_functionality():
    """Test House class implementation"""
    house = House(
        id=1,
        price=200000.0,
        area=2000.0,
        bedrooms=3,
        year_built=2010,
        quality_score=QualityScore.GOOD,
        available=True
    )
    
    # Test basic calculations
    price_per_sqft = house.calculate_price_per_square_foot()
    assert isinstance(price_per_sqft, float), "Price per square foot should be float"
    assert price_per_sqft == 100.0, "Incorrect price per square foot calculation"
    
    # Test new construction logic
    assert house.is_new_construction(2024) is False, "House should not be considered new construction"
    assert house.is_new_construction(2012) is True, "House should be considered new construction"
    
    # Test quality score generation
    house.quality_score = None
    house.get_quality_score()
    assert house.quality_score is not None, "Quality score should be generated"
    
    # Test house sale
    house.sell_house()
    assert house.available is False, "House should be marked as unavailable after sale"
    
    return house

def test_market_functionality(house: House):
    """Test HousingMarket class implementation"""
    # Create market with single house
    market = HousingMarket([house])
    
    # Test house retrieval
    retrieved_house = market.get_house_by_id(1)
    assert retrieved_house == house, "Retrieved house should match original"
    
    # Test average price calculation
    avg_price = market.calculate_average_price(bedrooms=3)
    assert avg_price == 200000.0, "Incorrect average price calculation"
    
    # Test requirements filtering
    matching_houses = market.get_houses_that_meet_requirements(
        max_price=250000,
        segment=Segment.AVERAGE
    )
    assert isinstance(matching_houses, list), "Should return list of matching houses"
    assert len(matching_houses) == 1, "Should find one matching house"
    
    return market

def test_consumer_functionality(market: HousingMarket):
    """Test Consumer class implementation"""
    consumer = Consumer(
        id=1,
        annual_income=80000.0,
        children_number=2,
        segment=Segment.AVERAGE,
        house=None,
        savings=20000.0,
        saving_rate=0.3,
        interest_rate=0.05
    )
    
    # Test savings calculation
    initial_savings = consumer.savings
    consumer.compute_savings(years=5)
    assert consumer.savings > initial_savings, "Savings should increase over time"
    
    # Test house purchase
    consumer.buy_a_house(market)
    assert consumer.house is not None or market.get_houses_that_meet_requirements(
        max_price=consumer.savings * 5,  # Assuming 20% down payment
        segment=consumer.segment
    ) is None, "Consumer should either buy a house or no suitable houses available"
    
    return consumer

def test_simulation(cleaned_data: List[Dict[str, Any]]):
    """Test Simulation class implementation"""
    simulation = Simulation(
        housing_market_data=cleaned_data,
        consumers_number=100,
        years=5,
        annual_income=AnnualIncomeStatistics(
            minimum=30000.0,
            average=60000.0,
            standard_deviation=20000.0,
            maximum=150000.0
        ),
        children_range=ChildrenRange(
            minimum=0,
            maximum=5
        ),
        down_payment_percentage=0.2,
        saving_rate=0.3,
        interest_rate=0.05,
        order=CleaningMarketMechanism.RANDOM
    )
    
    # Test market creation
    simulation.create_housing_market()
    assert hasattr(simulation, 'housing_market'), "Housing market should be created"
    
    # Test consumer creation
    simulation.create_consumers()
    assert hasattr(simulation, 'consumers'), "Consumers should be created"
    assert len(simulation.consumers) == 100, "Should create specified number of consumers"
    
    # Test savings computation
    simulation.compute_consumers_savings()
    assert all(c.savings > 0 for c in simulation.consumers), "All consumers should have savings"
    
    # Test market cleaning
    simulation.clean_the_market()
    
    # Test final statistics
    owners_rate = simulation.compute_owners_population_rate()
    assert 0 <= owners_rate <= 1, "Owners population rate should be between 0 and 1"
    
    availability_rate = simulation.compute_houses_availability_rate()
    assert 0 <= availability_rate <= 1, "Houses availability rate should be between 0 and 1"

# def test_market_analyzer():
#     """Test the functionality of the MarketAnalyzer class."""
    
#     # Path to the dataset
#     data_path = Path("files/ames_housing.csv")

#     # Initialize the analyzer
#     analyzer = MarketAnalyzer(data_path=str(data_path))

#     # Step 1: Test data cleaning
#     print("Testing data cleaning...")
#     try:
#         analyzer.clean_data()
#         assert analyzer.real_state_clean_data is not None, "Data cleaning failed: Cleaned data is None."
#         print("Data cleaning passed!")
#     except Exception as e:
#         print(f"Data cleaning failed: {e}")
#         return

#     # Step 2: Test price distribution analysis
#     print("Testing price distribution analysis...")
#     try:
#         price_stats = analyzer.generate_price_distribution_analysis()
#         assert isinstance(price_stats, pl.DataFrame), "Price distribution stats should be a Polars DataFrame."
#         print("Price distribution analysis passed!")
#     except Exception as e:
#         print(f"Price distribution analysis failed: {e}")
#         return

#     # Step 3: Test neighborhood price comparison
#     print("Testing neighborhood price comparison...")
#     try:
#         neighborhood_stats = analyzer.neighborhood_price_comparison()
#         assert isinstance(neighborhood_stats, pl.DataFrame), "Neighborhood stats should be a Polars DataFrame."
#         print("Neighborhood price comparison passed!")
#     except Exception as e:
#         print(f"Neighborhood price comparison failed: {e}")
#         return

#     # Step 4: Test correlation heatmap
#     print("Testing feature correlation heatmap...")
#     try:
#         numerical_variables = ["SalePrice", "GrLivArea", "YearBuilt", "OverallQual"]
#         analyzer.feature_correlation_heatmap(variables=numerical_variables)
#         print("Feature correlation heatmap passed!")
#     except Exception as e:
#         print(f"Feature correlation heatmap failed: {e}")
#         return

#     # Step 5: Test scatter plots
#     print("Testing scatter plots...")
#     try:
#         scatter_plots = analyzer.create_scatter_plots()
#         assert isinstance(scatter_plots, dict), "Scatter plots should be returned as a dictionary of Plotly figures."
#         assert all(isinstance(fig, go.Figure) for fig in scatter_plots.values()), "All scatter plot values should be Plotly figures."
#         print("Scatter plots passed!")
#     except Exception as e:
#         print(f"Scatter plots failed: {e}")
#         return

# def test_house_price_predictor():
#     """Test the functionality of the HousePricePredictor class."""

#     # Paths to the datasets
#     train_data_path = Path("files/train.csv")
#     test_data_path = Path("files/test.csv")

#     # Initialize predictor
#     predictor = HousePricePredictor(train_data_path=str(train_data_path), test_data_path=str(test_data_path))

#     # Step 1: Test data cleaning
#     print("Testing data cleaning...")
#     try:
#         predictor.clean_data()
#         print("Data cleaning passed!")
#     except Exception as e:
#         print(f"Data cleaning failed: {e}")
#         return

#     # Step 2: Test feature preparation
#     print("Testing feature preparation...")
#     try:
#         predictor.prepare_features(target_column="SalePrice")
#         print("Feature preparation passed!")
#     except Exception as e:
#         print(f"Feature preparation failed: {e}")
#         return

#     # Step 3: Test model training
#     print("Testing model training...")
#     try:
#         results = predictor.train_baseline_models()
#         for model_name, result in results.items():
#             metrics = result["metrics"]
#             print(f"{model_name} - Metrics:")
#             for metric, value in metrics.items():
#                 print(f"  {metric}: {value}")
#         print("Model training passed!")
#     except Exception as e:
#         print(f"Model training failed: {e}")
#         return

#     # Step 4: Test forecasting
#     print("Testing forecasting...")
#     try:
#         predictor.forecast_sales_price(model_type="Linear Regression")
#         print("Forecasting passed!")
#     except Exception as e:
#         print(f"Forecasting failed: {e}")
#         return

def main():
    """Main function to run all tests"""
    try:
        # Run all tests sequentially
        cleaned_data = test_data_loading_and_cleaning()
        test_descriptive_statistics(cleaned_data)
        house = test_house_functionality()
        market = test_market_functionality(house)
        test_consumer_functionality(market)
        test_simulation(cleaned_data)
        # test_market_analyzer()
        # test_house_price_predictor()
        
        print("All tests passed successfully!")
        return 0
    except AssertionError as e:
        print(f"Test failed: {str(e)}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return 2

if __name__ == "__main__":
    exit(main())