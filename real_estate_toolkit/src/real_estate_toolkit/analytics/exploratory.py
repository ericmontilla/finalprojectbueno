from typing import List, Dict, Any, Optional
import polars as pl
import plotly.express as px
from pathlib import Path
import numpy as np
import plotly.graph_objects as go


class MarketAnalyzer:
    def __init__(self, data_path: str):
        """
        Initialize the analyzer with data from a CSV file.
        
        Args:
            data_path (str): Path to the Ames Housing dataset
        """
        self.real_estate_data = pl.read_csv(data_path, null_values=["NA"])
        self.real_estate_clean_data = None

    def clean_data(self) -> None:
        """
        Perform comprehensive data cleaning.
        """
        data = self.real_estate_data.clone()

        # Step 1: Drop columns with >50% missing values
        missing_threshold = 0.5 * len(data)
        for col in data.columns:
            missing_count = data.select(pl.col(col).is_null().sum()).row(0)[0]
            if missing_count > missing_threshold:
                data = data.drop(col)

        # Step 2: Fill missing values for all columns
        for col in data.columns:
            if data[col].dtype == pl.Utf8:  # String (categorical) columns
                data = data.with_columns(data[col].fill_null("Unknown").alias(col))
            elif data[col].dtype in [pl.Float64, pl.Int64]:  # Numeric columns
                median = data.select(pl.median(col)).row(0)[0]
                data = data.with_columns(data[col].fill_nan(median).alias(col))

        # Step 3: Convert categorical columns to category type
        categorical_columns = [
            "MSZoning", "Street", "LotShape", "LandContour", "Utilities", "LotConfig",
            "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle",
            "RoofStyle", "Exterior1st", "Exterior2nd", "MasVnrType", "Foundation",
            "Heating", "CentralAir", "GarageType", "SaleType", "SaleCondition"
        ]
        for col in categorical_columns:
            if col in data.columns:
                data = data.with_columns(data[col].cast(pl.Categorical).alias(col))

        # Step 4: Remove columns of type 'object' that are not relevant for analysis
        object_columns = [col for col in data.columns if data[col].dtype == pl.Utf8]
        irrelevant_columns = ["Alley", "PoolQC", "Fence", "MiscFeature"]  # Assuming these are not relevant
        for col in irrelevant_columns:
            if col in object_columns:
                object_columns.remove(col)

        data = data.drop(object_columns)  # Remove non-relevant object columns

        # Step 5: Convert categorical variables to dummies (binary variables)
        for col in categorical_columns:
            if col in data.columns:
                categories = data[col].unique().to_list()
                for category in categories:
                    dummy_col = pl.when(pl.col(col) == category).then(1).otherwise(0).alias(f"{col}_{category}")
                    data = data.with_columns(dummy_col)

        self.real_estate_clean_data = data

    def generate_price_distribution_analysis(self) -> pl.DataFrame:
        """
        Analyze sale price distribution using clean data.
        
        Tasks to implement:
        1. Compute basic price statistics and generate another data frame called price_statistics:
            - Mean
            - Median
            - Standard deviation
            - Minimum and maximum prices
        2. Create an interactive histogram of sale prices using Plotly.
        
        Returns:
            - Statistical insights dataframe
        """
        if self.real_estate_clean_data is None:
            raise ValueError("Data must be cleaned before generating analysis.")

        # Extract the 'SalePrice' column
        sale_price_column = self.real_estate_clean_data["SalePrice"]

        # 1. Compute basic price statistics
        price_statistics = {
            "mean": sale_price_column.mean(),
            "median": sale_price_column.median(),
            "std_dev": sale_price_column.std(),
            "min": sale_price_column.min(),
            "max": sale_price_column.max(),
        }

        # Convert the statistics into a polars DataFrame
        price_statistics_df = pl.DataFrame(price_statistics)

        # 2. Create an interactive histogram of sale prices using Plotly
        fig = px.histogram(
            x=sale_price_column.to_list(),  # Convert the polars series to a list for Plotly
            nbins=50,
            labels={"x": "Sale Price", "y": "Frequency"},
            title="Sale Price Distribution"
        )

        # Step to save the plot as an HTML file
        output_dir = Path("src/real_estate_toolkit/analytics/outputs/")
        output_dir.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

        plot_file = output_dir / "sale_price_distribution.html"
        fig.write_html(plot_file)  # Save the figure to an HTML file

        print(f"Histogram saved as: {plot_file}")

        # Return the price statistics dataframe
        return price_statistics_df

    def neighborhood_price_comparison(self) -> pl.DataFrame:
        """
        Create a boxplot comparing house prices across different neighborhoods.

        Tasks to implement:
        1. Group data by neighborhood
        2. Calculate price statistics for each neighborhood
        3. Create Plotly boxplot with:
            - Median prices
            - Price spread
            - Outliers

        Returns:
            - Return neighborhood statistics dataframe
            - Save Plotly figures for neighborhood price comparison in src/real_estate_toolkit/analytics/outputs/ folder.
        """
        if self.real_estate_clean_data is None:
            raise ValueError("Data must be cleaned before generating analysis.")

        # Step 1: Group data by neighborhood and compute statistics
        neighborhood_stats = (
            self.real_estate_clean_data
            .group_by("Neighborhood")
            .agg([
                pl.col("SalePrice").mean().alias("mean_price"),
                pl.col("SalePrice").median().alias("median_price"),
                pl.col("SalePrice").std().alias("std_dev_price"),
                pl.col("SalePrice").min().alias("min_price"),
                pl.col("SalePrice").max().alias("max_price"),
            ])
        )

        # Step 2: Create a boxplot using Plotly
        fig = px.box(
            self.real_estate_clean_data.to_pandas(),  # Convert Polars DataFrame to Pandas for Plotly
            x="Neighborhood",
            y="SalePrice",
            title="Neighborhood Price Comparison",
            labels={"Neighborhood": "Neighborhood", "SalePrice": "Sale Price"},
            color="Neighborhood",
            points="all",  # Show all points, including outliers
        )
        fig.update_layout(
            xaxis_title="Neighborhood",
            yaxis_title="Sale Price",
            showlegend=False
        )

        # Step 3: Save Plotly figure
        output_path = Path("src/real_estate_toolkit/analytics/outputs/neighborhood_price_comparison.html")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path)

        print(f"Boxplot saved as: {output_path}")

        # Return statistics DataFrame
        return neighborhood_stats

    def feature_correlation_heatmap(self, variables: list) -> None:
     """
     Generate a correlation heatmap for variables input using Plotly Express.

     Tasks to implement:
     1. Pass a list of numerical variables
     2. Compute correlation matrix and plot it

     Args:
         variables (List[str]): List of variables to correlate

     Returns:
         Save Plotly figures for correlation heatmap in src/real_estate_toolkit/analytics/outputs/ folder.
     """
     try:
         # Verify that the passed variables are present in the data
         missing_vars = [var for var in variables if var not in self.real_estate_clean_data.columns]
         if missing_vars:
             raise ValueError(f"Missing variables in the dataset: {', '.join(missing_vars)}")

         # Convert Polars DataFrame to Pandas for compatibility with Plotly Express
         data_for_correlation = self.real_estate_clean_data.select(variables).to_pandas()

         # Compute the correlation matrix
         correlation_matrix = data_for_correlation.corr()

         # Create a heatmap using Plotly Express
         fig = px.imshow(
             correlation_matrix,
             title="Correlation Heatmap of Selected Features",
             color_continuous_scale="RdBu",
             zmin=-1,  # Min value for color scale
             zmax=1    # Max value for color scale
         )

         # Step to save the heatmap as an HTML file
         output_path = Path("src/real_estate_toolkit/analytics/outputs/correlation_heatmap.html")
         output_path.parent.mkdir(parents=True, exist_ok=True)
         fig.write_html(output_path)

         print(f"Heatmap saved as: {output_path}")

     except Exception as e:
         print(f"An error occurred: {e}")

    def create_scatter_plots(self) -> dict:
        """
        Create scatter plots exploring relationships between key features.
        
        Scatter plots to create:
        1. House price vs. Total square footage
        2. Sale price vs. Year built
        3. Overall quality vs. Sale price
        
        Tasks to implement:
        - Use Plotly Express for creating scatter plots
        - Add trend lines
        - Include hover information
        - Color-code points based on a categorical variable
        - Save them in src/real_estate_toolkit/analytics/outputs/ folder.
        
        Returns:
            Dictionary of Plotly Figure objects for different scatter plots.
        """
        if self.real_estate_clean_data is None:
            raise ValueError("Data must be cleaned before generating plots.")
        
        # Scatter plot 1: House price vs. Total square footage
        fig1 = px.scatter(
            self.real_estate_clean_data.to_pandas(),  # Convert Polars to Pandas for Plotly
            x="GrLivArea",
            y="SalePrice",
            color="OverallQual",  # Color points by Overall Quality
            title="House Price vs. Total Square Footage",
            labels={"GrLivArea": "Total Square Footage", "SalePrice": "Sale Price", "OverallQual": "Overall Quality"},
            hover_data=["OverallQual"]
        )

        # Add a trendline manually (using Polars)
        corr = self.real_estate_clean_data.select(
            pl.corr("GrLivArea", "SalePrice").alias("correlation")
        ).to_dicts()[0]["correlation"]
        
        # Calculate slope (b) and intercept (a) for the line: y = ax + b
        x = self.real_estate_clean_data["GrLivArea"].to_numpy()
        y = self.real_estate_clean_data["SalePrice"].to_numpy()
        slope, intercept = np.polyfit(x, y, 1)

        # Add trendline to the figure
        trendline_x = np.array([x.min(), x.max()])
        trendline_y = slope * trendline_x + intercept
        fig1.add_traces(go.Scatter(x=trendline_x, y=trendline_y, mode='lines', name="Trendline", line=dict(color='red')))

        # Scatter plot 2: Sale price vs. Year built
        fig2 = px.scatter(
            self.real_estate_clean_data.to_pandas(),
            x="YearBuilt",
            y="SalePrice",
            color="OverallQual",
            title="Sale Price vs. Year Built",
            labels={"YearBuilt": "Year Built", "SalePrice": "Sale Price", "OverallQual": "Overall Quality"},
            hover_data=["OverallQual"]
        )

        # Scatter plot 3: Overall quality vs. Sale price
        fig3 = px.scatter(
            self.real_estate_clean_data.to_pandas(),
            x="OverallQual",
            y="SalePrice",
            color="OverallQual",
            title="Overall Quality vs. Sale Price",
            labels={"OverallQual": "Overall Quality", "SalePrice": "Sale Price"},
            hover_data=["OverallQual"]
        )

        # Save figures as HTML files
        output_dir = Path("src/real_estate_toolkit/analytics/outputs/")
        output_dir.mkdir(parents=True, exist_ok=True)

        fig1.write_html(output_dir / "Sale_price_vs_GrLivArea.html")
        fig2.write_html(output_dir / "Sale_price_vs_Year_built.html")
        fig3.write_html(output_dir / "Sale_price_vs_Overall_quality.html")

        # Return dictionary of all plots
        return {
            "scatter_plot_1": fig1,
            "scatter_plot_2": fig2,
            "scatter_plot_3": fig3
        }

if __name__ == "__main__":
    data_path = Path("src/real_estate_toolkit/data/input/train.csv")  # Adjust the path to your CSV file
    analyzer = MarketAnalyzer(data_path=data_path)
    print("Cleaning data...")
    analyzer.clean_data()
    print("Data cleaning complete.")

    # Generate price distribution analysis and show the table
    result = analyzer.generate_price_distribution_analysis()
    print("Price distribution analysis:")
    print(result)
    
    # Generate neighborhood price comparison
    print("Generating neighborhood price comparison...")
    neighborhood_stats = analyzer.neighborhood_price_comparison()
    print("\nNeighborhood price statistics:")
    print(neighborhood_stats)
    print("\nTo visualize the boxplot, open the HTML file saved at:")
    print("src/real_estate_toolkit/analytics/outputs/neighborhood_price_comparison.html")
    
    # Test correlation heatmap
    print("Testing feature correlation heatmap...")
    try:
        numerical_variables = ["SalePrice", "GrLivArea", "YearBuilt", "OverallQual"]
        analyzer.feature_correlation_heatmap(variables=numerical_variables)
        print("Feature correlation heatmap passed!")
    except Exception as e:
        print(f"Feature correlation heatmap failed: {e}")
    
    # Test scatter plots
    print("Testing scatter plots...")
    try:
        scatter_plots = analyzer.create_scatter_plots()
        print("Scatter plots created successfully!")
    except Exception as e:
        print(f"Scatter plot creation failed: {e}")
    
    # Verify scatter plots
    try:
        scatter_plots = analyzer.create_scatter_plots()
        assert isinstance(scatter_plots, dict), "Scatter plots should be returned as a dictionary of Plotly figures."
        assert all(isinstance(fig, go.Figure) for fig in scatter_plots.values()), "All scatter plot values should be Plotly figures."
        print("Scatter plots passed!")
    except Exception as e:
        print(f"Scatter plots failed: {e}")
