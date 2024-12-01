from typing import List, Dict, Any
from pathlib import Path
import sys

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import polars as pl  # Polars for data handling


class HousePricePredictor:
    def __init__(self, train_data_path: str, test_data_path: str):
        """
        Initialize the predictor class with paths to the training and testing datasets.
        """
        try:
            train_path = Path(train_data_path)
            test_path = Path(test_data_path)

            if not train_path.exists():
                raise FileNotFoundError(f"Training file not found: {train_data_path}")
            if not test_path.exists():
                raise FileNotFoundError(f"Test file not found: {test_data_path}")

            # Read test and train datasets
            self.train_data = pl.read_csv(train_path, null_values=["NA"])
            self.test_data = pl.read_csv(test_path, null_values=["NA"])
            self.full_transformer = None
            self.baseline_models = {}
            self.selected_predictors = None  # Save selected columns for consistency
            print("Data loaded successfully.")

        except Exception as e:
            print(f"Error during initialization: {e}")
            raise

    def clean_data(self) -> None:
        """
        Perform comprehensive data cleaning on both the training and testing datasets.
        """
        def clean_dataset(data: pl.DataFrame) -> pl.DataFrame:
            cleaned_data = data.clone()
            missing_threshold = 0.5 * len(cleaned_data)
            for col in cleaned_data.columns:
                if cleaned_data.select(pl.col(col).is_null().sum()).row(0)[0] > missing_threshold:
                    cleaned_data = cleaned_data.drop(col)

            for col in cleaned_data.columns:
                if cleaned_data[col].dtype == pl.Utf8:
                    cleaned_data = cleaned_data.with_columns(cleaned_data[col].fill_null("Unknown").alias(col))
                elif cleaned_data[col].dtype in [pl.Float64, pl.Int64]:
                    median = cleaned_data.select(pl.median(col)).row(0)[0]
                    cleaned_data = cleaned_data.with_columns(cleaned_data[col].fill_nan(median).alias(col))

            irrelevant_columns = ["Alley", "PoolQC", "Fence", "MiscFeature"]
            for col in irrelevant_columns:
                if col in cleaned_data.columns:
                    cleaned_data = cleaned_data.drop(col)

            return cleaned_data

        # Clean datasets
        self.train_data = clean_dataset(self.train_data)
        self.test_data = clean_dataset(self.test_data)
        print("Data cleaned successfully.")

    def prepare_features(self, target_column: str = 'SalePrice', selected_predictors: List[str] = None):
        """
        Prepare the dataset for machine learning by separating features and the target variable.
        """
        try:
            irrelevant_columns = [
                'Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu', 'GarageType',
                'GarageFinish', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                'Street', 'Utilities', 'LandSlope', 'Condition1', 'Condition2', 'ExterCond', 'Functional'
            ]

            # Select prediction columns if not passed as argument
            if selected_predictors is None:
                selected_predictors = [col for col in self.train_data.columns if col != target_column and col not in irrelevant_columns]

            self.selected_predictors = selected_predictors  # Save the columns selected for consistency

            X = self.train_data.select(selected_predictors)
            y = self.train_data.select(target_column)

            # Separate the numeric and categorical columns
            numeric_features = [col for col in selected_predictors if self.train_data[col].dtype in [pl.Float64, pl.Int64]]
            categorical_features = [col for col in selected_predictors if self.train_data[col].dtype == pl.Utf8]

            # Create the pipeline for processing
            numeric_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median'))
            ])

            categorical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])

            # Mix the numerical and categorical pipelines
            self.full_transformer = ColumnTransformer(
                transformers=[
                    ('numeric', numeric_pipeline, numeric_features),
                    ('categorical', categorical_pipeline, categorical_features)
                ],
                remainder='passthrough'
            )

            # Divide the dataset into train and test
            X_train, X_test, y_train, y_test = train_test_split(X.to_pandas(), y.to_numpy().ravel(), test_size=0.2, random_state=42)

            print("Features and target separated, preprocessing pipeline created.")
            return X_train, X_test, y_train, y_test

        except Exception as e:
            print(f"Error during feature preparation: {e}")
            raise

    def train_baseline_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Train and evaluate baseline machine learning models for house price prediction.
        """
        try:
            X_train, X_test, y_train, y_test = self.prepare_features(target_column="SalePrice")

            models = {
                "Linear Regression": LinearRegression(),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42)
            }

            results = {}
            for model_name, model in models.items():
                pipeline = Pipeline(steps=[
                    ('preprocessor', self.full_transformer),
                    ('model', model)
                ])

                pipeline.fit(X_train, y_train)
                y_train_pred = pipeline.predict(X_train)
                y_test_pred = pipeline.predict(X_test)

                metrics = {
                    "MSE_Train": mean_squared_error(y_train, y_train_pred),
                    "MSE_Test": mean_squared_error(y_test, y_test_pred),
                    "R2_Train": r2_score(y_train, y_train_pred),
                    "R2_Test": r2_score(y_test, y_test_pred),
                    "MAE_Train": mean_absolute_error(y_train, y_train_pred),
                    "MAE_Test": mean_absolute_error(y_test, y_test_pred),
                    "MAPE_Train": mean_absolute_percentage_error(y_train, y_train_pred),
                    "MAPE_Test": mean_absolute_percentage_error(y_test, y_test_pred),
                }

                results[model_name] = {
                    "model": model,
                    "metrics": metrics
                }

            self.baseline_models = results  # Save models in the baseline_models dictionary

            return results

        except Exception as e:
            print(f"Error during model training: {e}")
            raise

    def forecast_sales_price(self, model_type: str = 'Advanced') -> None:
        """
        Generate house price predictions using the trained models.
        """
        try:
            if model_type == 'LinearRegression':
                model = self.baseline_models['Linear Regression']['model']
            elif model_type == 'Advanced':
                model = self.baseline_models['Gradient Boosting']['model']
            else:
                raise ValueError("Invalid model type. Choose 'LinearRegression' or 'Advanced'.")

            # Prepare test data
            test_columns = self.selected_predictors
            X_test = self.test_data.select(test_columns).to_pandas()

            # Apply the same transformation to the test data
            X_test_transformed = self.full_transformer.transform(X_test)

            # Generate predictions
            predictions = model.predict(X_test_transformed)

            # Create a DataFrame for submission
            submission_df = self.test_data.select(["Id"]).to_pandas()
            submission_df["SalePrice"] = predictions

            output_dir = Path("src/real_estate_toolkit/analytics/outputs")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file_path = output_dir / "submission.csv"
            submission_df.to_csv(output_file_path, index=False)

            print(f"Forecast completed successfully. Predictions saved to: {output_file_path}")

        except KeyError as e:
            print(f"Model not found in baseline_models: {e}")
        except Exception as e:
            print(f"Error during forecasting: {e}")


from pathlib import Path

if __name__ == "__main__":
    # Paths to data files
    train_data_path = Path("src/real_estate_toolkit/data/input/train.csv")
    test_data_path = Path("src/real_estate_toolkit/data/input/test.csv")

    # Check if files exist
    if not train_data_path.exists():
        print(f"Error: Training dataset file not found at {train_data_path}")
        sys.exit(1)
    elif not test_data_path.exists():
        print(f"Error: Testing dataset file not found at {test_data_path}")
        sys.exit(1)
    else:
        try:
            # Initialize predictor
            predictor = HousePricePredictor(str(train_data_path), str(test_data_path))
            predictor.clean_data()
            predictor.prepare_features(target_column="SalePrice")
            results = predictor.train_baseline_models()

            # Display metrics of trained models
            for model_name, result in results.items():
                metrics = result["metrics"]
                print(f"{model_name} - Metrics:")
                for metric, value in metrics.items():
                    print(f"  {metric}: {value}")

            # Testing forecasting with Linear Regression
            print("Testing forecasting...")
            try:
                predictor.forecast_sales_price(model_type="LinearRegression")
                print("Forecasting passed!")
            except Exception as e:
                print(f"Forecasting failed: {e}")

        except Exception as e:
            print(f"Error in the main execution flow: {e}")
