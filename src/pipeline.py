"""
Pipeline utilities for outcome prediction.

This module provides functions and transformers for feature engineering,
preprocessing, and building complete ML pipelines.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, StandardScaler

# Constants for ordinal encoding
CUT_ORDER = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLOUR_ORDER = ["J", "I", "H", "G", "F", "E", "D"]
CLARITY_ORDER = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features from raw data.
    
    Creates volume feature and log-transformed price and carat features, and removes original dimensions and price/carat columns.
    
    Args:
        df: DataFrame with columns x, y, z, price, carat.
        
    Returns:
        DataFrame with new features and original columns removed.
    """
    df = df.copy()
    df["volume"] = df["x"] * df["y"] * df["z"]
    df["log_price"] = np.log1p(df["price"])
    df["log_carat"] = np.log1p(df["carat"])
    return df.drop(columns=["x", "y", "z", "carat", "price"])


feature_engineering_transformer = FunctionTransformer(
    feature_engineering,
    validate=False,
    check_inverse=False
)


def build_preprocessor() -> ColumnTransformer:
    """
    Build a preprocessor for numerical and categorical features.
    
    Standardizes numerical features and applies ordinal encoding to
    categorical features (cut, color, clarity) with predefined orderings.
    
    Returns:
        ColumnTransformer configured for diamond dataset preprocessing.
    """
    categorical_columns = ["cut", "color", "clarity"]
    
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), make_column_selector(dtype_include=np.number)),
            ("cat", OrdinalEncoder(categories=[CUT_ORDER, COLOUR_ORDER, CLARITY_ORDER]), categorical_columns),
        ],
    )


def build_pipeline(model: BaseEstimator) -> Pipeline:
    """
    Build a complete ML pipeline.
    
    Combines feature engineering, preprocessing, and the model into a single pipeline.
    
    Args:
        model: Scikit-learn estimator to use as the final step.
        
    Returns:
        Pipeline with feature engineering, preprocessing, and model steps.
    """
    return Pipeline(steps=[
        ("feature_engineering", feature_engineering_transformer),
        ("preprocessor", build_preprocessor()),
        ("model", model)
    ])