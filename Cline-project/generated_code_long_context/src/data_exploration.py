import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing

def explore_data():
    """
    Load the California Housing dataset and perform initial exploration.
    - Check features and ranges
    - Check for missing data and drop features with missing values
    - Check data types and perform conversions if needed

    Returns:
        pd.DataFrame: The cleaned dataset
    """
    # Load dataset
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['MedHouseVal'] = data.target  # Add target

    print("Dataset shape:", df.shape)
    print("\nFeature names:", list(df.columns))
    print("\nData types:")
    print(df.dtypes)
    print("\nSummary statistics:")
    print(df.describe())

    # Check for missing values
    missing = df.isnull().sum()
    print("\nMissing values per feature:")
    print(missing)

    # Drop features with missing values
    features_to_drop = missing[missing > 0].index.tolist()
    if features_to_drop:
        df = df.drop(columns=features_to_drop)
        print(f"\nDropped features with missing values: {features_to_drop}")
    else:
        print("\nNo features with missing values found.")

    # Check data types - all should be numeric for this dataset
    # If any conversions needed, but sklearn dataset is clean

    return df
