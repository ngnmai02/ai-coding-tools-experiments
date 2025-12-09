"""Data exploration module for California Housing dataset.

This module handles:
- Checking dataset features and their ranges
- Identifying and dropping features with missing data
- Checking and converting data types for consistency
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing


def load_data():
    """
    Load the California Housing dataset from scikit-learn.
    Uses the original feature set without creating synthetic columns.
    
    Returns:
        pd.DataFrame: The dataset as a pandas DataFrame
    """
    print("Loading California Housing dataset...")
    california_housing = fetch_california_housing(as_frame=True)
    df = california_housing.frame.copy()
    
    # Add target column
    df['MedHouseVal'] = california_housing.target
    
    print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
    print(f"Available columns: {list(df.columns)}")
    return df


def explore_data(df: pd.DataFrame):
    """
    Explore the dataset: check features, ranges, missing data, and data types.
    
    Args:
        df: Input DataFrame
        
    Returns:
        pd.DataFrame: Cleaned DataFrame after exploration and cleaning
    """
    print("\n" + "="*60)
    print("DATA EXPLORATION")
    print("="*60)
    
    # Display basic information
    print("\n1. Dataset Shape:")
    print(f"   Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    
    # Display feature names and ranges
    print("\n2. Feature Information:")
    print(df.info())
    
    print("\n3. Feature Ranges (Min-Max):")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        print(f"   {col}: [{df[col].min():.4f}, {df[col].max():.4f}]")
    
    # Check for missing data
    print("\n4. Missing Data Check:")
    missing_data = df.isnull().sum()
    features_with_missing = missing_data[missing_data > 0]
    
    if len(features_with_missing) > 0:
        print(f"   Features with missing data: {len(features_with_missing)}")
        for feature, count in features_with_missing.items():
            print(f"   - {feature}: {count} missing values ({count/len(df)*100:.2f}%)")
        
        # Drop features with missing data
        print("\n   Dropping features with missing data...")
        df_cleaned = df.drop(columns=features_with_missing.index)
        print(f"   Dropped columns: {list(features_with_missing.index)}")
    else:
        print("   No missing data found!")
        df_cleaned = df.copy()
    
    # Check data types
    print("\n5. Data Type Check:")
    print(df_cleaned.dtypes)
    
    # Convert data types if needed (ensure numeric columns are numeric)
    print("\n6. Data Type Conversion:")
    for col in df_cleaned.select_dtypes(include=['object']).columns:
        if col != 'ocean_proximity':  # Keep ocean_proximity as categorical
            try:
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
                print(f"   Converted {col} to numeric")
            except:
                pass
    
    print("\n" + "="*60)
    print(f"Final dataset shape: {df_cleaned.shape}")
    print("="*60 + "\n")
    
    return df_cleaned
