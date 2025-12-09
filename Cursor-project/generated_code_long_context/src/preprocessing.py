"""Preprocessing module for data transformation.

This module handles:
- Log transformation for skewed features (total_rooms, total_bedrooms, population, households)
- One-hot encoding for ocean_proximity feature
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def apply_log_transformation(df: pd.DataFrame, columns: list = None):
    """
    Apply log transformation to specified columns to handle right-skewed data.
    
    Args:
        df: Input DataFrame
        columns: List of column names to transform. If None, applies to:
                 ['AveRooms', 'AveBedrms', 'Population', 'AveOccup']
        
    Returns:
        pd.DataFrame: DataFrame with log-transformed columns
    """
    if columns is None:
        columns = ['AveRooms', 'AveBedrms', 'Population', 'AveOccup']
    
    df_transformed = df.copy()
    
    print("\n" + "="*60)
    print("PREPROCESSING - Log Transformation")
    print("="*60)
    
    for col in columns:
        if col in df_transformed.columns:
            # Add small epsilon to avoid log(0)
            df_transformed[col] = np.log1p(df_transformed[col])  # log1p(x) = log(1+x)
            print(f"   Applied log transformation to: {col}")
        else:
            print(f"   Warning: Column '{col}' not found in dataframe")
    
    print("="*60 + "\n")
    return df_transformed


def one_hot_encode_ocean_proximity(df: pd.DataFrame):
    """
    Transform ocean_proximity to one-hot encoding with 5 columns.
    Categories: <1H OCEAN, INLAND, ISLAND, NEAR BAY, NEAR OCEAN
    
    Args:
        df: Input DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with one-hot encoded ocean_proximity
    """
    print("\n" + "="*60)
    print("PREPROCESSING - One-Hot Encoding")
    print("="*60)
    
    if 'ocean_proximity' not in df.columns:
        print("   Warning: 'ocean_proximity' column not found")
        return df
    
    df_encoded = df.copy()
    
    # Get unique values
    unique_values = df['ocean_proximity'].unique()
    print(f"   Unique ocean_proximity values: {unique_values}")
    
    # Perform one-hot encoding
    ocean_proximity_dummies = pd.get_dummies(df['ocean_proximity'], prefix='ocean_proximity')
    
    # Drop original column and add encoded columns
    df_encoded = df_encoded.drop(columns=['ocean_proximity'])
    df_encoded = pd.concat([df_encoded, ocean_proximity_dummies], axis=1)
    
    print(f"   Created {len(ocean_proximity_dummies.columns)} one-hot encoded columns")
    print(f"   Columns: {list(ocean_proximity_dummies.columns)}")
    print("="*60 + "\n")
    
    return df_encoded


def preprocess_data(df: pd.DataFrame):
    """
    Apply all preprocessing steps: log transformation and one-hot encoding.
    
    Args:
        df: Input DataFrame
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame
    """
    # Apply log transformation
    df_processed = apply_log_transformation(df)
    
    # Apply one-hot encoding
    df_processed = one_hot_encode_ocean_proximity(df_processed)
    
    return df_processed


def normalize_data(X_train, X_test):
    """
    Normalize the data using StandardScaler (fits on train, transforms both).
    
    Args:
        X_train: Training features
        X_test: Testing features
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler) - Normalized data and scaler object
    """
    print("\n" + "="*60)
    print("PREPROCESSING - Data Normalization")
    print("="*60)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame to preserve column names
    if isinstance(X_train, pd.DataFrame):
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    print("   Data normalized using StandardScaler")
    print("="*60 + "\n")
    
    return X_train_scaled, X_test_scaled, scaler
