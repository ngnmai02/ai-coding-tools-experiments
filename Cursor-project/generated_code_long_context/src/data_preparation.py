"""Data preparation module for train-test splitting.

This module handles:
- Splitting the dataset into training (80%) and testing (20%) sets
"""

from sklearn.model_selection import train_test_split
import pandas as pd


def split_data(df: pd.DataFrame, target_column: str = 'MedHouseVal', test_size: float = 0.2, random_state: int = 42):
    """
    Split the dataset into training and testing sets.
    
    Args:
        df: Input DataFrame
        target_column: Name of the target column
        test_size: Proportion of dataset to include in the test split (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test) - Features and targets for train and test sets
    """
    print("\n" + "="*60)
    print("DATA PREPARATION")
    print("="*60)
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state
    )
    
    print(f"\nTraining set: {X_train.shape[0]} samples ({X_train.shape[0]/len(df)*100:.1f}%)")
    print(f"Testing set: {X_test.shape[0]} samples ({X_test.shape[0]/len(df)*100:.1f}%)")
    print(f"Features: {X_train.shape[1]}")
    print("="*60 + "\n")
    
    return X_train, X_test, y_train, y_test
