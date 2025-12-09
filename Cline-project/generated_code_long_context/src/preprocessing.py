import numpy as np

def preprocess_data(X_train, X_test):
    """
    Apply preprocessing steps: log transformation to skewed features.

    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Testing features

    Returns:
        tuple: Preprocessed X_train, X_test
    """
    # Apply log transformation to features that are skewed to the right
    skewed_features = ['AveRooms', 'AveBedrms', 'Population', 'AveOccup']

    for feature in skewed_features:
        if feature in X_train.columns:
            X_train[feature] = np.log1p(X_train[feature])
            X_test[feature] = np.log1p(X_test[feature])
            print(f"Applied log transformation to {feature}")
        else:
            print(f"Feature {feature} not found in dataset")

    # Note: Ocean proximity transformation skipped as it's not in California Housing dataset

    return X_train, X_test
