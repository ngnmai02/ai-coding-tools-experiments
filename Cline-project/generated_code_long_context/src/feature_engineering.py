def engineer_features(X_train, X_test):
    """
    Create new features based on existing correlations.

    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Testing features

    Returns:
        tuple: X_train with new features, X_test with new features
    """
    # Create bedroom_ratio: ratio between AveBedrms and AveRooms
    X_train['bedroom_ratio'] = X_train['AveBedrms'] / X_train['AveRooms']
    X_test['bedroom_ratio'] = X_test['AveBedrms'] / X_test['AveRooms']

    # Create household_rooms: ratio between AveRooms and AveOccup
    X_train['household_rooms'] = X_train['AveRooms'] / X_train['AveOccup']
    X_test['household_rooms'] = X_test['AveRooms'] / X_test['AveOccup']

    print("Added new features: bedroom_ratio, household_rooms")

    return X_train, X_test
