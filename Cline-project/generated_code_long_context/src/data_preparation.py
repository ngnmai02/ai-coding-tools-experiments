from sklearn.model_selection import train_test_split

def prepare_data(df):
    """
    Split the dataset into training and testing sets with 80/20 ratio.

    Args:
        df (pd.DataFrame): The dataset

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test
