import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

def visualize_data(df):
    """
    Visualize the feature distributions and correlations.

    Args:
        df (pd.DataFrame): The dataset
    """
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)

    # Plot histograms for each feature to check skewness and distribution
    df.hist(figsize=(12, 10), bins=30)
    plt.suptitle('Feature Distributions')
    plt.tight_layout()
    plt.savefig('plots/feature_distributions.png')
    plt.close()

    # Correlation heatmap
    corr = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.savefig('plots/correlation_matrix.png')
    plt.close()

def visualize_preprocessed_data(X_train, y_train):
    """
    Visualize the preprocessed training data distributions and correlations.

    Args:
        X_train (pd.DataFrame): Preprocessed training features
        y_train (pd.Series): Training targets
    """
    # Combine X_train and y_train for visualization
    train_df = pd.concat([X_train, y_train], axis=1)

    # Plot histograms for each feature to check distributions after preprocessing
    train_df.hist(figsize=(12, 10), bins=30)
    plt.suptitle('Preprocessed Feature Distributions')
    plt.tight_layout()
    plt.savefig('plots/preprocessed_distributions.png')
    plt.close()

    # Correlation heatmap
    corr = train_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Preprocessed Correlation Matrix')
    plt.savefig('plots/preprocessed_correlation.png')
    plt.close()
