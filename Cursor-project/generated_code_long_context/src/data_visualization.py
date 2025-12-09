"""Data visualization module for exploring data distributions and correlations.

This module handles:
- Visualizing feature distributions to check skewness
- Plotting correlation maps between features
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os


def visualize_distributions(df: pd.DataFrame, save_dir: str = 'outputs/plots'):
    """
    Visualize the distribution of all numeric features to check skewness.
    
    Args:
        df: Input DataFrame
        save_dir: Directory to save the plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("DATA VISUALIZATION - Feature Distributions")
    print("="*60)
    
    # Get numeric columns (exclude target if present)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if 'MedHouseVal' in numeric_cols:
        numeric_cols = numeric_cols.drop('MedHouseVal')
    
    n_cols = len(numeric_cols)
    n_rows = (n_cols + 2) // 3  # 3 columns per row
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for idx, col in enumerate(numeric_cols):
        ax = axes[idx]
        df[col].hist(bins=50, ax=ax, edgecolor='black')
        ax.set_title(f'{col} Distribution', fontsize=10, fontweight='bold')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        # Calculate and display skewness
        skewness = df[col].skew()
        ax.text(0.7, 0.9, f'Skewness: {skewness:.2f}', 
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Hide empty subplots
    for idx in range(n_cols, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/feature_distributions.png', dpi=300, bbox_inches='tight')
    print(f"\nFeature distributions plot saved to {save_dir}/feature_distributions.png")
    plt.close()


def visualize_correlations(df: pd.DataFrame, save_dir: str = 'outputs/plots'):
    """
    Plot correlation map between features using seaborn.
    
    Args:
        df: Input DataFrame
        save_dir: Directory to save the plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("DATA VISUALIZATION - Correlation Map")
    print("="*60)
    
    # Get numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/correlation_map.png', dpi=300, bbox_inches='tight')
    print(f"\nCorrelation map saved to {save_dir}/correlation_map.png")
    plt.close()
    
    # Print strong correlations
    print("\nStrong correlations (|r| > 0.5):")
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.5:
                print(f"   {corr_matrix.columns[i]} <-> {corr_matrix.columns[j]}: {corr_val:.3f}")


def visualize_all(df: pd.DataFrame, save_dir: str = 'outputs/plots'):
    """
    Run all visualization functions.
    
    Args:
        df: Input DataFrame
        save_dir: Directory to save the plots
    """
    visualize_distributions(df, save_dir)
    visualize_correlations(df, save_dir)
    print("\n" + "="*60 + "\n")
