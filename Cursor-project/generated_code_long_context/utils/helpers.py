"""Helper utility functions for data handling and model operations."""

import numpy as np
import pickle
import os


def save_model(model, filepath: str):
    """
    Save a trained model to disk.
    
    Args:
        model: The model object to save
        filepath: Path where the model should be saved
    """
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")


def load_model(filepath: str):
    """
    Load a saved model from disk.
    
    Args:
        filepath: Path to the saved model file
        
    Returns:
        The loaded model object
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model


def ensure_dir(directory: str):
    """Ensure a directory exists, create if it doesn't."""
    os.makedirs(directory, exist_ok=True)


