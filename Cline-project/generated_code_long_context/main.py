import sys
sys.path.append('src')

from data_exploration import explore_data
from data_preparation import prepare_data
from data_visualization import visualize_data, visualize_preprocessed_data
from preprocessing import preprocess_data
from model_training import train_models
from evaluation import evaluate_models

if __name__ == "__main__":
    print("Starting ML Pipeline...")

    # Data exploration
    df = explore_data()

    # Data visualization
    visualize_data(df)

    # Data preparation
    X_train, X_test, y_train, y_test = prepare_data(df)

    # Preprocessing
    X_train, X_test = preprocess_data(X_train, X_test)

    # Visualization after preprocessing
    visualize_preprocessed_data(X_train, y_train)

    # Model training
    r2_scores = train_models(X_train, X_test, y_train, y_test)

    # Evaluation
    evaluate_models(X_train, X_test, y_train, y_test, r2_scores)

    print("Pipeline completed!")
