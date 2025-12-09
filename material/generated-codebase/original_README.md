# A simple model for California House Prices Prediction

## Basic description
We want to build a machine learning model that could predict house prices in California using the scikit-learn dataset California Housing. The model is expected to predict the prices with above 80% accuracy.

## Getting started
Guide to use this repository to recreate the project locally

## Project structure
The project structure should include at lease some folders for utilities and sources.

## Pipeline
- Data exploration -> data preparation -> data visualization -> preprocessing -> feature engineering -> model training -> model evaluation -> testing

## Information regarding each component of the pipeline

### Data exploration
- Checking the features of this dataset and the range.
- Checking feature channel that has missing data and dropping those features to remove the hassle.
- Checking data type to be consistent and doing conversion if needed.

### Data preparation
- Splitting the data set into training set and testing set with the ratio 80 / 20

### Data visualization
- Visualize the feature channels to check the skewness and the data distribution.
- Plot the correlation map between features with seaborn to see the features relations.

### Preprocessing
- Apply log transformation to features: "total_rooms", "total_bedrooms", "population", "households" as they will be too skewed to right side
- Transform the ocean proximity data to one-hot encoding data with 5 columns (<1h Ocean, Inland, Island, Near bay and Near ocean)

### Feature engineering
- Using the existing correlation patterns, create new feature "bedroom_ratio" which is the ratio between "total_bedrooms" and "total_rooms". Also create new feature "household_rooms" which is the ratio between "total_rooms" and "households"

### Model training

#### Baseline model
- Average all training ground truth data to retrieve a mean value
- Assign this mean value as the predicted value for the test set
- Calculated R-Squared score for this baseline

#### Linear regression model
- Normalize the data before training
- Train the linear regression model (using existing library)
- Report R-Squared score for this model
- Save model parameter for future testing

#### NN model
- Normalize the data before training
- Create a class called "ModelTrainer" for the simple ANN, using ReLU as the activation function
- Train the ANN with training data
- Report R-Squared score for this trained model
- Save model parameter for future testing

### Evaluation and testing
- Reporting all the evaluation result of the models (previous calculated R-Squared)
- Calculating Mean Squared Error metric for all trained models with training data
- Calculating Mean Squared Error metric for all models with testing data
- Reporting all these results
