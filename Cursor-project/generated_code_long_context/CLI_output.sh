================================================================================
CALIFORNIA HOUSING PRICE PREDICTION - ML PIPELINE
================================================================================

Loading California Housing dataset...
Dataset loaded: 20640 samples, 9 features
Available columns: ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude', 'MedHouseVal']

============================================================
DATA EXPLORATION
============================================================

1. Dataset Shape:
   Rows: 20640, Columns: 9

2. Feature Information:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 20640 entries, 0 to 20639
Data columns (total 9 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   MedInc       20640 non-null  float64
 1   HouseAge     20640 non-null  float64
 2   AveRooms     20640 non-null  float64
 3   AveBedrms    20640 non-null  float64
 4   Population   20640 non-null  float64
 5   AveOccup     20640 non-null  float64
 6   Latitude     20640 non-null  float64
 7   Longitude    20640 non-null  float64
 8   MedHouseVal  20640 non-null  float64
dtypes: float64(9)
memory usage: 1.4 MB
None

3. Feature Ranges (Min-Max):
   MedInc: [0.4999, 15.0001]
   HouseAge: [1.0000, 52.0000]
   AveRooms: [0.8462, 141.9091]
   AveBedrms: [0.3333, 34.0667]
   Population: [3.0000, 35682.0000]
   AveOccup: [0.6923, 1243.3333]
   Latitude: [32.5400, 41.9500]
   Longitude: [-124.3500, -114.3100]
   MedHouseVal: [0.1500, 5.0000]

4. Missing Data Check:
   No missing data found!

5. Data Type Check:
MedInc         float64
HouseAge       float64
AveRooms       float64
AveBedrms      float64
Population     float64
AveOccup       float64
Latitude       float64
Longitude      float64
MedHouseVal    float64
dtype: object

6. Data Type Conversion:

============================================================
Final dataset shape: (20640, 9)
============================================================


============================================================
DATA PREPARATION
============================================================

Training set: 16512 samples (80.0%)
Testing set: 4128 samples (20.0%)
Features: 8
============================================================


============================================================
DATA VISUALIZATION - Feature Distributions
============================================================

Feature distributions plot saved to outputs/plots/feature_distributions.png

============================================================
DATA VISUALIZATION - Correlation Map
============================================================

Correlation map saved to outputs/plots/correlation_map.png

Strong correlations (|r| > 0.5):
   MedInc <-> MedHouseVal: 0.688
   AveRooms <-> AveBedrms: 0.848
   Latitude <-> Longitude: -0.925

============================================================


============================================================
PREPROCESSING - Log Transformation
============================================================
   Applied log transformation to: AveRooms
   Applied log transformation to: AveBedrms
   Applied log transformation to: Population
   Applied log transformation to: AveOccup
============================================================


============================================================
PREPROCESSING - One-Hot Encoding
============================================================
   Warning: 'ocean_proximity' column not found

============================================================
PREPROCESSING - Log Transformation
============================================================
   Applied log transformation to: AveRooms
   Applied log transformation to: AveBedrms
   Applied log transformation to: Population
   Applied log transformation to: AveOccup
============================================================


============================================================
PREPROCESSING - One-Hot Encoding
============================================================
   Warning: 'ocean_proximity' column not found

============================================================
DATA VISUALIZATION - Feature Distributions
============================================================

Feature distributions plot saved to outputs/plots_post/feature_distributions.png

============================================================
DATA VISUALIZATION - Correlation Map
============================================================

Correlation map saved to outputs/plots_post/correlation_map.png

Strong correlations (|r| > 0.5):
   MedInc <-> AveRooms: 0.548
   AveRooms <-> AveBedrms: 0.516
   Latitude <-> Longitude: -0.924

============================================================


================================================================================
MODEL TRAINING
================================================================================

1. Training Baseline Model...
   Baseline model fitted. Mean value: 2.0719
Model saved to outputs/models/baseline_model.pkl

2. Training Linear Regression Model...

============================================================
PREPROCESSING - Data Normalization
============================================================
   Data normalized using StandardScaler
============================================================

   Linear regression model fitted
Model saved to outputs/models/linear_regression_model.pkl
Model saved to outputs/models/linear_regression_scaler.pkl

3. Training ANN Model (ReLU activation)...

============================================================
PREPROCESSING - Data Normalization
============================================================
   Data normalized using StandardScaler
============================================================

   Epoch 20/100 - Train Loss: 0.2556
   Epoch 40/100 - Train Loss: 0.2306
   Epoch 60/100 - Train Loss: 0.2145
   Epoch 80/100 - Train Loss: 0.2054
   Epoch 100/100 - Train Loss: 0.1982
   ANN model trained for 100 epochs
Model saved to outputs/models/ann_model.pkl
Model saved to outputs/models/ann_scaler.pkl

================================================================================
MODEL EVALUATION RESULTS
================================================================================

Model                     Train R²     Test R²      Train MSE       Test MSE       
--------------------------------------------------------------------------------
Baseline Model                0.0000     -0.0002         1.3368         1.3107
Linear Regression             0.6614      0.6437         0.4527         0.4669
ANN (ReLU)                    0.8603      0.7922         0.1868         0.2722

================================================================================
DETAILED METRICS
================================================================================

Baseline Model:
  Training Set:
    R² Score:  0.000000
    MSE:       1.336778
    RMSE:      1.156191
  Testing Set:
    R² Score:  -0.000219
    MSE:       1.310696
    RMSE:      1.144856

Linear Regression:
  Training Set:
    R² Score:  0.661373
    MSE:       0.452669
    RMSE:      0.672807
  Testing Set:
    R² Score:  0.643734
    MSE:       0.466854
    RMSE:      0.683267

ANN (ReLU):
  Training Set:
    R² Score:  0.860264
    MSE:       0.186797
    RMSE:      0.432200
  Testing Set:
    R² Score:  0.792245
    MSE:       0.272244
    RMSE:      0.521770

================================================================================


================================================================================
PIPELINE SUMMARY
================================================================================

✓ Data exploration completed
✓ Data split into training (80%) and testing (20%) sets
✓ Data visualizations saved to outputs/plots/
✓ Preprocessing applied (log transformation, one-hot encoding)
✓ All models trained:
  - Baseline Model
  - Linear Regression
  - ANN with ReLU activation
✓ All models evaluated (R² and MSE metrics)
✓ Model parameters saved to outputs/models/

================================================================================

Target Achievement Check (R² > 0.8):
  Baseline Model: ✗ Not achieved (Test R² = -0.0002)
  Linear Regression: ✗ Not achieved (Test R² = 0.6437)
  ANN (ReLU): ✗ Not achieved (Test R² = 0.7922)

================================================================================
