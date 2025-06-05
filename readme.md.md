# Parkinson's Disease Prediction

This project implements several machine learning models to predict Parkinson's disease status based on biomedical voice measurements. It includes data preprocessing, model training with hyperparameter tuning, ensembling, and performance evaluation.

## Dataset

- The dataset used is "parkinsons_disease.csv".
- The 'name' column is dropped as it is non-informative.
- The target variable is "status" (1 indicates Parkinson's, 0 indicates healthy).

## Features

- Data standardization using "StandardScaler".
- Handling class imbalance using SMOTE (Synthetic Minority Over-sampling Technique).
- Models used:
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (SVM)
  - XGBoost
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - Gradient Boosting
  - AdaBoost
- Hyperparameter tuning using Grid Search with cross-validation.
- Ensemble model using soft voting with tuned classifiers.

## Usage

1. Install required packages listed in "requirements.txt".
2. Place "parkinsons_disease.csv" in the working directory.
3. Run the script "parkinsons_disease_prediction.py" to train models and evaluate performance.
4. The script outputs:
   - Model performance metrics (accuracy, precision, recall, F1-score, ROC-AUC).
   - Visualizations for model comparison.
   - Feature importance plot for Gradient Boosting.
   - Confusion matrix and ROC curve for Gradient Boosting.

## Requirements

- Python 3.7 or higher
- See "requirements.txt" for package dependencies.

## Author

Sri Ranjani