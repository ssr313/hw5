# README

## Machine Learning Homework 5: Random Forest Classification of ICU Patient Survival

This repository contains the code and supplementary materials for the Machine Learning course (Tsinghua University Course 80250993) Homework 5. The task involves classifying patients' survival (0: survived; 1: dead) using 108 features from their Intensive Care Unit (ICU) records.

## Programming Environment

- **Operating System**: Windows 11
- **Python Version**: 3.10
- **Libraries and Versions**:
  - numpy: 1.21.2
  - pandas: 1.3.3
  - scikit-learn: 1.0.2
  - matplotlib: 3.4.3

## Dataset

- **Source**: Kaggle (WIDS Datathon 2020)
- **Dataset Name**: ICU Patient Dataset
- **Number of Features**: 108
- **Number of Samples**: Training Set - 5000, Test Set - 1097
- **Features**: A mixture of numeric and binary variables such as age, BMI, height, weight, heart rate, blood pressure, etc.

## Experiment Setup

The experiment involves training a Random Forest (RF) classifier using various parameter settings to classify patient survival. The goal was to find the optimal parameters that maximize the model's predictive performance on the test set. The following RF configurations were used:

- **Parameter Grid**:
  - `n_estimators`: 100, 200, 300
  - `max_depth`: None, 10, 20, 30
  - `min_samples_split`: 2, 5, 10
  - `min_samples_leaf`: 1, 2, 4

## Files

- `train1_icu_data.csv`: Training set feature data.
- `train1_icu_label.csv`: Training set labels.
- `test1_icu_data.csv`: Test set feature data.
- `test1_icu_label.csv`: Test set labels.
- `random_forest_classification.ipynb`: Jupyter Notebook containing the random forest classification code.
- `experiment_report.pdf`: Detailed report of the experiment observations and analysis.
- `feature_importance.png`: Bar chart displaying the importance of each feature in the trained random forest model.

## Experiment Results

The experiment resulted in identifying the optimal parameters for the random forest model as `max_depth=30`, `min_samples_leaf=4`, `min_samples_split=10`, and `n_estimators=300`. The model achieved an accuracy of 0.7985 on the test set. The feature importance plot identified 'age' and 'bmi' as the most influential features in predicting patient survival.

## Analysis

The analysis of the experiment revealed that the random forest model performed well in classifying patient survival with a high degree of accuracy. The feature importance analysis provided insights into which features were most predictive of patient outcomes, with age and BMI being the most significant. The least important features were related to undefined diagnostic categories, suggesting that these features may not be as relevant for predicting survival in the ICU.

This experiment demonstrates the effectiveness of random forests in handling high-dimensional data and the ability to identify key features that contribute to patient survival in an ICU setting. The results can be used to guide clinical decision-making and resource allocation in healthcare settings.

Please ensure that you have the necessary libraries installed to run the code and reproduce the experiments. If you have any questions or require further information, please feel free to reach out.
