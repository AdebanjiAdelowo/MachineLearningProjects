# Machine Learning Projects

A set of classic supervised-learning exercises in Python (scikit-learn), each in its own notebook.

## Contents

- `SimpleLinearRegressionModel.ipynb`: linear regression on a 10-row toy salary-vs-experience dataset. Introductory exercise, no held-out evaluation of consequence given the dataset size.
- `HousePricePredictionUsingBoston_Dataset.ipynb`: house price prediction on the Boston housing dataset (a standard pedagogical dataset), comparing linear regression, decision tree, and random forest regressors via 10-fold cross-validation, followed by grid-search hyperparameter tuning of the random forest. Cross-validated R² scores: 0.725 (linear regression), 0.773 (decision tree), 0.873 (random forest); best grid-search R² of 0.882; R² of 0.673 on the held-out test split with the tuned model.
- `Co2Emissions_MachineLearning.ipynb`: CO2 emissions prediction from vehicle fuel-consumption data using linear regression. Single-feature (engine size) model: MSE 1177.13, test R² 0.72. Multi-feature model (engine size, cylinders, fuel-consumption metrics): test R² 0.89.
- `SurvivalPredictionUsingRandomForest.ipynb`: Titanic-style passenger survival prediction with feature engineering (age/fare binning, cabin/name parsing, label encoding) and a random-forest classifier tuned via grid search. Held-out accuracy: 0.81; 10-fold cross-validated accuracy: 0.82.
- `FakeNews_detection_Machine_Learning.ipynb`: binary fake/real news classification on a labelled news dataset (6,335 articles) using TF-IDF features and a Passive-Aggressive classifier. Test accuracy: 92.74%, F1 score: 92.77%.

## Requirements

Python 3 with pandas, numpy, scikit-learn, matplotlib, and seaborn.

## Usage

Notebooks were developed in Google Colab and expect datasets to be uploaded or fetched at runtime (see the upload/URL cells at the top of each notebook).
