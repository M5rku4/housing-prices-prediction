# Housing Prices Prediction

A complete machine learning pipeline to predict housing prices based on property features.

## What this project does

The notebook `housing_prices_analysis.ipynb` walks through the following steps:

1. **Exploratory Data Analysis (EDA)** – Overview of the dataset, data types, missing values, and distributions
2. **Feature Visualization** – Scatter plots, boxplots, and a correlation heatmap to understand how each feature relates to the price
3. **Feature Selection** – Correlation-based analysis to identify and remove low-value features (`hotwaterheating`)
4. **Data Cleaning & Encoding** – Binary and ordinal encoding of categorical variables, outlier removal via IQR, and log-transformation of the target variable
5. **Model Comparison** – Training and evaluating six models: Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, and Gradient Boosting using 5-fold cross-validation
6. **Best Model Evaluation** – In-depth analysis of the best model (Gradient Boosting) including actual vs. predicted plots, residual analysis, feature importance, and a learning curve

## Results

Gradient Boosting achieved the best performance with an R² of ~0.60 on the test set. The most important features were `area`, `bathrooms`, `airconditioning`, and `stories`.

## Dataset

The data comes from Kaggle:
[Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset/data) by Yasser H.

It contains 545 records of residential properties with 13 features including area, number of bedrooms, bathrooms, stories, and various binary amenity flags.

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install via:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```
