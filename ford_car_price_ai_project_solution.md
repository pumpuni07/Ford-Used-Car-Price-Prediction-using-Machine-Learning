# Ford Used Car Price Prediction using Machine Learning

## Project Overview

This project focuses on building a predictive machine learning model capable of estimating the optimal resale price of Ford vehicles based on historical sales data. The objective of this analysis is to help a used car dealership generate accurate and data-driven pricing quotations for vehicles in their inventory.

The project combines data cleaning, exploratory data analysis (EDA), visualization, feature analysis, predictive modeling, and hyperparameter optimization techniques to identify the best-performing regression model.

---

# Business Problem

Used car pricing is influenced by several factors such as:

- Vehicle age
- Mileage
- Engine size
- Fuel type
- Transmission type
- Fuel efficiency
- Tax costs

Incorrect pricing may result in:

- Reduced profitability
- Longer inventory holding periods
- Poor customer conversion rates

The goal of this project is to create a machine learning solution that predicts vehicle prices with high accuracy and helps the dealership make informed pricing decisions.

---

# Dataset Information

The dataset contains historical Ford car sales records.

### Dataset Features

| Feature | Description |
|---|---|
| model | Ford car model |
| year | Manufacturing year |
| transmission | Transmission type |
| mileage | Total miles driven |
| fuelType | Fuel category |
| tax | Annual tax amount |
| mpg | Miles per gallon |
| engineSize | Engine displacement |
| price | Selling price (Target Variable) |

Dataset Source:
Public Ford car pricing dataset from Kaggle (modified for educational purposes).

---

# Technologies Used

- Python
- Pandas
- NumPy
- Seaborn
- Matplotlib
- Scikit-learn

---

# Project Workflow

The project was completed in the following stages:

1. Data Loading
2. Data Cleaning
3. Exploratory Data Analysis
4. Feature Correlation Analysis
5. Data Visualization
6. Model Development
7. Model Evaluation
8. Hyperparameter Optimization
9. Final Model Selection

---

# 1. Data Loading

The dataset was imported into a Pandas DataFrame for analysis.

```python
import pandas as pd

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMSkillsNetwork-AI0271EN-SkillsNetwork/labs/v1/m3/data/used_car_price_analysis.csv"

df = pd.read_csv(url)

print(df.head())
```

---

# 2. Data Cleaning

Data quality issues were addressed before modeling.

## Missing Values

Missing numerical values were replaced using column mean imputation.

```python
numeric_cols = df.select_dtypes(include=['number']).columns

for col in numeric_cols:
    df[col].fillna(df[col].mean(), inplace=True)
```

## Duplicate Records

Duplicate entries were identified and removed.

```python
df.drop_duplicates(inplace=True)
```

---

# 3. Exploratory Data Analysis (EDA)

EDA was performed to better understand relationships between variables and vehicle prices.

## Correlation Analysis

The numerical features most strongly correlated with price were identified.

```python
correlation = df.corr(numeric_only=True)
print(correlation['price'].sort_values(ascending=False))
```

### Key Findings

Typical observations included:

- Newer vehicles generally have higher prices
- Lower mileage vehicles tend to cost more
- Larger engine sizes often correlate with higher vehicle prices

---

# 4. Fuel Type Analysis

The distribution of vehicles by fuel type was analyzed.

```python
fuel_counts = df['fuelType'].value_counts()
print(fuel_counts)
```

## Visualization

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(data=df, x='fuelType')
plt.xticks(rotation=45)
plt.title('Vehicle Count by Fuel Type')
plt.show()
```

### Insight

This analysis helps identify dominant fuel categories within the dealership inventory.

---

# 5. Transmission Type Outlier Analysis

A boxplot was created to identify which transmission type contained the highest number of price outliers.

```python
sns.boxplot(x='transmission', y='price', data=df)
plt.title('Price Distribution by Transmission Type')
plt.show()
```

### Insight

This analysis highlights pricing inconsistencies across transmission categories and helps identify unusually expensive or underpriced vehicles.

---

# 6. Machine Learning Models

Several regression models were developed and compared.

---

## Model 1: Simple Linear Regression

### Objective

Predict vehicle price using only the `mpg` feature.

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

X = df[['mpg']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

### Evaluation Metrics

```python
print(r2_score(y_test, predictions))
print(mean_squared_error(y_test, predictions))
```

### Observation

Using a single variable limits predictive performance because vehicle prices depend on multiple factors.

---

## Model 2: Multiple Linear Regression

### Features Used

- year
- mileage
- tax
- mpg
- engineSize

```python
features = ['year', 'mileage', 'tax', 'mpg', 'engineSize']

X = df[features]
y = df['price']
```

### Model Training

```python
multi_model = LinearRegression()
multi_model.fit(X_train, y_train)
```

### Observation

This model significantly improved prediction performance compared to simple linear regression because it captured more vehicle characteristics.

---

## Model 3: Polynomial Regression Pipeline

Polynomial regression was used to capture nonlinear relationships between variables.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('model', LinearRegression())
])
```

### Observation

Polynomial transformations improved the model’s ability to capture complex relationships in the dataset.

---

## Model 4: Ridge Regression

Ridge regression was applied to reduce overfitting caused by polynomial features.

```python
from sklearn.linear_model import Ridge

ridge_model = Ridge(alpha=0.1)
```

### Observation

Regularization improved model stability and reduced variance.

---

# 7. Hyperparameter Optimization using GridSearchCV

A Grid Search was performed to determine the best regularization strength (`alpha`) for Ridge Regression.

## Parameters Tested

```python
parameters = {
    'alpha': [0.01, 0.1, 1, 10, 100]
}
```

## Grid Search

```python
from sklearn.model_selection import GridSearchCV

ridge = Ridge()

grid = GridSearchCV(
    ridge,
    parameters,
    cv=4,
    scoring='r2'
)
```

### Purpose

This process automatically identifies the hyperparameter value that produces the best cross-validation performance.

---

# Model Evaluation Metrics

The following metrics were used:

| Metric | Purpose |
|---|---|
| R² Score | Measures how well the model explains variance |
| Mean Squared Error (MSE) | Measures average prediction error |

---

# Final Results

| Model | Strength |
|---|---|
| Simple Linear Regression | Baseline model |
| Multiple Linear Regression | Better performance using multiple variables |
| Polynomial Regression | Captures nonlinear relationships |
| Ridge Regression | Reduces overfitting and improves generalization |

### Best Performing Model

The optimized Ridge Regression model with polynomial features produced the best balance between predictive accuracy and generalization.

---

# Key Business Insights

## 1. Vehicle Age Matters

Newer vehicles consistently achieved higher resale prices.

## 2. Mileage Strongly Affects Pricing

Higher mileage vehicles generally experienced lower resale values.

## 3. Engine Size Influences Price

Vehicles with larger engines often commanded premium prices.

## 4. Transmission Type Affects Price Distribution

Certain transmission categories showed significantly more price outliers.

## 5. Multi-feature Models Perform Better

Using multiple vehicle characteristics dramatically improved predictive accuracy.

---

# Challenges Faced

- Handling missing values
- Managing duplicate records
- Preventing overfitting in polynomial models
- Selecting optimal model hyperparameters

---

# Future Improvements

Possible future enhancements include:

- Adding more vehicle features
- Using advanced ensemble models such as Random Forest or XGBoost
- Deploying the model as a web application
- Integrating real-time vehicle market pricing data
- Adding VIN-based vehicle history information

---

# Conclusion

This project successfully developed a machine learning solution capable of predicting Ford used car prices using historical sales data.

The workflow included:

- Data cleaning
- Exploratory data analysis
- Visualization
- Regression model development
- Polynomial feature engineering
- Ridge regularization
- Hyperparameter tuning

Among all tested approaches, Ridge Regression combined with polynomial feature transformation delivered the best predictive performance.

The final solution provides a scalable and data-driven pricing framework that can assist used car dealerships in generating competitive and optimized vehicle quotations.

---

# Author

Jack Pumpuni Frimpong-Manso

