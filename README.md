# 🚗 Ford Used Car Price Prediction using Machine Learning

## 📌 Project Overview

This project applies machine learning techniques to predict the resale price of Ford vehicles using historical sales data. The goal is to help a used car dealership estimate optimal pricing for inventory using data-driven insights.

The project covers:

* Data cleaning and preprocessing
* Exploratory Data Analysis (EDA)
* Feature correlation analysis
* Data visualization
* Machine learning model development
* Model evaluation and comparison
* Hyperparameter tuning using GridSearchCV

---

## 📊 Dataset Information

The dataset contains historical Ford car sales records with the following features:

| Feature      | Description                     |
| ------------ | ------------------------------- |
| model        | Car model name                  |
| year         | Manufacturing year              |
| transmission | Transmission type               |
| mileage      | Total miles driven              |
| fuelType     | Type of fuel used               |
| tax          | Annual tax cost                 |
| mpg          | Miles per gallon                |
| engineSize   | Engine size                     |
| price        | Selling price (target variable) |

---

## 🛠️ Technologies Used

* Python 🐍
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## 📁 Project Workflow

### 1. Data Loading

* Imported dataset using Pandas
* Previewed data structure

### 2. Data Cleaning

* Handled missing values using mean imputation
* Removed duplicate records

### 3. Exploratory Data Analysis (EDA)

* Correlation analysis with price
* Fuel type distribution
* Transmission type outlier analysis

### 4. Model Development

Built and compared multiple regression models:

* Linear Regression (single feature)
* Multiple Linear Regression
* Polynomial Regression
* Ridge Regression

### 5. Model Optimization

* Applied GridSearchCV
* Tuned Ridge Regression hyperparameter (alpha)

---

## 📈 Key Insights

* Newer cars generally have higher prices
* Mileage negatively impacts resale value
* Engine size positively correlates with price
* Transmission type affects price variability
* Multi-feature models significantly improve prediction accuracy

---

## 🧠 Best Performing Model

The **Polynomial Ridge Regression model** provided the best balance between:

* Accuracy (R² score)
* Generalization
* Reduced overfitting

---

## 🚀 How to Run This Project

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ford-car-price-prediction.git
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebook

Open Jupyter Notebook or JupyterLab and run:

```
Ford_Car_Price_Analysis.ipynb
```

---

## 📌 Model Evaluation Metrics

* R² Score → Model accuracy measure
* Mean Squared Error (MSE) → Prediction error

---

## 🔮 Future Improvements

* Deploy model using Flask or Streamlit
* Try advanced models (XGBoost, Random Forest)
* Add real-time car market data
* Improve feature engineering

---

## 👨‍💻 Author

Jack Pumpuni Frimpong-Manso

---

## ⭐ Acknowledgements

* Kaggle dataset contributors
* IBM Skills Network lab resources
* Scikit-learn documentation

---

## 📜 License

This project is for educational purposes only.
