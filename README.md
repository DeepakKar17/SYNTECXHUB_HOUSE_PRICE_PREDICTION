# 🏠 House Price Prediction
### SyntecxHub Internship — Project 1 | Machine Learning Domain

---

## 📌 Project Overview

This project builds a **Linear Regression model** to predict median house prices using the **California Housing Dataset**. It covers the complete machine learning pipeline — from data loading and exploration to model training, evaluation, and saving.

---

## 🎯 Objectives

- Load a housing dataset, clean and explore features
- Select features, split train/test, and train a Linear Regression model
- Evaluate using **RMSE** and **R²**, and interpret coefficients
- Save model and show example predictions

---

## 📁 Project Structure

```
House_Price_Prediction/
│
├── House_Price_Prediction.ipynb    # Main Jupyter Notebook
├── house_price_model.pkl           # Saved trained model (generated on run)
├── scaler.pkl                      # Saved StandardScaler (generated on run)
├── README.md                       # This file
│
└── plots/ (generated on run)
    ├── target_distribution.png
    ├── correlation_heatmap.png
    ├── feature_distributions.png
    ├── scatter_plots.png
    ├── model_performance.png
    ├── feature_coefficients.png
    └── example_predictions.png
```

---

## 📦 Requirements

Install all dependencies with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib
```

Or using a requirements file:

```bash
pip install -r requirements.txt
```

### `requirements.txt`
```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
joblib>=1.1.0
jupyter>=1.0.0
```

---

## 🚀 How to Run

### Option 1 — Jupyter Notebook (Recommended)
```bash
# Clone or download the project
cd House_Price_Prediction

# Launch Jupyter
jupyter notebook House_Price_Prediction.ipynb
```

### Option 2 — JupyterLab
```bash
jupyter lab House_Price_Prediction.ipynb
```

Run all cells in order with **Kernel → Restart & Run All**.

---

## 📊 Dataset

**California Housing Dataset** (from `sklearn.datasets`)

| Property | Value |
|----------|-------|
| Samples  | 20,640 |
| Features | 8 original + 3 engineered |
| Target   | Median House Value (in $100,000s) |
| Source   | `sklearn.datasets.fetch_california_housing` |

### Features Used

| Feature | Description |
|---------|-------------|
| `MedInc` | Median income in block group |
| `HouseAge` | Median house age in block group |
| `AveRooms` | Average number of rooms per household |
| `AveBedrms` | Average number of bedrooms per household |
| `Population` | Block group population |
| `AveOccup` | Average number of household members |
| `Latitude` | Block group latitude |
| `Longitude` | Block group longitude |
| `RoomsPerHousehold` | *(Engineered)* Rooms / HouseAge |
| `BedroomsPerRoom` | *(Engineered)* Bedrooms / Rooms |
| `PopulationPerHouse` | *(Engineered)* Population / AveOccup |

---

## 🔧 ML Pipeline

```
Raw Data
   ↓
Load Dataset (fetch_california_housing)
   ↓
EDA (distributions, correlation heatmap, scatter plots)
   ↓
Data Cleaning (outlier capping via IQR)
   ↓
Feature Engineering (3 new derived features)
   ↓
Train/Test Split (80% / 20%, random_state=42)
   ↓
StandardScaler (fit on train, transform both)
   ↓
Linear Regression (model.fit)
   ↓
Evaluation (RMSE, MAE, R²)
   ↓
Coefficient Interpretation
   ↓
Save Model + Scaler (joblib)
   ↓
Example Predictions on New Data
```

---

## 📈 Model Results

| Metric | Train Set | Test Set |
|--------|-----------|----------|
| **RMSE** | ~0.72 | ~0.73 |
| **MAE** | ~0.52 | ~0.53 |
| **R² Score** | ~0.62 | ~0.60 |

> 💡 The model explains ~60% of variance in house prices — a solid baseline for Linear Regression.
> For better accuracy, consider upgrading to **Random Forest** or **XGBoost**.

---

## 🔎 Key Insights

- **MedInc (Median Income)** is the strongest predictor of house price — higher income areas have significantly higher prices.
- **Latitude & Longitude** reflect geographic effects — coastal California properties are more expensive.
- **AveOccup (overcrowding)** has a negative effect — more occupants per room lowers price.
- **HouseAge** has a slight positive effect in some areas (older homes in established neighborhoods).

---

## 💾 Saving & Loading the Model

The model is saved using `joblib` for fast serialization:

```python
import joblib

# Save
joblib.dump(model,  'house_price_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Load and predict
loaded_model  = joblib.load('house_price_model.pkl')
loaded_scaler = joblib.load('scaler.pkl')

prediction = loaded_model.predict(loaded_scaler.transform(new_data))
print(f"Predicted Price: ${prediction[0] * 100_000:,.0f}")
```

---

## 🏡 Quick Prediction Example

```python
import pandas as pd, joblib

model  = joblib.load('house_price_model.pkl')
scaler = joblib.load('scaler.pkl')

house = pd.DataFrame([{
    'MedInc': 6.0, 'HouseAge': 20.0, 'AveRooms': 6.0,
    'AveBedrms': 1.1, 'Population': 1000, 'AveOccup': 2.8,
    'Latitude': 37.5, 'Longitude': -122.0,
    'RoomsPerHousehold': 6.0/20, 'BedroomsPerRoom': 1.1/6,
    'PopulationPerHouse': 1000/2.8
}])

price = model.predict(scaler.transform(house))[0]
print(f"Predicted House Price: ${price * 100_000:,.0f}")
```

---

## 📚 References

- [Scikit-learn Linear Regression Docs](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [California Housing Dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)
- SyntecxHub Example Links provided in project brief

---

## 👤 Author

**SyntecxHub ML Internship**
Domain: Machine Learning | Project: 1
> *"Create | Think | Solve"*

---

*Feel free to extend this project with Ridge/Lasso Regression, Random Forest, or XGBoost for improved accuracy!*
