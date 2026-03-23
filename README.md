# 🩺 Diabetes Prediction with CART Decision Tree

Binary classification model predicting diabetes onset using the Pima Indians Diabetes Dataset and a tuned CART (Classification and Regression Trees) decision tree.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## Problem Statement

Given 8 diagnostic measurements, predict whether a patient has diabetes (Outcome = 1) or not (Outcome = 0). The dataset is the [Pima Indians Diabetes Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database) containing 768 observations.

## Dataset Features

| Feature | Description |
|---------|-------------|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-Hour serum insulin (mu U/ml) |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Diabetes pedigree function |
| Age | Age in years |
| **Outcome** | **Target: 1 = diabetic, 0 = non-diabetic** |

## Approach

1. **EDA** — Distribution analysis, target balance check, feature correlations
2. **Missing Value Handling** — Zeros treated as missing; imputed with class-conditional medians
3. **Feature Engineering** — BMI clinical categories (Underweight → Obesity 3), Insulin normality flag
4. **Outlier Handling** — IQR-based winsorization
5. **Encoding** — Label encoding for binary, rare encoding, one-hot encoding for multi-category
6. **Model** — CART Decision Tree with GridSearchCV (5-fold CV) over max_depth and min_samples_split

## Results

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| CART (default) | ~72% | — |
| CART (tuned via GridSearchCV) | **~75%** | Computed via `roc_auc_score` |

**Why CART?** Decision trees offer full interpretability — critical in healthcare applications where clinicians need to understand the reasoning behind predictions.

## Engineered Features

| Feature | Description |
|---------|-------------|
| `NewBMI` | BMI bucketed into 6 clinical categories |
| `New_Insulin` | Insulin flagged as Normal (16–166) or Abnormal |

## Tech Stack

- **Python 3.8+** — Core language
- **Scikit-learn** — DecisionTreeClassifier, GridSearchCV, metrics
- **Pandas / NumPy** — Data manipulation
- **Seaborn / Matplotlib** — Visualization
- **pydotplus** — Decision tree visualization

## Project Structure

```
Diabetes_Prediction_ML_CART/
├── helpers/
│   ├── __init__.py
│   ├── data_prep.py          # Outlier handling, imputation, encoding utilities
│   └── eda.py                # EDA summary and visualization functions
├── Diabetes_prediction_CART.py  # Main ML pipeline
├── requirements.txt
└── README.md
```

## Getting Started

```bash
git clone https://github.com/eboekenh/Diabetes_Prediction_ML_CART.git
cd Diabetes_Prediction_ML_CART
pip install -r requirements.txt
```

Download `diabetes.csv` from the [Kaggle dataset](https://www.kaggle.com/uciml/pima-indians-diabetes-database) and place it in the project root.

```bash
python Diabetes_prediction_CART.py
```

## License

MIT
