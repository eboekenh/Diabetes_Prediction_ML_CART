# 🩺 Diabetes Prediction with CART (Decision Tree)

> **Binary classification project predicting diabetes onset using a CART Decision Tree — with emphasis on model interpretability.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Decision%20Tree-orange.svg)](https://scikit-learn.org/)
[![Dataset](https://img.shields.io/badge/Dataset-Pima%20Indians-lightblue.svg)](https://www.kaggle.com/uciml/pima-indians-diabetes-database)
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

---

## 📖 About the Project

This project uses a **CART (Classification and Regression Trees)** Decision Tree to predict whether a patient is likely to have diabetes, based on diagnostic measurements from the **Pima Indians Diabetes Dataset**.

CART was chosen deliberately for its **interpretability** — in a healthcare context, understanding *why* a prediction is made is just as important as the prediction itself.

---

## 🎯 Problem Statement

**Binary classification**: Predict whether a patient has diabetes (`1`) or not (`0`) based on medical diagnostic features.

---

## 📋 Dataset Features

| Feature | Description |
|---|---|
| `Pregnancies` | Number of times pregnant |
| `Glucose` | Plasma glucose concentration |
| `BloodPressure` | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | Triceps skinfold thickness (mm) |
| `Insulin` | 2-Hour serum insulin (µU/ml) |
| `BMI` | Body mass index |
| `DiabetesPedigreeFunction` | Diabetes pedigree function |
| `Age` | Age in years |
| `Outcome` | **Target**: 1 = diabetic, 0 = non-diabetic |

**Source**: [Pima Indians Diabetes Database (Kaggle/UCI)](https://www.kaggle.com/uciml/pima-indians-diabetes-database)

---

## 🔧 Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Scikit-learn | CART Decision Tree model |
| Pandas, NumPy | Data manipulation |
| Matplotlib / Seaborn | Visualisation & tree plots |
| Jupyter Notebook | Interactive development |

---

## 🚀 Getting Started

```bash
git clone https://github.com/eboekenh/Diabetes_Prediction_ML_CART.git
cd Diabetes_Prediction_ML_CART
pip install -r requirements.txt
jupyter notebook notebooks/diabetes_cart.ipynb
```

---

## 📊 Approach

1. **EDA** — Distributions, correlations, class imbalance check
2. **Preprocessing** — Handle zero-values, feature scaling
3. **Model Training** — CART Decision Tree with depth tuning
4. **Tree Visualisation** — Plot the full decision tree for interpretability
5. **Evaluation** — Accuracy, confusion matrix, ROC curve

---

## 📄 License

MIT License

---

## 👤 Author

**[@eboekenh](https://github.com/eboekenh)**