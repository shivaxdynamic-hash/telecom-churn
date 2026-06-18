# Telecom Customer Churn Prediction

A supervised machine learning project that predicts whether a telecom customer will churn (leave the service), built end-to-end in Python using scikit-learn.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)

---

## Project Overview

Customer churn is one of the most critical business problems in the telecommunications industry. This project builds and evaluates five classification models to predict churn from customer account and service-usage data. The best-performing model (**SVM**) achieved **82.39% accuracy** on the held-out test set.

---

## Dataset

| Property | Value |
|---|---|
| Source | IBM Watson Analytics — Telco Customer Churn |
| File | `WA_Fn-UseC_-Telco-Customer-Churn.csv` |
| Rows (original) | 7,043 |
| Columns | 21 |
| Target variable | `Churn` (Yes / No) |

### Features

The dataset contains customer demographics, account information, and subscribed services:

- **Demographics:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- **Account info:** `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
- **Services:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`

---

## Project Workflow

### 1. Libraries
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
```

### 2. Exploratory Data Analysis (EDA)
- Loaded the dataset and inspected the first 5 rows with `df.head()` and `df.info()`
- Confirmed **no null values** across all 21 columns (`df.isnull().sum()` → all zeros)
- Confirmed **0 duplicate rows** (`df.duplicated().sum()` → 0)
- Original shape: **(7043, 21)**

### 3. Outlier Detection & Removal (IQR Method)
Applied the Interquartile Range (IQR) method to all numerical columns:
- `SeniorCitizen`: **1,142 outlier rows detected** → removed
- `tenure`: 0 outliers
- `MonthlyCharges`: 0 outliers

Shape after outlier removal: **(5901, 21)**

### 4. Label Encoding
All categorical (object-type) columns were encoded to numeric values using `LabelEncoder`, making the data compatible with sklearn classifiers.

### 5. Feature / Target Split
```python
X = df.drop("Churn", axis=1)   # 20 input features
Y = df["Churn"]                 # target label
```

### 6. Train-Test Split
```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
```
- Training set: 80% (4,720 rows)
- Test set: 20% (1,181 rows)

### 7. Feature Scaling
`StandardScaler` was applied — fit on training data only, then used to transform both train and test sets, preventing data leakage.

### 8. Model Training & Benchmarking
Five classifiers were trained and compared:

| Model | Accuracy |
|---|---|
| Logistic Regression | 81.46% |
| Decision Tree | 72.82% |
| Random Forest | 80.78% |
| KNN | 77.05% |
| **SVM (Best)** | **82.39%** |

---

## Results

The **SVM (Support Vector Machine)** achieved the highest accuracy of **82.39%**.

### Detailed Classification Report (SVM)

```
              precision    recall  f1-score   support

           0       0.85      0.94      0.89       916
           1       0.68      0.41      0.51       265

    accuracy                           0.82      1181
   macro avg       0.76      0.68      0.70      1181
weighted avg       0.81      0.82      0.81      1181
```

### Confusion Matrix (SVM)

```
[[864  52]
 [156 109]]
```

- **True Negatives (not churned, correctly predicted):** 864
- **True Positives (churned, correctly predicted):** 109
- **False Positives:** 52
- **False Negatives (missed churners):** 156

> **Note:** The model performs well on the majority class (non-churners) but has lower recall on churners (class 1 = 41%). This reflects the class imbalance in the dataset and is a known limitation.

---

## Technologies Used

| Library | Version | Purpose |
|---|---|---|
| Python | 3.x | Core language |
| NumPy | latest | Numerical operations |
| Pandas | latest | Data loading and manipulation |
| scikit-learn | latest | Preprocessing, modeling, evaluation |
| Google Colab | — | Development environment |

---

## How to Run

### Option 1 — Google Colab (Recommended)
1. Open `telecomchurn.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Upload `WA_Fn-UseC_-Telco-Customer-Churn.csv` to your Google Drive under `MyDrive/`
3. Mount your Drive when prompted, then run all cells in order

### Option 2 — Local (Jupyter Notebook)
1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/telecom-churn-prediction.git
   cd telecom-churn-prediction
   ```
2. Install dependencies:
   ```bash
   pip install numpy pandas scikit-learn jupyter
   ```
3. Update the dataset path in Cell 2:
   ```python
   df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
   ```
4. Launch Jupyter and run:
   ```bash
   jupyter notebook telecomchurn.ipynb
   ```

---

## Project Structure

```
telecom-churn-prediction/
│
├── telecomchurn.ipynb               # Main notebook (all steps)
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Dataset
└── README.md                        # Project documentation
```

---

## Author

**Siva Kumar Thota**

*Data Analyst Portfolio Project — built with Python and scikit-learn on Google Colab*
