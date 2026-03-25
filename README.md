# 🩺 Comparative Analysis of Machine Learning Algorithms for Diabetes Mellitus Classification

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-8A2BE2?style=flat-square)
![CRISP-DM](https://img.shields.io/badge/Methodology-CRISP--DM-blue?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Grade](https://img.shields.io/badge/Capstone%20Grade-A-brightgreen?style=flat-square)

**Cairo University · Faculty of Graduate Studies for Statistical Research (FGSSR)**
**Data Science Department · Academic Diploma Capstone Project**

*Supervised by Prof. Dr. Muhammad Abdel Hamid Sabry*
*Head of Mathematical Statistics Department*

</div>

---

## 📌 Project Overview

Diabetes Mellitus is a chronic disease affecting hundreds of millions worldwide, often
going undetected until serious complications arise. Traditional screening relies on
time-consuming laboratory tests — creating a critical need for automated, reliable
early-detection tools.

This project builds and benchmarks **5 supervised ML classifiers** on clinical data,
with a primary focus on **maximizing recall (sensitivity)** — ensuring the fewest
possible missed diagnoses in a real-world screening context.

> **Core Question:** Which algorithm — Logistic Regression, Decision Tree, Random
> Forest, KNN, or MLP — provides the highest diagnostic sensitivity and screening
> safety for diabetes classification?

---

## 🏆 Key Results

| Metric | Best Model | Score |
|---|---|---|
| **Recall (Sensitivity)** | Logistic Regression | **92.6%** |
| **F2-Score** | Logistic Regression | **0.806** |
| **ROC AUC** | Random Forest | **0.813** |
| **PR AUC (Avg Precision)** | Random Forest | **0.686** |
| **Missed Diagnoses Reduced** | vs. Default Threshold | **75% reduction** |
| **False Negatives (Test Set)** | Random Forest | **5 cases** |

### Full Model Comparison (Tuned Threshold — Test Set)

| Model | Recall | F2 | ROC AUC | Accuracy | Precision |
|---|---|---|---|---|---|
| **Logistic Regression** | **0.926** | **0.806** | 0.812 | 0.688 | 0.532 |
| Random Forest | 0.907 | 0.798 | **0.813** | 0.695 | 0.538 |
| Decision Tree | 0.870 | 0.781 | 0.787 | 0.708 | 0.553 |
| KNN | 0.833 | 0.740 | 0.781 | 0.662 | 0.511 |
| MLP | 0.778 | 0.698 | 0.758 | 0.643 | 0.494 |

---

## 📂 Repository Structure
```
diabetes-mellitus-ml-classification/
│
├── 📓 Diabetes_Prediction_Comparative_Final.ipynb   ← Full pipeline notebook
├── 📊 diabetes.csv                                  ← Pima Indians dataset
├── 📑 presentation.pdf                              ← Capstone presentation slides
├── requirements.txt                                 ← Python dependencies
├── LICENSE
└── README.md
```

---

## 🔬 Methodology — CRISP-DM
```
Domain Understanding → Data Understanding → Data Preparation → Modeling → Evaluation
```

### 1. 📊 Data Source
- **Pima Indians Diabetes Database** (UCI / Kaggle)
- 768 records · 8 clinical features · Binary classification (Diabetic / Non-Diabetic)
- Class imbalance: 65.1% negative, 34.9% positive

### 2. 🔍 Exploratory Data Analysis
- Identified **bio-impossible zeros** in Glucose, BloodPressure, SkinThickness, Insulin, BMI
- Outlier analysis: DiabetesPedigreeFunction (3.78%), Insulin (3.12%)
- Correlation ranking: Glucose (0.49) > BMI (0.31) > Insulin (0.30) > SkinThickness (0.26)
- Pairwise analysis revealed a **Glucose-BMI "Danger Zone"** for high-risk patients

### 3. ⚙️ Data Preprocessing Pipeline
```
Bio-Zero → NaN Re-encoding
    ↓
Train / Val / Test Split  (64% / 16% / 20%)  stratify=y
    ↓
Custom NaNStandardScaler  (NaN-safe scaling before imputation)
    ↓
KNNImputer  (k=5 neighbors, patient-similarity-based imputation)
    ↓
StandardScaler  (final normalization for model readiness)
```

**Key design decisions:**
- Imputation runs **inside each CV fold** to guarantee zero data leakage
- `stratify=y` preserves 65/35 class ratio across all splits
- Custom `NaNStandardScaler` solves the scale-before-impute dilemma

### 4. 🤖 Modeling

| Model | Search Strategy | Key Hyperparameters Tuned |
|---|---|---|
| Logistic Regression | GridSearchCV | C, penalty (L1/L2) |
| Decision Tree | GridSearchCV | max_depth, min_samples_leaf |
| Random Forest | RandomizedSearchCV | n_estimators (200–800), max_features |
| KNN | GridSearchCV | n_neighbors (3–15), distance metric |
| MLP | RandomizedSearchCV | hidden_layer_sizes, alpha, learning_rate |

All models used **Stratified 5-Fold Cross-Validation**, optimizing **F1-Score** during training.

### 5. 📏 Threshold Optimization

Default threshold (0.5) is suboptimal for imbalanced medical screening data.

- Tuned decision threshold `t*` on held-out **validation set**
- Objective: **Maximize F2-Score** (weights Recall 2x over Precision)
- Result: **75% reduction in missed diagnoses** vs. default threshold

### 6. 🔎 Explainability — SHAP + Feature Importance

Applied **5 complementary feature importance methods**:
- SHAP values (Decision Tree & Random Forest)
- Logistic Regression coefficients
- Permutation importance (MLP & KNN)

**Consistent finding across all methods:**

> `Glucose` > `BMI` > `Insulin` / `Age` >> `BloodPressure` / `SkinThickness`

---

## 💡 Key Findings

**Biological Insights:**
- **Glucose** is the absolute dominant predictor across all model architectures
- **BMI > 30 kg/m²** creates a high-risk clinical "danger zone" when combined with elevated glucose
- Blood Pressure and Skin Thickness show high biological redundancy in this dataset

**Clinical Recommendations:**
- **Logistic Regression** → Primary screening tool (maximum sensitivity, interpretable coefficients)
- **Random Forest** → Secondary confirmatory tool (highest AUC, best precision-recall balance)

**Methodological Contributions:**
- Custom NaN-safe preprocessing pipeline that prevents data leakage during imputation
- F2-optimized threshold tuning framework applicable to any imbalanced screening problem
- Reproducible CRISP-DM compliant pipeline with full explainability layer

---

## 🛠️ Tech Stack
```
Python 3.10+
├── scikit-learn      ← ML pipelines, GridSearch, KNNImputer, all classifiers
├── pandas / numpy    ← Data manipulation
├── matplotlib / seaborn  ← Visualization
├── shap              ← Model explainability
└── jupyter           ← Interactive notebook
```

---

## 🚀 Getting Started
```bash
# 1. Clone the repository
git clone https://github.com/alyayman2020/diabetes-mellitus-ml-classification.git
cd diabetes-mellitus-ml-classification

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the notebook
jupyter notebook Diabetes_Prediction_Comparative_Final.ipynb
```

---

## 📋 Requirements
```
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
shap>=0.42.0
jupyter>=1.0.0
```

---

## 📜 Academic Context

| | |
|---|---|
| **Institution** | Cairo University — FGSSR, Data Science Department |
| **Program** | Academic Diploma in Data Science |
| **Supervisor** | Prof. Dr. Muhammad Abdel Hamid Sabry |
| **Grade** | **A** |
| **Dataset** | Pima Indians Diabetes Database (UCI / Kaggle) |
| **Methodology** | CRISP-DM |

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Aly Ayman Ibrahim**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Aly%20Ayman-0077B5?style=flat-square&logo=linkedin)](https://linkedin.com/in/alyayman)
[![GitHub](https://img.shields.io/badge/GitHub-alyayman2020-181717?style=flat-square&logo=github)](https://github.com/alyayman2020)
[![Kaggle](https://img.shields.io/badge/Kaggle-alyaymanai-20BEFF?style=flat-square&logo=kaggle)](https://www.kaggle.com/alyaymanai)
```

---
## 📄 3. LICENSE — MIT
```
MIT License

Copyright (c) 2025 Aly Ayman Ibrahim

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
