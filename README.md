# credit-card-churn-prediction
Predicting credit card customer churn using machine learning with behavioral analysis and explainable AI techniques.

## Key Objectives
- Predict credit card customer churn
- Analyze behavioral patterns related to customer attrition
- Build interpretable machine learning models


---

## 📌 Overview

A complete end-to-end machine learning project that predicts whether a credit card customer will churn (leave the bank) based on their demographic and transaction data.

The project includes data preprocessing, exploratory data analysis, model training, hyperparameter tuning, SHAP explainability, a REST API and an interactive Streamlit UI.

---

## 📂 Project Structure
```
credit-card-churn-prediction/
├── data/
│   ├── BankChurners.csv              # Raw data
│   ├── clean_churn_data.csv          # Cleaned data
│   └── processed/                    # Train/test arrays
│       ├── X_train.npy
│       ├── X_test.npy
│       ├── y_train.npy
│       ├── y_test.npy
│       └── feature_names.csv
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb  # Encoding, SMOTE, Scaling
│   ├── 03_model_training.ipynb       # Model Training & Comparison
│   ├── 04_hyperparameter_tuning.ipynb # GridSearchCV Tuning
│   └── 05_shap_explainability.py     # SHAP Feature Importance
├── deployment/                    
│   ├── streamlit_app.py              # Streamlit UI
│   └── xgboost_tuned.pkl             # Saved Model
├── reports/
│   ├── 01_class_distribution.png
│   ├── 02_feature_distributions.png
│   ├── 03_correlation_heatmap.png
│   ├── 04_categorical_churnrate.png
│   ├── 05_model_comparison.png
│   ├── 06_feature_importance.png
│   ├── 07_tuned_vs_baseline.png
│   ├── 08_shap_summary_bar.png
│   ├── 09_shap_beeswarm.png
│   └── 10_shap_waterfall.png
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

- **Source**: BankChurners Dataset
- **Rows**: 10,127 customers
- **Target**: `Attrition_Flag` (0 = Existing, 1 = Churned)
- **Class imbalance**: 83.9% existing vs 16.1% churned

---

## 🔬 Project Pipeline

### 1️⃣ Exploratory Data Analysis
- Class imbalance analysis
- Feature distributions (churned vs existing)
- Correlation heatmap
- Churn rate by categorical features

### 2️⃣ Feature Engineering
- Label encoding of 5 categorical columns
- Dropped `Avg_Open_To_Buy` (0.99 correlation with `Credit_Limit`)
- Stratified 80/20 train/test split
- SMOTE to fix class imbalance on training data
- StandardScaler for feature scaling

### 3️⃣ Model Training


### 4️⃣ Hyperparameter Tuning
- GridSearchCV with 5-fold StratifiedKFold
- Best params: `max_depth=7`, `n_estimators=300`, `learning_rate=0.1`
- Tuned AUC: 0.9912 (baseline was already optimal)

### 5️⃣ Model Explainability (SHAP)
Top features driving churn:
- 🔴 `Total_Trans_Ct` — fewer transactions = higher churn risk
- 🔴 `Total_Trans_Amt` — lower spend = higher churn risk
- 🔴 `Total_Revolving_Bal` — low balance = higher churn risk
- 🔴 `Months_Inactive_12_mon` — more inactivity = higher churn risk

### 6️⃣ Deployment
- **Streamlit** interactive UI for non-technical users

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/your-username/credit-card-churn-prediction.git
cd credit-card-churn-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Streamlit UI
```bash
cd deployment
streamlit run streamlit_app.py
```

## 📦 Requirements
```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
shap
joblib
fastapi
uvicorn
streamlit
pydantic
```

---

## 📈 Key Results

- ✅ **Best Model**: XGBoost
- ✅ **ROC-AUC**: 0.9917
- ✅ **No overfitting**: Train AUC 1.0 vs Test AUC 0.9917 (gap = 0.0083)
- ✅ **Top predictor**: Total Transaction Count

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.13 | Core language |
| Pandas & NumPy | Data manipulation |
| Scikit-learn | ML pipeline |
| XGBoost | Best model |
| SHAP | Explainability |
| FastAPI | REST API |
| Streamlit | Interactive UI |
| Joblib | Model serialization |
| Git & GitHub | Version control |

---

## 👩‍💻 Author

**Shruti**
https://github.com/ShrutiSingh2909/credit-card-churn-prediction.git

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
