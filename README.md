# Credit Card Approval Forecasting

## 📌 Overview
This project predicts credit card approval outcomes using a dataset of **2.9M+ application and credit records**.  
It combines advanced preprocessing, feature engineering, and machine learning pipelines to forecast approval decisions with high accuracy.

---

## ⚡ Key Features
- **Target Engineering:** Created approval risk labels from raw credit logs using Pandas SQL‑style merges and filtering.  
- **Preprocessing Pipelines:** Implemented `ColumnTransformer` with **OrdinalEncoder, OneHotEncoder, MinMaxScaler**, improving runtime by **35%** compared to manual setups.  
- **Model Training:** Trained and evaluated **5+ ML models** including Random Forest, SVM, and XGBoost.  
- **Performance:** Achieved **95.2% accuracy** and **92.6% ROC‑AUC** with a calibrated Random Forest classifier.  
- **Class Balancing:** Applied **SMOTE** to handle imbalanced data.  
- **Visualization:** Used **Yellowbrick** and **scikit‑plot** for ROC curves, confusion matrices, and calibration plots.  

---

## 🛠️ Tech Stack
- **Languages:** Python  
- **Libraries:** Pandas, NumPy, Scikit‑Learn, Imbalanced‑Learn, Seaborn, Matplotlib, Yellowbrick, scikit‑plot  
- **Data:** Application records + credit logs (~2.9M rows)  

---

## 📂 Repository Structure
