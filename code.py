"""
Credit Card Approval Forecasting
--------------------------------
Predicts credit card approval outcomes using application and credit record data.
Includes feature engineering, preprocessing, class balancing, and model training.
"""

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Load datasets
applications = pd.read_csv("dataset/application_record.csv")
credit_logs = pd.read_csv("dataset/credit_record.csv")

# Feature Engineering
# Account age: minimum months balance per ID
account_age = credit_logs.groupby("ID")["MONTHS_BALANCE"].min().reset_index()
account_age.rename(columns={"MONTHS_BALANCE": "Account age"}, inplace=True)
applications = applications.merge(account_age, on="ID", how="left")

# Target variable: Is high risk (1 = Yes, 0 = No)
credit_logs["dep_value"] = None
credit_logs.loc[credit_logs["STATUS"].isin(["2", "3", "4", "5"]), "dep_value"] = "Yes"
risk_flag = credit_logs.groupby("ID")["dep_value"].count().reset_index()
risk_flag["dep_value"] = np.where(risk_flag["dep_value"] > 0, "Yes", "No")
applications = applications.merge(risk_flag, on="ID", how="inner")
applications["Is high risk"] = applications["dep_value"].map({"Yes": 1, "No": 0})
applications.drop("dep_value", axis=1, inplace=True)

# Rename columns for readability
applications.rename(columns={
    "CODE_GENDER": "Gender",
    "FLAG_OWN_CAR": "Has a car",
    "FLAG_OWN_REALTY": "Has a property",
    "CNT_CHILDREN": "Children count",
    "AMT_INCOME_TOTAL": "Income",
    "NAME_INCOME_TYPE": "Employment status",
    "NAME_EDUCATION_TYPE": "Education level",
    "NAME_FAMILY_STATUS": "Marital status",
    "NAME_HOUSING_TYPE": "Dwelling",
    "DAYS_BIRTH": "Age",
    "DAYS_EMPLOYED": "Employment length",
    "FLAG_MOBIL": "Has a mobile phone",
    "FLAG_WORK_PHONE": "Has a work phone",
    "FLAG_PHONE": "Has a phone",
    "FLAG_EMAIL": "Has an email",
    "OCCUPATION_TYPE": "Job title",
    "CNT_FAM_MEMBERS": "Family member count"
}, inplace=True)

# Train-test split
train_df, test_df = train_test_split(applications, test_size=0.2, random_state=42)

# Quick EDA
msno.matrix(train_df)
plt.show()

sns.histplot(np.abs(train_df["Age"]) / 365.25, bins=50, kde=True)
plt.title("Age Distribution")
plt.show()

# Model Training (Random Forest Example)
X_train = train_df.drop("Is high risk", axis=1)
y_train = train_df["Is high risk"]

# Encode categorical variables
X_train = pd.get_dummies(X_train)

# Handle imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train_bal, y_train_bal)

# Evaluation
y_pred = rf_model.predict(X_train_bal)
print(classification_report(y_train_bal, y_pred))
print("ROC-AUC:", roc_auc_score(y_train_bal, rf_model.predict_proba(X_train_bal)[:, 1]))
