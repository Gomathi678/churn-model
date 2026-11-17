
# =====================================================
# TASK 1: DATA AND EDA - CUSTOMER CHURN PREDICTION
# =====================================================

# 1. Import required libraries

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

sns.set_style("whitegrid")

# 2. Load dataset

print("\n===== LOAD DATASET =====")

df = pd.read_csv("data.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

# 3. Check dataset information

print("\n===== DATASET INFORMATION =====")

print("\nShape:", df.shape)

print("\nData Types:")
print(df.dtypes)

print("\nDataset Info:")
df.info()

# 4. Check missing values

print("\n===== MISSING VALUES =====")

# Convert TotalCharges into numeric format
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

print(df.isnull().sum())

print("\nTotal Missing Values:")
print(df.isnull().sum().sum())

# 5. Check duplicate records

print("\n===== DUPLICATE CHECK =====")

print("Duplicate rows:", df.duplicated().sum())

print(
    "Duplicate Customer IDs:",
    df["customerID"].duplicated().sum()
)

# 6. Check class balance

print("\n===== CHURN CLASS BALANCE =====")

print(df["Churn"].value_counts())

print("\nChurn Percentage:")
print(df["Churn"].value_counts(normalize=True) * 100)

# 7. Split data into training and testing sets

print("\n===== TRAIN TEST SPLIT =====")

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

print("\nTraining class balance:")
print(y_train.value_counts(normalize=True) * 100)

print("\nTesting class balance:")
print(y_test.value_counts(normalize=True) * 100)

# 8. Create EDA dataset using training data only

eda_df = X_train.copy()
eda_df["Churn"] = y_train

# 9. EDA CHART 1: Churn Distribution

plt.figure(figsize=(7, 5))

sns.countplot(data=eda_df, x="Churn")

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()

# 10. EDA CHART 2: Contract Type vs Churn

plt.figure(figsize=(8, 5))

sns.countplot(
    data=eda_df,
    x="Contract",
    hue="Churn"
)

plt.title("Contract Type vs Customer Churn")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()

# 11. EDA CHART 3: Internet Service vs Churn

plt.figure(figsize=(8, 5))

sns.countplot(
    data=eda_df,
    x="InternetService",
    hue="Churn"
)

plt.title("Internet Service vs Customer Churn")
plt.xlabel("Internet Service")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()

# 12. EDA CHART 4: Tenure vs Churn

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=eda_df,
    x="Churn",
    y="tenure"
)

plt.title("Customer Tenure vs Churn")
plt.xlabel("Churn")
plt.ylabel("Tenure (Months)")

plt.tight_layout()
plt.show()

# 13. EDA CHART 5: Monthly Charges vs Churn

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=eda_df,
    x="Churn",
    y="MonthlyCharges"
)

plt.title("Monthly Charges vs Customer Churn")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")

plt.tight_layout()
plt.show()

# 14. Assumptions and Metric Selection

print("\n===== ASSUMPTIONS =====")

print("1. Churn Yes means customer left the company.")
print("2. Churn No means customer stayed.")
print("3. Customer ID is not a predictive feature.")
print("4. Contract type and tenure may influence churn.")
print("5. Dataset contains class imbalance.")

print("\n===== METRIC SELECTION =====")

print("Selected metric: F1 Score")
print("Reason: F1 balances Precision and Recall.")
print("It is useful for imbalanced classification problems.")

print("\n===== TASK 1 COMPLETED SUCCESSFULLY =====")
