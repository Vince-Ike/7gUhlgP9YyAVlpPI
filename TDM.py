# ============================================================
# XGBoost Classification Model
# ============================================================
# Purpose:
# Prepare the dataset
# train an XGBoost classifier, and evaluate model
# performance using classification metrics and 5-fold validation.
# ============================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ==============================================================
# Load Dataset
# ==============================================================

df = pd.read_csv('/Users/vinceike/Desktop/Apziva Projects/term-deposit-marketing-2020.csv')
print(df.isnull().any())
print(df.dtypes)

# ==============================================================
# Seperate Features and Target
# ==============================================================

X = df.drop(columns=['y'])
y = df['y']

# y needs to be mapped since its in str[yes, no]
# used the strip and lower string format incase of inconsistencies
y = y.str.strip().str.lower().map({'yes': 1, 'no' : 0})
print(y)
# ==============================================================
# Create Training and Testing Splits
# ==============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=21,
    stratify=y
)

# ==============================================================
# Identify Feature types
# ==============================================================

categorical_features = X.select_dtypes(include=['object','str']).columns

numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

print("\nCategorical Features")
print(categorical_features)
print("\nNumerical Features")
print(numerical_features)