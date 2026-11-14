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
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
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

# y needs to be mapped since its in str format [yes, no]
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

# ==============================================================
# Define categorical preprocessing pipelines
# ==============================================================
# Categorical preprocessing:
# 1. Replace missing values with the most frequent category.
# 2. Convert categorical variables into numerical features
#    using one-hot encoding.
# 3. Ignore categories that were not present during training.

categorical_pipline = Pipeline([
    ('imputer',
     SimpleImputer(strategy='most_frequent')
     ),
     (
         'encoder',
         OneHotEncoder(
             handle_unknown='ignore',
             sparse_output=False
         )
     )
])

# ==============================================================
# Define numerical preprocessing pipelines
# ==============================================================

# Numerical preprocessing
# replace any missing numerical values with the median 

numerical_pipeline = Pipeline([
    ('imputer',
     SimpleImputer(strategy='median')
     )
])

# ============================================================
# Combine Preprocessing Pipelines
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        ('categorical',
        categorical_pipline,
        categorical_features
        ),
        ('numerical',
        numerical_pipeline,
        numerical_features
        )
    ]
)

# ============================================================
# Prepare Training & Testing Data
# ============================================================

# Fit processing only on the training set
# This prevents information from the test set
# from influencing the preprocesing process and 
# helps prevent data leakage

X_train_prepared = preprocessor.fit_transform(X_train)

# Applying the already fitted preprocessing to the test data
X_test_prepared = preprocessor.transform(X_test)

# ============================================================
# Verify Prepared Data
# ============================================================

print("\nPrepared Training Shape:")
print(X_train_prepared.shape)

print("\nPrepared Testing Shape:")
print(X_test_prepared.shape)

print("\nTraining Target Shape:")
print(y_train.shape)

print("\nTesting Target Shape:")
print(y_test.shape)

# ============================================================
# Check Target Distribution
# ============================================================

print("\nTraining Target Distribution:")
print(y_train.value_counts())

print("\nTesting Target Distribution:")
print(y_test.value_counts())

print("\nTraining Target Percentages:")
print(y_train.value_counts(normalize=True) * 100)

# ============================================================
# Initialize XGBoost Model 
# ============================================================

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    random_state=21,
    eval_metric='logloss'
)

# ============================================================
# Train  Model 
# ============================================================

# Train XGBoost using the prepared data 
xgb_model.fit(
    X_train_prepared, 
    y_train
)

# ============================================================
# Generate Predictions
# ============================================================

# Use trained model to predict target values 
# on the unseen test data
y_pred = xgb_model.predict(X_test_prepared)

# ============================================================
# Evaluate Model Performance
# ============================================================

# Calculate CLassification metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("n\Model Performance")
print(40 * '=')
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# ============================================================
# Classification Report
# ============================================================

# Display precision, recall, F1 score, and support
# for each target class.

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# Generate Confusion Matrix
# ============================================================

# Calculate the confusion matrix to examine correct and
# incorrect predictions for each class.

cm = confusion_matrix(
    y_test,
    y_pred
)
