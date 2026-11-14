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
from sklearn.model_selection import (
    train_test_split, 
    StratifiedKFold,
    cross_validate
)
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

# verify target values and add raise error for missing data
if y.isnull().any():
    raise ValueError("Target column contains missing or unmapped values.")

print("n\Target Distribution:")
print(y.value_counts())
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
# Combine Preprocessing and XGBoost  Model 
# ============================================================
# Combined  the preproceesing and model together because each cross
# validation fold will independently fit its own imputer and encoder 
# using that fold's training data

model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', xgb_model)
])

# ============================================================
# 5-Fold Stratified Cross-Validation
# ============================================================

# Divides the trainin set into 5 folds, each fold maintains
# the same target
# Using stratified cross validation instead of Kfold 
# because our target is binary no/yes

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=21
)
# ==============================================================
# Evaluate model using classification metrics
# ==============================================================

cv_results = cross_validate(
    model_pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring={
       'accuracy': 'accuracy',
       'precision': 'precision',
       'recall' : 'recall',
       'f1' :'f1'
    },
    return_train_score=False
 )

# ==============================================================
# Display Cross- Validation results
# ==============================================================

print("\n5-Fold Cross-Validation Results")
print(50 * "=")

for metric in ['accuracy', 'precision', 'recall', 'f1']:
    scores = cv_results[f'test_{metric}']

    print(f"\n{metric.capitalize()} Scores: {scores}")

    print(
        f"Mean {metric.capitalize()}: "
        f"{scores.mean():.4f}"
    )
    print(
        f"Standard Deviation: "
        f"{scores.std():.4f}"
    )
# ============================================================
# Train  Model 
# ============================================================

# Train XGBoost using the prepared data 
model_pipeline.fit(
    X_train, 
    y_train
)

# ============================================================
# Generate Predictions
# ============================================================

# Use trained model to predict target values 
# on the unseen test data
y_pred = model_pipeline.predict(X_test)

# ============================================================
# Evaluate Model Performance
# ============================================================

# Calculate CLassification metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("n\Model Performance")
print(50 * '=')
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
