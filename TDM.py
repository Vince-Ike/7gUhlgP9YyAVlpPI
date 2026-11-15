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
#print(df.isnull().any())
#print(df.dtypes)

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

#print("n\Target Distribution:")
#print(y.value_counts())
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

#print("\nCategorical Features")
#print(categorical_features)
#print("\nNumerical Features")
#print(numerical_features)

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

# ============================================================
# Experiment with Class Weight Adjustments
# ============================================================
# Creating the scale parameter to give greater importance to the 
# minority class with only training data

class_ratio = (
    y_train.value_counts()[0] /
    y_train.value_counts()[1]
)

# Experiment with different weights 

weights = [1,2,3,4,5,7,10, class_ratio]

results = []

for weight in weights:
    print(f"Testing scale_pos_weight: {weight:.2f}")
    # ============================================================
    # Initialize XGBoost Model 
    # ============================================================

    xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    scale_pos_weight = weight,
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

    # ==============================================================
    # Perform Cross-Validation
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
    return_train_score=False,
    n_jobs= -1
    )
    # Store metrics at each weight and take the mean of the 5 fold cv metrics 
    # and the std of the F1 score since it will be the determining 
    # factor for the best weight
    results.append({
        'Weight' : weight,
        'Accuracy': cv_results['test_accuracy'].mean(),
        'Precision': cv_results['test_precision'].mean(),
        'Recall': cv_results['test_recall'].mean(),
        'F1 Score': cv_results['test_f1'].mean(),
        'F1 Std': cv_results['test_f1'].std(),
    })
# ==============================================================
# Class Weight Results Comparison
# ==============================================================
results_df = pd.DataFrame(results)

#using the F1 Score as the determinant of the best weight
results_df = results_df.sort_values(
    by= 'F1 Score',
    ascending=False
).reset_index(drop=True)

print("n\Class Weight Comparison")
print(50 * '=')

print(
    results_df.to_string(
    index=False,
    float_format= lambda x: f'{x:.4f}'
    )
)
# ============================================================
# Select Best Class Weight
# ============================================================

best_weight = results_df.loc[0,'Weight']

print("\n Selected Class Weight")
print(best_weight)

# ============================================================
# Initialize Final  Model
# ==============================================================
# made a final initialization of the model after looping through different 
# weights to find the best weight 
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    scale_pos_weight=best_weight,
    random_state=21,
    eval_metric='logloss'
)


# ============================================================
# Combine Preprocessing and Final Model
# ==============================================================

model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', xgb_model)
])

# ============================================================
# Train Final Model 
# ============================================================

# Train Final Model using the prepared data and the best weight
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

print("\nConfusion Matrix")
print(cm)