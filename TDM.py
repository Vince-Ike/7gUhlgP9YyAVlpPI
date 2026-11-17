# ============================================================
# XGBoost Classification Model
# ============================================================
# Purpose:
# Prepare the dataset
# train an XGBoost classifier, and evaluate model
# performance using classification metrics and 5-fold validation.
# ============================================================

import matplotlib
from pathlib import Path
import matplotlib.pyplot as plt
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
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==============================================================
# Load Dataset
# ==============================================================
# Get the directory containing TDM.py
BASE_DIR = Path(__file__).resolve().parent

# Move up one directory to Apziva Projects
DATA_PATH = BASE_DIR.parent / 'term-deposit-marketing-2020.csv'

# Load dataset
df = pd.read_csv(DATA_PATH)
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
# Create Training, Validation, and Testing Splits
# ==============================================================

# First seperating the final test set

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=21,
    stratify=y
)

# Creating the the validation set from the training set 

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.20,
    random_state=21,
    stratify=y_train_full
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
# turned results into a dataframe for a clean comparison display 
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

# Train Final Model using training data but leaving out 
# the validation data
model_pipeline.fit(
    X_train, 
    y_train
)

# ============================================================
# Predicition Probabilty Threshold Optimization
# ============================================================

# Generate subscription probabities using the validation data
# needed since there is a class imbalance in the dataset

y_val_prob = model_pipeline.predict_proba(X_val)[:,1]

# Testing different probablity thresholds

thresholds = np.arange(0.20, 0.71, 0.01)

threshold_results = []

for threshold in thresholds:

    # Convert probabilities into predictions
    
    y_val_pred = (y_val_prob >= threshold).astype(int)

    threshold_results.append({
        'Threshold' : threshold,
        'Accuracy' : accuracy_score(
            y_val,
            y_val_pred
        ),
        'Precision' : precision_score(
            y_val,
            y_val_pred,
            zero_division=0
        ),
        'Recall' : recall_score(
            y_val,
            y_val_pred,
            zero_division=0
        ),
        'F1 Score' : f1_score(
            y_val,
            y_val_pred,
            zero_division=0
        )
    })

# Convert threshold results in Datafrane

threshold_df = pd.DataFrame(threshold_results)

# Sort results based on F1 Score

threshold_df = threshold_df.sort_values(
    by='F1 Score',
    ascending=False
).reset_index(drop=True)

print("\nProbabilty Threshold Comparison")
print(50 * "=")

print(
    threshold_df.to_string(
        index=False,
        float_format= lambda x: f'{x:.4f}'
    )
)


# ============================================================
# Select Best Probability Threshold
# ============================================================

# Finds the threshold with the highest F1 Score since that is 
# what the thresholds are sorted by
best_threshold = threshold_df.loc[0, 'Threshold']

print("\nSelected Probability Thrshold")
print(best_threshold)

# ============================================================
# Retrain Final Model using entire training set
# ============================================================
# Combine original training and validation data

model_pipeline.fit(
    X_train_full,
    y_train_full
)

# ============================================================
# Generate Final Predicitions Using Selected Threshold
# ============================================================

# Generate probabilites on unseen test data

y_test_prob = model_pipeline.predict_proba(X_test)[:,1]

# Apply selected probabilty threshold

y_pred = (y_test_prob >= best_threshold).astype(int)

# ============================================================
# Evaluate Final Model Performance
# ============================================================

# Calculate CLassification metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("n\Final Model Performance")
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
# ==============================================================
# Feature Importance
# ==============================================================
# Need to Extract final trained Model 
trained_model = model_pipeline.named_steps['classifier']

# Extract Feature names after preprocessing

feature_names = (
    model_pipeline
    .named_steps['preprocessor']
    .get_feature_names_out()
    
)

# Get Feature Importance

feat_importance = trained_model.feature_importances_
# Create Dataframe
feat_importance_df = pd.DataFrame({
    'Feature' : feature_names,
    'Importance' : feat_importance,
})

feat_importance_df = feat_importance_df.sort_values(
    by='Importance',
    ascending=False
)

print("\nTopFeature Importance")
print(50 * '=')

print(feat_importance_df.to_string(index=False))

# ============================================================
# Group Feature Importance by Original Dataset Features
# ============================================================

# Create a mapping between encoded features and original features

feature_mapping = {}

for feature in categorical_features:

    prefix = f'categorical__{feature}_'

    for encoded_feature in feature_names:

        if encoded_feature.startswith(prefix):

            feature_mapping[encoded_feature] = feature


for feature in numerical_features:

    feature_mapping[f'numerical__{feature}'] = feature


# Add original feature names to importance DataFrame

feat_importance_df['Original Feature'] = (
    feat_importance_df['Feature'].map(feature_mapping)
)


# Combine importance scores by original feature

grouped_importance = (
    feat_importance_df
    .groupby('Original Feature')['Importance']
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)


print("\nGrouped Original Feature Importance")
print(50 * '=')

print(grouped_importance.to_string(index=False))


# ============================================================
# VISUAL RESULTS
# ============================================================

# Create folder for visualizations
BASE_DIR = Path(__file__).resolve().parent
VISUAL_DIR = BASE_DIR / "visuals"
VISUAL_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------
# CONFUSION MATRIX
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 6))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["No Subscription", "Subscribed"],
    values_format="d",
    ax=ax
)

ax.set_title(
    "XGBoost Model - Confusion Matrix",
    fontsize=14,
    fontweight="bold"
)

ax.set_xlabel("Predicted Outcome")
ax.set_ylabel("Actual Outcome")

plt.tight_layout()

# Save confusion matrix
plt.savefig(
    VISUAL_DIR / "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ------------------------------------------------------------
# GROUPED FEATURE IMPORTANCE
# ------------------------------------------------------------

# Retrieve the trained XGBoost classifier
xgb_model = model_pipeline.named_steps['classifier']

# Retrieve feature names after preprocessing
feature_names = (
    model_pipeline[:-1].get_feature_names_out()
)

# Retrieve feature importance scores
importance_scores = xgb_model.feature_importances_

# Match importance scores to encoded feature names
importance_df = pd.DataFrame({
    "Encoded Feature": feature_names,
    "Importance": importance_scores
})

# Original feature names from the dataset
original_features = list(X.columns)

# Map encoded features back to original dataset features
def get_original_feature(encoded_name):

    cleaned_name = encoded_name.split("__", 1)[-1]

    matches = [
        feature for feature in original_features
        if cleaned_name == feature
        or cleaned_name.startswith(feature + "_")
    ]

    if matches:
        return max(matches, key=len)

    return cleaned_name


importance_df["Original Feature"] = (
    importance_df["Encoded Feature"]
    .apply(get_original_feature)
)

# Aggregate one-hot encoded features
grouped_importance = (
    importance_df.groupby("Original Feature")["Importance"]
    .sum()
    .sort_values(ascending=False)
)

# Select the top 10 original features
top_features = grouped_importance.head(10).sort_values()

# Create feature importance visualization
fig, ax = plt.subplots(figsize=(10, 7))

top_features.plot(
    kind="barh",
    ax=ax
)

ax.set_title(
    "Top 10 Features Influencing Term Deposit Predictions",
    fontsize=14,
    fontweight="bold"
)

ax.set_xlabel("Relative Feature Importance")
ax.set_ylabel("Original Dataset Feature")

plt.tight_layout()

# Save feature importance chart
plt.savefig(
    VISUAL_DIR / "feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

print("\nVisualizations successfully saved!")
print(f"Saved to: {VISUAL_DIR}")
