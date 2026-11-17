# Term Deposit Marketing Prediction

### Predicting customer subscriptions with XGBoost


## Project Overview

This project develops a machine learning classification workflow to predict whether a bank customer is likely to subscribe to a term-deposit product following a marketing campaign.

The goal is to help a bank move beyond broad, one-size-fits-all outreach by identifying customer and campaign characteristics associated with a higher likelihood of subscription. A more focused approach can help marketing teams prioritize prospects, allocate campaign resources, and evaluate opportunities to improve conversion.

The solution uses **XGBoost**, supported by a Scikit-learn preprocessing pipeline, stratified cross-validation, class-imbalance experiments, and probability-threshold optimization.

## Business Problem

Bank marketing campaigns often contact a large number of customers, while only a small proportion ultimately subscribe. In this dataset, approximately 
**7.24%** of the test observations belong to the subscriber class.

This imbalance creates an important modeling challenge: a classifier can achieve high accuracy by frequently predicting that a customer will not subscribe, while still missing many actual subscribers.

The project therefore evaluates more than accuracy. Precision, recall, and F1 score are used to understand the trade-off between identifying likely subscribers and limiting false-positive prospects.

### Project objectives

- Predict whether a customer will subscribe to a term deposit (`yes` or `no`).
- Exceed the project benchmark of **81% average accuracy** using five-fold cross-validation.
- Investigate class imbalance and its effect on minority-class detection.
- Identify customer and campaign attributes that contribute to model predictions.
- Translate model outputs into practical considerations for campaign targeting.

## Results at a Glance

| Evaluation | Result |
|---|---:|
| Five-fold cross-validation accuracy | **92.69%** |
| Final test accuracy | **93.23%** |
| Test precision | **52.46%** |
| Test recall | **68.22%** |
| Test F1 score | **59.31%** |
| Selected probability threshold | **0.53** |
| Project accuracy benchmark | **81%** |

The five-fold cross-validation accuracy exceeded the project benchmark by **11.69 percentage points**. On the held-out test set, the selected threshold identified 395 of the 579 actual subscribers.

Because the positive class is relatively rare, the precision, recall, and F1 score provide important context alongside accuracy.

## Dataset and Target

The project uses the `term-deposit-marketing-2020.csv` dataset. The target variable is `y`, which indicates whether the customer subscribed to the term deposit.

- `yes` is mapped to `1` (subscriber).
- `no` is mapped to `0` (non-subscriber).

The remaining columns are used as model features. They include customer, financial, contact, and campaign-related information such as age, job, education, balance, housing status, contact channel, campaign month, call duration, and previous campaign outcomes.

The target is normalized by trimming whitespace and converting values to lowercase before mapping. The script also checks for missing or unmapped target values.

> **Class imbalance:** The held-out test set contains 7,421 non-subscribers and 579 subscribers (8,000 observations total). A model should therefore be assessed on its ability to identify subscribers, not accuracy alone.

## Technology Stack

- **Python** — project implementation
- **Pandas** — tabular data loading and manipulation
- **NumPy** — numerical operations and probability-threshold iteration
- **Scikit-learn** — preprocessing, pipelines, data splitting, cross-validation, and evaluation metrics
- **XGBoost** — gradient-boosted decision-tree classifier
- **Matplotlib** — visualization support where used

## Project Workflow

The workflow is organized to keep preprocessing, model selection, validation, and final testing distinct.

### 1. Load the data and define the prediction target

The dataset is loaded into a Pandas DataFrame. The feature matrix `X` contains all input columns except `y`, while `y` is converted from text labels to binary values.

This establishes a clear supervised-learning task: use customer and campaign information to predict subscription.

### 2. Split the data into training, validation, and test sets

The code first reserves **20%** of the data as a final holdout test set. The remaining 80% is then split into training and validation subsets, with 20% of that development portion allocated to validation.

`stratify=y` is used in both splits to preserve the relative proportion of subscribers and non-subscribers. A fixed `random_state=21` makes the splits reproducible.

The validation set is used for probability-threshold selection. The test set is held back until the final evaluation.

### 3. Identify numerical and categorical features

The input columns are separated by data type:

- Categorical columns are identified for imputation and one-hot encoding.
- Numerical columns are identified for median imputation.

This allows each feature type to receive an appropriate preprocessing strategy.

### 4. Build the preprocessing pipeline

A Scikit-learn `ColumnTransformer` combines two preprocessing pipelines:

**Categorical features**
1. Missing values are replaced with the most frequent category.
2. `OneHotEncoder` converts categories into numeric indicator columns.
3. `handle_unknown='ignore'` allows the pipeline to process categories that were not observed during fitting.

**Numerical features**
1. Missing values are replaced with the median.

The preprocessor is combined with XGBoost in a single Scikit-learn `Pipeline`. This is important during cross-validation: each fold fits its own preprocessing steps using that fold's training data, helping prevent information leakage from validation folds into preprocessing.

### 5. Establish stratified five-fold cross-validation

`StratifiedKFold` divides the training data into five folds while maintaining the target-class proportions in each fold.

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 score

The mean performance across folds provides a more informative estimate than relying on a single train/validation split.

### 6. Investigate class imbalance with `scale_pos_weight`

Because subscribers are the minority class, the project tests multiple `scale_pos_weight` settings in XGBoost.

The parameter increases the relative training penalty associated with errors on the positive class. The experiments compare several candidate weights, including the ratio of negative to positive observations in the training data.

For each candidate weight, the pipeline is evaluated with five-fold stratified cross-validation. The results are compared using F1 score, with its fold-to-fold standard deviation also recorded.

This makes the trade-off visible: increasing the positive-class weight can improve subscriber recall, but may also increase false positives and reduce precision or overall accuracy.

### 7. Select the class weight and train the development model

The class-weight comparison is sorted by mean cross-validation F1 score. The best-performing weight is then used to initialize the final XGBoost configuration.

The selected classifier is combined with the preprocessing pipeline and fitted on the training subset. The validation subset remains separate for threshold optimization.

### 8. Optimize the classification probability threshold

XGBoost's default classification decision generally uses a probability threshold of 0.50. However, the best threshold for a particular business problem may differ, especially when the target classes are imbalanced and the costs of false positives and false negatives are not equal.

The model generates subscriber probabilities for the validation data. The script tests thresholds from **0.20 to 0.70**, in increments of 0.01.

At each threshold:
- Probabilities at or above the threshold are classified as subscribers.
- Probabilities below the threshold are classified as non-subscribers.
- Accuracy, precision, recall, and F1 score are calculated.

The threshold with the highest validation F1 score is selected. In the reported final run, the selected threshold was **0.53**.

This step allows the classification decision to be tuned to the balance between precision and recall rather than accepting the default cutoff without evaluation.

### 9. Retrain and evaluate on the held-out test set

After selecting the threshold, the pipeline is refitted using the combined training and validation data. The test set remains unseen during model development and threshold selection.

The final model generates probabilities for the test observations, and the selected 0.53 threshold converts those probabilities into binary predictions.

The held-out evaluation reports accuracy, precision, recall, F1 score, a classification report, and a confusion matrix.

**Reported test confusion matrix**

Rows represent actual classes and columns represent predicted classes.

|  | Predicted: No | Predicted: Yes |
|---|---:|---:|
| **Actual: No** | 7,063 | 358 |
| **Actual: Yes** | 184 | 395 |

Interpretation:
- **7,063 true negatives:** non-subscribers correctly identified.
- **358 false positives:** customers predicted to subscribe who did not.
- **184 false negatives:** actual subscribers the model did not identify.
- **395 true positives:** subscribers correctly identified.

The model identified **395 of 579 actual subscribers**, corresponding to 68.22% recall.

### 10. Analyze feature importance

After final training, the XGBoost classifier is extracted from the Scikit-learn pipeline. The preprocessor supplies the transformed feature names, allowing feature-importance scores to be associated with the encoded inputs.

Because one-hot encoding can create multiple columns for a single original categorical variable, the script maps encoded columns back to their source features and aggregates their importance scores.

In the reported analysis, **month, duration, and job** were the three leading features and together accounted for approximately **70% of total feature importance**.

These results suggest that campaign timing, call engagement, and employment-related characteristics were influential in the model's predictions. Feature importance describes how the fitted model used variables; it does not establish that those variables cause a customer to subscribe.

## Business Interpretation

The model's predictions can be used to help prioritize campaign outreach for further testing.

At the selected threshold, the model flagged **753 customers** as potential subscribers:

- 395 were actual subscribers.
- 358 were false-positive prospects.
- The selected group had a subscription rate of **52.46%**, compared with approximately **7.24%** across the test set.

That is roughly **7.25 times the overall test-set subscription rate**. This concentration of subscribers suggests a potential opportunity to focus campaign resources on a smaller, higher-propensity group.

The feature-importance analysis also offers hypotheses for campaign planning:

- **Campaign month:** February, March, April, June, and October were among the months identified as influential in the model's predictions.
- **Contact channel:** Cellular contact and other contact-related patterns may warrant further investigation.
- **Customer profile:** Employment and education characteristics, as well as housing status, may help distinguish customer groups with different predicted subscription likelihoods.

These are model-derived associations, not guaranteed targeting rules. Before changing campaign strategy, the bank should validate these segments using observed conversion rates and controlled campaign tests.

### Connecting predictions to revenue

The model does not calculate revenue directly. It creates a potential targeting signal that can be connected to business outcomes by tracking:

- Number of customers contacted
- Subscription conversion rate
- Cost per contact and cost per acquired subscriber
- Net contribution or revenue per term-deposit subscription
- Incremental conversions compared with a control group

A campaign pilot or A/B test would help determine whether using model scores improves conversion and profitability after accounting for outreach costs.

## Important Limitation: Call Duration

Call duration was one of the strongest features in the model. However, duration is only known after a customer has been contacted.

This creates an important distinction between two possible use cases:

- **During or after a call:** duration may help analyze customer engagement and campaign outcomes.
- **Before the first call:** duration is unavailable and should not be used as a predictor.

Therefore, the reported model should not be interpreted as a fully deployable pre-call prospecting model. A production-oriented pre-call model should be retrained and evaluated without duration, using only information available before the outreach decision. Its performance should be reported separately.

## Key Skills Demonstrated

This project demonstrates practical experience with:

- End-to-end supervised machine learning workflows
- Data cleaning and target-variable preparation
- Numerical and categorical feature preprocessing
- Scikit-learn pipelines and `ColumnTransformer`
- XGBoost classification
- Stratified train/validation/test splitting
- Five-fold stratified cross-validation
- Class-imbalance investigation with `scale_pos_weight`
- Probability calibration at the decision-threshold level (threshold selection)
- Classification evaluation and confusion-matrix interpretation
- Feature-importance extraction and aggregation
- Translating model outputs into business recommendations

## Conclusion

This project demonstrates an end-to-end approach to applying machine learning to a practical marketing problem. I developed an XGBoost classification pipeline, built automated preprocessing for mixed data types, investigated class imbalance through cross-validation, optimized the classification threshold, and evaluated the final model on a held-out test set.

The final model achieved **92.69% average accuracy across five-fold cross-validation** and **93.23% accuracy on the held-out test set**, with **68.22% recall** for customers who subscribed. More importantly, the analysis translated model predictions into a potential campaign-prioritization strategy: the customers selected by the model had a substantially higher observed subscription rate than the overall test population.

The project also reinforced the importance of evaluating models in the context of how predictions will be used. In particular, identifying call duration as a highly influential but post-contact variable highlighted the need to distinguish between campaign analysis and genuine pre-call targeting.

For hiring managers and recruiters, this work reflects my ability to move through the full data science workflow—from preparing data and developing models to evaluating trade-offs, interpreting results, identifying limitations, and communicating potential business value.

## Repository Contents

```text
7gUhlgP9YyAVlpPI/
├── TDM.py
├── requirements.txt
├── visuals/
└── README.md
```

## Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/Vince-Ike/7gUhlgP9YyAVlpPI.git
cd 7gUhlgP9YyAVlpPI
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install numpy pandas scikit-learn xgboost matplotlib
```

### 4. Add the dataset and update the data path

Place `term-deposit-marketing-2020.csv` in the project directory or another accessible location. Update the `pd.read_csv()` path in `TDM.py` to match its location.

The current script uses a machine-specific absolute file path, so this change is necessary before running it on another computer.

### 5. Run the script

```bash
python TDM.py
```

The script prints cross-validation comparisons, class-weight results, threshold comparisons, final classification metrics, the confusion matrix, and feature-importance summaries.

## Notes

- The dataset is not listed among the repository files; obtain it separately and place it at the path configured in the script.
- The reported metrics correspond to the project run summarized above. Results can vary if the dataset, split, preprocessing, model settings, or threshold-selection procedure changes.
- For a reproducible deployment workflow, use a relative or configurable dataset path and persist the fitted preprocessing/model pipeline together with the selected decision threshold.
