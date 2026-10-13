# online-payment-fraud-detection-ml-anomaly-detection
# Online Payment Fraud Detection Using Machine Learning and Anomaly Detection

## Project Overview

This project investigates how effectively machine-learning and anomaly-detection techniques can identify fraudulent online payment transactions within highly imbalanced transaction data.

Four approaches are compared:

- Logistic Regression
- Random Forest
- XGBoost
- Isolation Forest

The project includes exploratory data analysis, feature engineering, class-imbalance treatment, chronological validation, threshold optimisation, model evaluation, SHAP-based explainability, automated artefact testing and development of a Streamlit fraud-risk decision-support prototype.

Because fraudulent transactions represent only a small proportion of the data, model performance is evaluated using fraud-sensitive measures such as precision, recall, F1-score, ROC-AUC and particularly PR-AUC rather than relying on accuracy alone.

## Research Question

**How effectively can machine-learning and anomaly-detection techniques identify fraudulent online payment transactions within highly imbalanced datasets?**

## Project Objectives

1. Audit, preprocess and engineer transaction-level features.
2. Compare Logistic Regression, Random Forest, XGBoost and Isolation Forest.
3. Evaluate the original distribution, class weighting, random undersampling and SMOTE.
4. Optimise models and decision thresholds without using the official test set.
5. Evaluate models using precision, recall, F1-score, ROC-AUC and PR-AUC.
6. Apply SHAP and error analysis to explain model behaviour.
7. Develop and test a Streamlit fraud-risk research prototype.

## Dataset

The project uses the **Credit Card Transactions Fraud Detection Dataset generated using the Sparkov simulator**.

The training dataset contains:

- **1,289,169 legitimate transactions**
- **7,506 fraudulent transactions**
- **Fraud prevalence: 0.5789%**

The raw dataset is highly imbalanced, which makes fraud detection more challenging and makes overall accuracy an inappropriate primary evaluation measure.

The raw dataset files are not included in this repository because of their size.

## Exploratory Data Analysis

The exploratory analysis investigates:

- class distribution;
- transaction-amount distribution;
- fraud rate by merchant category;
- fraud rate by transaction hour;
- fraud rate by day of week;
- monthly fraud rate; and
- differences between training and official-test fraud prevalence.

## Feature Engineering

The modelling pipeline uses transaction-level behavioural, temporal and geographical information.

Important features include:

- `amt`
- `age`
- `distance_km`
- `hour`
- `weekday`
- `month`
- `weekend`
- `category`
- `gender`
- `state`
- `city_pop`

Customer-to-merchant distance is derived from geographical coordinates using the Haversine calculation.

## Class-Imbalance Treatment

The project investigates several approaches:

- original class distribution;
- class weighting;
- random undersampling; and
- SMOTE.

Resampling is restricted to training data so that validation and test distributions remain representative of the original transaction data.

## Machine-Learning Models

### Logistic Regression

Logistic Regression provides an interpretable linear baseline.

### Random Forest

Random Forest provides a non-linear ensemble approach capable of capturing interactions between transaction characteristics.

### XGBoost

XGBoost provides a gradient-boosting approach for learning complex fraud patterns.

### Isolation Forest

Isolation Forest provides an unsupervised anomaly-detection benchmark and allows abnormality detection to be compared with supervised fraud classification.

## Validation Strategy

The project uses chronological development and validation to reduce optimistic evaluation.

Model selection is primarily based on **PR-AUC**, with **F1-score** used as a secondary criterion.

The official test dataset remains isolated until:

- model selection is complete;
- the imbalance strategy is selected;
- hyperparameters are fixed; and
- the operating threshold is fixed.

This reduces the risk of test-set leakage.

## Evaluation Metrics

The main evaluation measures are:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC / Average Precision
- Confusion Matrix

PR-AUC is especially important because fraud is a rare class.

## Threshold Optimisation

The default classification threshold of `0.50` is not automatically assumed to be optimal.

Different thresholds are evaluated using validation data to study the trade-off between:

- fraud detection;
- false positives;
- precision;
- recall; and
- F1-score.

The final operating threshold is selected without optimising against the official test set.

## Explainability

SHAP is used to analyse the behaviour of the selected model.

Important predictive features include:

- transaction amount;
- merchant category;
- transaction hour;
- customer age;
- geographical distance; and
- customer-location characteristics.

SHAP results are interpreted as explanations of model behaviour rather than evidence of causality.

## Streamlit Fraud-Risk Prototype

A Streamlit application was developed as a research decision-support prototype.

The interface provides:

- transaction input controls;
- fraud-risk score;
- selected decision threshold;
- high-risk or low-risk output;
- model-performance information;
- explainability information; and
- project limitations.

The prototype is intended for research demonstration and should not be used to automatically block genuine financial transactions.

## Artefact Testing

Automated tests were created to verify technical consistency.

Testing includes:

- model-file availability;
- metadata availability;
- threshold validity;
- prediction-score validity;
- prediction reproducibility;
- boundary-value handling;
- missing-value handling;
- unseen-category handling; and
- explainability-output availability.

The current project evidence records **12 automated tests as passing**.

A separate manual Streamlit test plan covers interface behaviour and should be completed with actual results, Pass/Fail status and screenshot evidence before final submission.

## Project Structure

```text
online-payment-fraud-detection-ml/
│
├── README.md
├── requirements.txt
├── model_metadata.json
├── app.py
├── notebook.ipynb
│
├── models/
│
├── outputs/
│   ├── metrics/
│   ├── plots/
│   ├── thresholds/
│   ├── testing/
│   └── project_management/
│
└── screenshots/




The exact filenames should be adjusted to match the final repository.
Main Outputs
Typical project outputs include:
precision_recall_curves.png
roc_curves.png
random_forest_confusion_matrix.png
xgboost_confusion_matrix.png
logistic_regression_confusion_matrix.png
isolation_forest_confusion_matrix.png
random_forest_threshold_tradeoff.png
xgboost_threshold_tradeoff.png
logistic_regression_threshold_tradeoff.png
isolation_forest_threshold_tradeoff.png
top_feature_importance.png
train_test_fraud_rate.png
experiment_log.csv
project_decision_log.csv
automated_artifact_test_results.csv
manual_streamlit_test_plan.csv
model_metadata.json
requirements.txt
shap_feature_importance.csv

Running the Project
Google Colab
1. Open the project notebook in Google Colab.
2. Install/import the required libraries.
3. Upload the training and official test datasets.
4. Run preprocessing and exploratory analysis.
5. Train the candidate models.
6. Review validation results.
7. Select the model and operating threshold.
8. Run final evaluation after all development decisions are frozen.
9. Download the generated output package.
CHECK Mode and FINAL Mode
CHECK Mode
CHECK mode is intended to confirm that:
- the notebook executes correctly;
- preprocessing works;
- models train correctly;
- plots and outputs are generated;
- artefact files can be created; and
- the final output package can be produced.
CHECK-mode outputs should not automatically be treated as submission-final empirical evidence.
FINAL Mode
FINAL mode should be used after the modelling configuration has been frozen.
After final evaluation begins, no further changes should be made to:
- the selected model;
- imbalance strategy;
- hyperparameters; or
- decision threshold.
Installation
Install dependencies with:
pip install -r requirements.txt

Main libraries include:
pandas
numpy
matplotlib
scikit-learn
imbalanced-learn
xgboost
shap
joblib
streamlit

Running the Streamlit Application
If app.py is in the repository root:
streamlit run app.py

If it is stored in an app folder:
streamlit run app/app.py

Reproducibility
The project supports reproducibility through:
- fixed random seeds;
- saved preprocessing and modelling pipelines;
- stored model metadata;
- recorded operating thresholds;
- experiment logs;
- project decision logs;
- threshold-analysis outputs;
- automated testing; and
- dependency records.
Limitations
The project has several limitations:
- the Sparkov dataset is simulated;
- the prototype has not been tested in a production banking environment;
- real financial costs of false positives and false negatives are unavailable;
- SHAP explanations do not demonstrate causality;
- fraud behaviour may change over time; and
- production use would require monitoring, security, governance and independent validation.
Legal, Social and Ethical Considerations
Real-world fraud detection would require careful consideration of:
- privacy;
- data minimisation;
- secure access;
- fairness;
- subgroup performance;
- explainability;
- false-positive impact;
- false-negative risk;
- human oversight;
- auditability;
- customer redress; and
- model monitoring.
The developed artefact is therefore presented as a research prototype rather than an autonomous financial decision-making system.
Key Findings
The experimental work indicates that supervised ensemble approaches provide stronger fraud discrimination than the linear baseline and standalone anomaly-detection approach.
Random Forest and XGBoost perform particularly strongly under fraud-sensitive evaluation.
The project demonstrates that:
- accuracy alone is unsuitable for highly imbalanced fraud detection;
- PR-AUC provides important information about minority-class performance;
- threshold choice materially changes model behaviour;
- anomaly detection is not equivalent to supervised fraud classification;
- feature engineering contributes to predictive performance;
- explainability should not be interpreted causally; and
- leakage-aware validation is essential for credible model evaluation.
Future Work
Future work could investigate:
- real financial transaction data;
- longer observation periods;
- monetary cost-sensitive learning;
- probability calibration;
- subgroup fairness testing;
- concept-drift monitoring;
- adaptive thresholds;
- real-time processing;
- production monitoring; and
- hybrid supervised/anomaly-detection approaches.
Technologies Used
Python
Pandas
NumPy
Matplotlib
Scikit-learn
Imbalanced-learn
XGBoost
SHAP
Joblib
Streamlit
Google Colab
Git
GitHub

Academic Purpose
This repository was developed as part of an MSc research project investigating online payment fraud detection using machine learning and anomaly detection.
It is provided for educational and research purposes only.
Author
Student: Shashivardhan Chintha Kunta
Programme: MSc Computer Science
Project: Online Payment Fraud Detection Using Machine Learning and Anomaly Detection
Year: 2026
