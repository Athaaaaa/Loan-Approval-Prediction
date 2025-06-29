# Loan-Approval-Prediction

## Loan Approval Prediction with Stacked Models

This program implements a **machine learning pipeline** to predict loan approvals based on borrower profiles and loan application data. It follows a structured data science workflow: from data cleaning and feature engineering to model training and ensemble learning.

---

### Objectives

* Predict whether a loan should be approved based on applicant and loan characteristics
* Identify the most influential factors in approval decisions
* Build a robust, interpretable, and high-performance model using ensemble learning

---

###  Dataset

* Source: [Kaggle Credit Risk Dataset](https://www.kaggle.com/)
* Combined files: `credit_risk_dataset.csv`, `train.csv`, `test.csv`
* Size: \~91,000 records, 12 original features, 36 total features after engineering

---

###  Key Steps

1. **Data Exploration**

   * Distribution analysis, skewness, missing values, and outlier detection
   * Insight: income, loan amount, and interest rate are strong discriminators

2. **Preprocessing**

   * KNN imputation for missing values
   * Label encoding for categorical features
   * Outlier clipping via IQR filtering
   * Feature normalization (e.g., log income)

3. **Feature Engineering**
   Constructed 25+ new features from 7 perspectives:

   * Income status & loan ratio
   * Credit history & risk scores
   * Repayment capacity
   * Loan intent consistency
   * Interest rate sensitivity

4. **Model Training & Evaluation**

   * Base models: LightGBM, XGBoost, CatBoost, RandomForest, etc.
   * Final model: **Stacked Ensemble** with optimized hyperparameters
   * Evaluation: ROC AUC \~0.96 (5-fold CV & test set)
   * Feature importance visualized for interpretability

---

###  Final Output

* Optimized stacking classifier:
  `LGBM + XGBoost + CatBoost → Logistic Regression (meta-model)`
* ROC AUC: **0.9581**
* Prediction probabilities exported to `try20241118.csv`

---

###  Key Insights

* **Most influential features**:
  `loan_grade`, `person_income`, `loan_intent`, `home_ownership`, `loan_int_rate`
* **Business implication**:
  Creditworthiness and repayment ability are the most decisive factors in loan approval

---

### Project Structure

```bash
├── data/
│   ├── credit_risk_dataset.csv
│   ├── train.csv
│   └── test.csv
├── notebook/
│   └── Project1_part2_code.ipynb
├── output/
│   └── try20241118.csv
└── README.md
```

