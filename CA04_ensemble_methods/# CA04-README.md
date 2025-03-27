# CA04 Ensemble Methods

This repository/notebook demonstrates how to build and evaluate various ensemble models (Random Forest, AdaBoost, Gradient Boost, and XGBoost) on a classification problem. We compare their performances in terms of **Accuracy** and **AUC** across different numbers of estimators.

---

## Overview

Ensemble methods combine multiple models to improve predictive performance compared to individual models. This notebook focuses on four popular ensemble algorithms:

- **Random Forest**  
- **AdaBoost**  
- **Gradient Boost**  
- **XGBoost**

We systematically vary the **n_estimators** hyperparameter for each model and measure two performance metrics on a test set:

- **Accuracy**  
- **AUC** (Area Under the ROC Curve)

---

## Dataset

The dataset used here contains categorical features. The categorical variabless have been **encoded** (some via ordinal encoding, others via one-hot encoding). We split the data into:

- **Training Set** (`df_train`): Used for fitting the models.  
- **Test Set** (`df_test`): Used for evaluating final performance.

To access the dataset, read the following csv into your code: https://github.com/ArinB/MSBA-CA-03-Decision-Trees/blob/master/census_data.csv?raw=true'

**Important Note on Column Names**:  
- For XGBoost, columns must not contain special characters like `[`, `]`, or `<`. We rename them if necessary.

---

## Models and Experiments

1. **Random Forest**  
   - We vary `n_estimators` in `[50, 100, 150, 200, 250, 300, 350, 400, 450, 500]`.  
   - Measure and plot **Accuracy** and **AUC**.  
   - Identify the best `n_estimators`.

2. **AdaBoost**  
   - Similar approach: vary `n_estimators`, record **Accuracy** and **AUC**.

3. **Gradient Boost**  
   - Use `GradientBoostingClassifier` with the same range of `n_estimators`.  
   - Plot and find the best number of estimators.

4. **XGBoost**  
   - Use `XGBClassifier` with the same procedure.  
   - **Ensure** columns do not contain forbidden characters.

---

## Usage

1. **Clone or Download** the repository/notebook.  
2. **Install Dependencies** (see below).  
3. **Open** the notebook in Jupyter.  
4. **Run the Cells** sequentially:
   - Data loading & encoding.  
   - Train/Test split.  
   - Model training loops for each ensemble method.  
   - Plotting Accuracy and AUC vs. `n_estimators`.  
   - Summarizing best results.

5. **Interpret the Plots and Results** to decide which model and hyperparameter settings are best for your use case.

---

## Dependencies

The notebook uses the following major Python libraries:

- **Python** 3.7+  
- **NumPy**  
- **Pandas**  
- **scikit-learn**  
- **Matplotlib**  
- **XGBoost** (for the XGBClassifier)  

You can install them via:

```bash
pip install numpy pandas scikit-learn matplotlib xgboost
