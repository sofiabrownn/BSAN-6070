# CA03

## Overview
This project involves exploratory data analysis (EDA) and decision tree classification using census data. The dataset is loaded, analyzed, preprocessed, and used to train a Decision Tree model for classification. Model performance is evaluated using various metrics, and hyperparameter tuning is performed to improve accuracy.

## Dependencies
Ensure the following Python libraries are installed before running the script:
```python
pandas
numpy
sklearn
autoviz
```
You can install them using:
```bash
pip install pandas numpy scikit-learn autoviz
```

## Dataset
The dataset used in this project is sourced from:
```
https://raw.githubusercontent.com/ArinB/MSBA-CA-03-Decision-Trees/refs/heads/master/census_data.csv
```
It contains both categorical and numerical features, which are preprocessed before training the model.

## Steps in the Script

### 1. Data Loading & Initial Quality Check
- The dataset is loaded into a Pandas DataFrame.
- The `info()` method is used to inspect data types and check for missing values.
- Basic statistics are gathered using `describe(include="all")`.

### 2. Exploratory Data Analysis (EDA)
- `AutoViz` is used to generate visual insights on the dataset.
- Custom descriptive statistics functions generate additional insights on categorical and numerical features.

### 3. Data Preprocessing
- Encoding categorical features using `LabelEncoder`.
- Splitting data into training (`df_train`) and testing (`df_test`) based on a `flag` column.
- Separating features (`X_train`, `X_test`) and target variable (`y_train`, `y_test`).

### 4. Decision Tree Model Training & Evaluation
- A `DecisionTreeClassifier` is initialized with hyperparameters such as:
  - `max_depth=10`
  - `min_samples_leaf=5`
  - `random_state=101`
- The model is trained on the training set and evaluated using:
  - Accuracy score
  - Precision, Recall, F1-score (classification report)
  - Confusion matrix

### 5. Hyperparameter Tuning
- Multiple sets of hyperparameters are tested to improve model performance.
- Results are stored in a DataFrame and printed for comparison.

## Outputs
- Model performance metrics including accuracy, precision, recall, and F1-score.
- Confusion matrix for evaluating classification results.
- Comparative analysis of different hyperparameter configurations.

## Environment
The script was coded and run in Google Colab. Modify hyperparameters as needed to optimize performance.

## Author
Developed as part of a Machine Learning project assignment on Decision Trees.

