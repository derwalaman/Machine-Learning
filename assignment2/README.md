# Regression & Classification Project: Linear, Multiple Linear, Logistic Regression

## 🎯 Aim
To **implement and compare linear, multiple linear, and logistic regression models** using real datasets.

- **Linear Regression:** Predict house prices (Boston Housing Dataset).
- **Multiple Linear Regression:** Predict student final grade (Student Performance Dataset).
- **Logistic Regression:** Classify tumor type (Breast Cancer Dataset).

---

## 📚 Background Concepts

### 1. Linear Regression (Simple)
- Predicts a **continuous target** using a **single feature**.
- Formula:  
  \[
  y = a \cdot x + b
  \]
- Learn slope `a` and intercept `b` by minimizing squared errors.
- **Assumptions:**
  - Linear relationship
  - Independent errors
  - Homoscedasticity (equal variance)
  - Errors ~ Normal distribution (for inference)

---

### 2. Multiple Linear Regression
- Predicts a continuous target using **multiple features**.
- Formula:  
  \[
  y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p
  \]
- **Challenges:**
  - Multicollinearity (high correlation between features)
  - Overfitting when many predictors
- **Remedies:**
  - Remove correlated features
  - Regularization (Ridge, Lasso)

---

### 3. Logistic Regression
- Predicts **binary class (0/1)** using sigmoid function.
- Formula:  
  \[
  p = \frac{1}{1 + e^{-(\beta_0 + \beta^T x)}}
  \]
- Outputs probability `p` of class 1.
- Decision rule:  
  - If `p > 0.5` → class 1  
  - Else → class 0
- **Loss Function:** Cross-Entropy (Log Loss).
- **Interpretation:** Coefficients → odds ratios.

---

## 🛠 Methodology (Step by Step)

1. **Load and preprocess the dataset**
   - Handle missing values
   - Encode categorical variables
   - Scale numerical features (for logistic & regularization)

2. **Split into train/test sets**
   - Typical: 80% training, 20% testing
   - Use `stratify` for classification to preserve class ratios

3. **Train the model**
   - Linear Regression → OLS
   - Multiple Linear Regression → OLS with multiple features
   - Logistic Regression → Maximum likelihood (via log loss)

4. **Evaluate performance**
   - **Regression Metrics:**
     - MAE (Mean Absolute Error)
     - RMSE (Root Mean Squared Error)
     - R² Score (variance explained)
   - **Classification Metrics:**
     - Accuracy
     - Precision
     - Recall
     - F1 Score
     - ROC AUC

5. **Visualize results**
   - Regression:
     - Predicted vs Actual scatter
     - Residual plots, histograms
   - Classification:
     - Confusion matrix
     - ROC Curve
     - Precision-Recall Curve

---

## 📊 Datasets Used

- **Boston Housing Dataset (Linear Regression)**  
  - Predict median house price (`MEDV`) from features (e.g., RM = average rooms).
  - Input: numerical features like CRIM, RM, LSTAT, etc.

- **Student Performance Dataset (Multiple Linear Regression)**  
  - Predict final grade (`G3`) from demographics, study habits, and prior grades.
  - Input: mix of categorical & numeric features.

- **Breast Cancer Dataset (Logistic Regression)**  
  - Classify tumor as **benign (0)** or **malignant (1)**.
  - Input: 30 numeric features from cell nucleus measurements.

---

## 📏 Evaluation Metrics

### Regression
- **MAE:** Mean of absolute errors.  
- **RMSE:** Square root of mean squared errors. Penalizes large errors.  
- **R²:** Proportion of variance explained (1 = perfect).

### Classification
- **Accuracy:** Fraction of correctly classified samples.  
- **Precision:** True positives / (True positives + False positives).  
- **Recall (Sensitivity):** True positives / (True positives + False negatives).  
- **F1 Score:** Harmonic mean of precision & recall.  
- **ROC AUC:** Area under ROC curve (probability model ranks a positive higher than a negative).

---

## 📈 Visualizations

- **Regression**
  - Predicted vs Actual scatterplot
  - Residuals vs Predicted
  - Histogram of residuals

- **Classification**
  - Confusion Matrix
  - ROC Curve
  - Precision-Recall Curve

---

## ⚙️ Advanced Notes

- **Regularization**
  - Ridge (L2): Shrinks coefficients, good for multicollinearity.
  - Lasso (L1): Performs feature selection (zeroes out coefficients).
  - ElasticNet: Combination of Ridge + Lasso.

- **Cross-Validation**
  - k-Fold CV for regression
  - Stratified k-Fold CV for classification

- **Pipelines**
  - Ensure preprocessing & modeling are combined
  - Prevents data leakage from test set

- **VIF (Variance Inflation Factor)**
  - Detects multicollinearity in regression models.

---

## 🧪 Results (Expected)

| Model                         | Dataset              | Metrics Example (Test)              |
|-------------------------------|----------------------|--------------------------------------|
| **Simple Linear Regression**  | Boston Housing       | R² ≈ 0.48, RMSE ≈ 5.5                |
| **Multiple Linear Regression**| Student Performance  | R² ≈ 0.80, RMSE ≈ 1.5                |
| **Logistic Regression**       | Breast Cancer        | Accuracy ≈ 0.96, F1 ≈ 0.96, ROC AUC ≈ 0.99 |

(*Results may vary depending on preprocessing & splits*)

---

## 💡 Key Takeaways
- Linear regression is interpretable but limited with single features.  
- Multiple linear regression improves predictions but may suffer from multicollinearity.  
- Logistic regression is powerful for binary classification and provides probabilities.  
- Regularization and proper preprocessing (scaling, encoding) greatly improve stability.  
- Always validate models with proper metrics and visualization.  

---

## 🚀 How to Run
1. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib scikit-learn statsmodels requests joblib
   ```

2. Open regression_classification.ipynb in Jupyter or run step-by-step .py files.

3. Run each section (A, B, C) independently:
   - A: Linear Regression (Boston Housing)
   - B: Multiple Linear Regression (Student Performance)
   - C: Logistic Regression (Breast Cancer)

4. Check metrics & plots for evaluation.

5. Compare models in summary table.