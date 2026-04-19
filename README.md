# README — Code Submission for *Computational Materials Science*

---

## Overview

This repository contains the complete machine learning pipeline used for predicting the mean precipitate radius (`R_mean`) from phase-field simulation data of alloy coarsening systems with varying lattice mismatch (δ). The pipeline is divided into four sequential parts, each saved as a standalone Python script.

---

## Dataset

- **File:** `DATASET.csv`
- **Target variable:** `R_mean` (mean precipitate radius)
- **Key independent variables:** `time`, `delta` (lattice mismatch), `N_particles`, `Es_mean`, `F_mean`, `Al_std`, `W_std`, `OP1_mean`–`OP4_mean`, and additional microstructural descriptors
- The dataset contains time-series simulation data across multiple lattice mismatch values (δ)

---

## Requirements

### Python Version
Python 3.8 or higher

### Required Libraries

```
pandas
numpy
matplotlib
seaborn
scipy
scikit-learn
xgboost
lightgbm
joblib
```

Install all dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn xgboost lightgbm joblib
```

---

## Code Structure

The pipeline consists of four parts that must be run **in sequence**:

```
Part1_EDA.py
Part2_Preprocessing.py
Part3_Model_Training.py
Part4_Final_Evaluation.py
```

---

## Part 1 — Exploratory Data Analysis (`Part1_EDA.py`)

**Purpose:** Load the dataset and perform comprehensive exploratory analysis of the target variable and features.

**Steps performed:**
1. Load `DATASET.csv`
2. Inspect dataset shape, data types, and memory usage
3. Compute descriptive statistics for `R_mean`
4. Visualize target distribution (histogram, box plot, Q-Q plot, CDF, violin plot by δ)
5. Check data quality: missing values, duplicates, infinite values, constant columns
6. Compute Pearson correlations of all numerical features with `R_mean`
7. Detect outliers using the IQR method
8. Analyze `R_mean` statistics grouped by δ
9. Summarize time range per δ value

**Outputs:**
- `01_Target_Distribution_Analysis.png`
- `02_Feature_Correlation_with_Target.png`
- `data_after_eda.csv` *(input for Part 2)*

---

## Part 2 — Data Preprocessing and Feature Engineering (`Part2_Preprocessing.py`)

**Purpose:** Engineer new features, clean data, encode categorical variables, remove redundant features, and create train/validation/test splits.

**Input:** `data_after_eda.csv`

**Steps performed:**
1. Identify and exclude non-predictive columns (`FileID`, `row_id`, `original_delta_file`) and target-adjacent columns (`R_std`, `R_min`, `R_max`)
2. Engineer derived features:
   - `dR/dt` — coarsening rate (time derivative of `R_mean`)
   - `dN/dt` — particle dissolution rate
   - `Es/F ratio` — elastic to free energy ratio
   - `composition_gradient` — combined composition standard deviation
   - `OP_variance` — order parameter variance across OP1–OP4
   - `particle_density_norm` — normalized particle count
   - `time_squared`, `time_cubed`, `log_time` — time-based nonlinear features
   - `delta_time`, `delta_N`, `delta_Es` — interaction terms
   - `cumulative_time` — within-group cumulative time
   - Lag features (1-step lag) for `N_particles`, `Es_mean`, `F_mean`
3. Outliers in `R_mean` retained (physically meaningful)
4. Missing values filled by forward-fill → backward-fill → group median
5. One-hot encoding of categorical features; constant-value categorical columns dropped
6. Remove features with variance < 0.01
7. Remove features with pairwise Pearson correlation > 0.95
8. Quick Random Forest (100 trees, `max_depth=10`) for feature importance ranking
9. Stratified train/validation/test split (70% / 15% / 15%) stratified by δ

**Outputs:**
- `X_train.csv`, `X_val.csv`, `X_test.csv`
- `y_train.csv`, `y_val.csv`, `y_test.csv`
- `feature_names.txt`
- `03_Feature_Importance_RF.png`

---

## Part 3 — Feature Scaling and Model Training (`Part3_Model_Training.py`)

**Purpose:** Scale features and train multiple regression models; compare performance on the validation set.

**Input:** `X_train.csv`, `X_val.csv`, `X_test.csv`, `y_train.csv`, `y_val.csv`, `y_test.csv`

**Scaling:** `StandardScaler` (zero mean, unit variance) fit on training set only; saved as `scaler.pkl`

**Models trained:**

| # | Model |
|---|-------|
| 1 | Linear Regression |
| 2 | Ridge Regression |
| 3 | Lasso Regression |
| 4 | ElasticNet |
| 5 | Decision Tree |
| 6 | Random Forest |
| 7 | Extra Trees |
| 8 | Gradient Boosting |
| 9 | XGBoost |
| 10 | LightGBM |
| 11 | SVR (RBF kernel) |
| 12 | K-Nearest Neighbors |

**Evaluation metrics:** RMSE, MAE, R², MAPE, Max Error — computed on both training and validation sets

**Model selection:** Top 3 models ranked by validation R²; best model saved as `best_model.pkl`

**Outputs:**
- `model_comparison_results.csv`
- `scaler.pkl`
- `best_model.pkl`
- `model_<ModelName>.pkl` *(top 3)*
- `04_Model_Comparison.png`
- `05_Prediction_vs_Actual.png`
- `06_Residual_Analysis.png`

---

## Part 4 — Final Model Evaluation (`Part4_Final_Evaluation.py`)

**Purpose:** Perform cross-validation, retrain the best model on the combined train+validation set, and evaluate on the held-out test set with detailed error analysis by δ.

**Input:** All CSV splits, `scaler.pkl`, `best_model.pkl`, `model_comparison_results.csv`, `DATASET.csv`

**Steps performed:**
1. Load all preprocessed data and the best model from Part 3
2. 5-fold cross-validation on the combined train+validation set; report mean ± std for R² and RMSE
3. Retrain best model on train+validation combined
4. Final evaluation on held-out test set: R², RMSE, MAE
5. Per-δ error breakdown: R² and RMSE for each lattice mismatch value
6. Visualizations: predicted vs. actual, residual plot, residual histogram, per-δ error box plot

**Outputs:**
- `final_model.pkl`
- `final_predictions.csv` *(columns: `Actual`, `Predicted`, `Error`, `Abs_Error`, `Delta`)*
- `04_Final_Evaluation.png`

---

## Execution Order

Run the scripts strictly in the following order:

```bash
python Part1_EDA.py
python Part2_Preprocessing.py
python Part3_Model_Training.py
python Part4_Final_Evaluation.py
```

Each script reads output files produced by the previous step. Do not skip or reorder steps.

---

## Random Seed

All stochastic components use `random_state=42` for reproducibility.

---

## Notes

- The dataset (`DATASET.csv`) must be present in the working directory before running Part 1.
- Features directly derived from the target (`R_std`, `R_min`, `R_max`) are excluded in Part 2 to prevent data leakage.
- Outliers in `R_mean` are intentionally retained as they correspond to physically meaningful high-δ, late-time coarsening states.
- The train/validation/test split is stratified by δ to ensure all lattice mismatch values are represented in each subset.
