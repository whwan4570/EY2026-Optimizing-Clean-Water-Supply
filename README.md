# 🚰 EY 2026 – Optimizing Clean Water Supply

This repository contains the final modeling workflow and artifacts for the **EY 2026 Clean Water Supply Challenge**, where the goal was to predict key water quality indicators using geospatial, environmental, and remote sensing data.

## 🏆 Final Result

- **Final best submission file:** `submission_final_F4.csv`
- **Final best leaderboard score:** **0.4259**
- **Final leaderboard rank:** **66**

This final submission was selected after multiple rounds of experimentation and leaderboard-based validation.

## 🔍 Challenge Overview

The task was to predict three important water quality parameters:

- **Total Alkalinity (TA)**
- **Electrical Conductance (EC)**
- **Dissolved Reactive Phosphorus (DRP)**

Among these targets, **DRP was the most difficult to model** and became the main focus of later experiments.

## ⚙️ Modeling Approach

The final solution used a **target-specific modeling pipeline**:

- Separate models for **TA** and **EC**
- A **two-stage DRP model** using predicted TA and EC as additional inputs

Main techniques included:

- Spatially aware validation using **GroupKFold / regional holdout**
- Feature engineering from:
  - Landsat-derived indices
  - TerraClimate variables
  - ERA5 and precipitation anomaly features
  - HydroRIVERS and external environmental features
- Fold-safe preprocessing to reduce data leakage
- Iterative target-specific tuning for better generalization

## 🧪 Project Notes

This project went through many DRP-focused experiments, including:

- Feature selection and preprocessing refinements
- Spatial validation strategy adjustments
- Prior-informed DRP modeling
- Submission-based comparison across multiple variants

The final model was chosen based on **actual leaderboard performance**, not only local validation metrics.

## 📂 Repository Cleanup

For GitHub upload, all intermediate submission files were removed and only the final best submission file was kept.

- **Kept:** `submission_final_F4.csv`
- **Removed:** all other `submission*.csv` files

This repository is intentionally cleaned up to highlight the final result and main workflow.

## 📄 Main Script

- **Main training/inference script:** `run_benchmark_notebook.py`

## 💡 Summary

This project demonstrates:

- End-to-end machine learning workflow development
- Spatially aware model validation for environmental prediction
- Iterative experimentation guided by leaderboard feedback
- Practical handling of a difficult target variable under distribution shift

## 👤 Author

**Wonjoon Hwang**  
**Daehyun Lee**  
**Doyoung Jung**  
M.S. in Data Science, University of Washington
