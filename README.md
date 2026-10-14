# HIT140 Assignment 3 Code Repository

This folder contains the reproducible code and data used for the FIFA World Cup 2026 linear regression analysis.

## Main files

- `00_Data_Preparation.ipynb` explains the sources, cleaning steps, pre-match history and feature construction.
- `01_Task21_Goal_Difference.ipynb` predicts the signed goal difference between the two listed teams.
- `02_Task22_Team_Goals.ipynb` predicts goals scored by an individual team.
- `analysis.py` rebuilds the processed data, fits the OLS models and writes the results.
- `prematch.py` constructs the historical pre-match predictors.
- `verify_analysis.py` runs integrity and leakage checks.
- `requirements.txt` lists the Python packages.
- `data/raw` contains the retained source files and source metadata.
- `data/processed` contains regenerated modelling tables and feature lineage.
- `results` contains the model metrics, equations, summaries and figures.

## Run order

From this folder, install the packages from `requirements.txt`, run `python analysis.py`, and then run `python verify_analysis.py`. Open the three Notebooks after the analysis has completed.

The code compares a one-predictor simple OLS model with an eight-predictor multiple OLS model using the same test rows. It also evaluates the required eight-predictor model against a training-mean baseline and reports MAE, MSE, RMSE, NRMSE and R-squared.

