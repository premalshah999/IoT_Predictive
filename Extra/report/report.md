# Smart Manufacturing IoT Anomaly Detection & Predictive Maintenance

## Introduction

This project tackles the problem of **detecting anomalies**, **predicting maintenance requirements**, **estimating downtime risk**, **forecasting remaining useful life (RUL)** and **classifying the type of failure** in a smart manufacturing environment.  The data comes from the Smart Manufacturing IoT–Cloud Monitoring dataset consisting of 100 000 records from 50 machines sampled once per minute.  Each record includes sensor readings (temperature, vibration, humidity, pressure, energy consumption) along with labels such as `anomaly_flag`, `maintenance_required`, `downtime_risk`, `predicted_remaining_life` and `failure_type`.

The objective is to build a comprehensive pipeline that:

* Generates rich, TSFRESH‑style features from raw sensor streams.
* Evaluates a diverse set of machine‑learning models using stratified **k‑fold cross‑validation**.
* Compares performance across multiple tasks and selects suitable models for deployment.
* Provides a modular codebase and notebooks that can be extended or adapted to other datasets.

## Data Overview

The dataset contains the following columns:

- **timestamp**: datetime index at one‑minute resolution
- **machine_id**: identifier of the machine (1–50)
- **temperature**, **vibration**, **humidity**, **pressure**, **energy_consumption**: sensor readings
- **machine_status**: categorical operational mode (0=off, 1=normal, 2=alert)
- **anomaly_flag**: binary indicator of whether an anomaly was observed
- **predicted_remaining_life**: integer estimation of cycles until failure (1–499)
- **failure_type**: categorical cause of failure (Normal, Vibration Issue, Overheating, Pressure Drop, Electrical Fault)
- **downtime_risk**: float indicator of risk of near‑term downtime (0 or 1 after thresholding)
- **maintenance_required**: binary indicator that maintenance is required

Anomalies comprise roughly 9 % of the records, and maintenance events occur in roughly 20 % of the records.  The `failure_type` distribution is imbalanced: the majority of records are labeled **Normal**, while the other categories occur less frequently.

## Feature Engineering

We built a feature engineering module (`scripts/feature_engineering.py`) to emulate the functionality of TSFRESH without relying on the external library.  The module computes:

* **Basic transformations** – square, square‑root and log1p of each sensor.
* **Z‑scores** – normalised readings per machine to capture deviations from machine‑specific baselines.
* **Row statistics** – mean, standard deviation, min, max and range of the sensor vector.
* **Interaction terms** – products and ratios such as temperature × vibration and temperature / pressure.
* **Temporal encodings** – cyclical sine/cosine representation of hour of day, day of week and day of month.

These transformations yield approximately 110 features per record.  During model training, only the engineered features (not the raw sensor columns) are used.  Feature selection is performed implicitly by the tree‑based models.

## Modelling Approach

For each target variable we trained a suite of classical models:

- **Logistic Regression** (baseline linear model)
- **Random Forest**
- **Extra Trees**
- **Gradient Boosting**
- **AdaBoost**
- **Gaussian Naïve Bayes**
- **XGBoost**
- **LightGBM**
- **CatBoost**

For regression on `predicted_remaining_life` we used the corresponding regressor variants.  Hyperparameters were left mostly at their defaults for parity across models.  We employed **3‑fold stratified cross‑validation** to estimate the generalisation performance.  Metrics reported include mean **F1‑score**, **precision**, **recall** and **accuracy** for classification tasks and mean **MAE**, **RMSE** and **R²** for regression.

A hybrid deep‑learning module (`scripts/train_lstm_xgb_ensemble.py`) is also supplied.  It demonstrates how to extract sequence embeddings from time windows via a PyTorch LSTM and then fit an XGBoost classifier on top.  This module was not executed in this environment due to resource constraints, but it serves as a foundation for more advanced experiments.

## Cross‑Validation Results

Summary results for each task are provided in the `results/` folder.  Key observations:

### Anomaly Classification (`anomaly_flag`)

All tree‑based ensembles achieved near‑perfect performance, with F1-scores exceeding 0.997.  Random Forest, Gradient Boosting, AdaBoost, LightGBM and CatBoost all reached an F1 of roughly **0.9993**.  Logistic Regression and Extra Trees were slightly lower but still above **0.96**.

### Maintenance Prediction (`maintenance_required`)

The top models (LightGBM and Random Forest) achieved F1-scores around **0.984**, correctly identifying the majority of maintenance events.  Logistic Regression performed reasonably well at **0.898**, while Gaussian NB lagged behind due to its simplistic assumptions.

### Downtime Risk Classification (`downtime_risk`)

This binary task proved easier: Random Forest, Gradient Boosting, AdaBoost and XGBoost all achieved **perfect F1 = 1.0**, with Logistic Regression at **0.966** and Gaussian NB at **0.924**.

### Failure Type Classification (`failure_type`)

Despite class imbalance, multiple models exceeded F1 = 0.90.  Logistic Regression reached **0.928**, CatBoost **0.926**, and Gaussian NB **0.923**.  Tree‑based ensembles performed slightly lower on average but remained competitive.  Misclassifications primarily arose among the minor classes (Overheating, Pressure Drop and Electrical Fault).

### Remaining Life Regression (`predicted_remaining_life`)

The regression models achieved modest predictive power.  CatBoostRegressor and GradientBoostingRegressor produced mean MAE of approximately **113 cycles** and RMSE around **136**, corresponding to R² values near **0.15**.  Random Forest and LightGBM regressed less accurately, with R² around **0.05–0.14**.  Predicting precise remaining life from this dataset remains challenging.

## Discussion

* **Model selection** – For binary tasks, tree‑based ensembles (Random Forest, Gradient Boosting, AdaBoost, LightGBM, CatBoost) consistently outperformed linear models.  Given their speed and interpretability, **Random Forest** or **LightGBM** serve as strong default choices for deployment.
* **Class imbalance** – Minority classes in `failure_type` required care.  Stratified cross‑validation mitigated some bias, and weighted metrics were used.  Future work could explore cost‑sensitive learning or oversampling techniques to further improve recall on rare failure modes.
* **Feature richness** – Hand‑crafted features captured the dominant patterns, enabling high accuracy on anomaly and maintenance tasks.  Additional deep learned features (e.g. via the provided LSTM module) could improve performance on long‑horizon predictions like remaining life.
* **Real‑time deployment** – The trained models are lightweight and can be deployed on edge devices.  The feature engineering pipeline operates on sliding windows with minimal memory overhead.  Docker configuration facilitates reproducible deployment.

## Conclusion

This project demonstrates a full pipeline for detecting anomalies and predicting maintenance in IoT sensor data.  By leveraging TSFRESH‑style features and evaluating a broad spectrum of models under cross‑validation, we identified robust solutions for multiple predictive tasks.  While classical ensembles suffice for most targets, the included LSTM + XGBoost script lays the groundwork for incorporating sequence models where required.  The modular codebase, notebooks and Docker configuration enable easy reproduction and extension of these results.