# IoT Anomaly Detection - Mathematical Foundation

## Table of Contents
1. [Feature Engineering Mathematics](#feature-engineering)
2. [Model Mathematics](#model-mathematics)
3. [Evaluation Metrics](#evaluation-metrics)
4. [Cross-Validation](#cross-validation)

---

## Feature Engineering Mathematics {#feature-engineering}

### Raw Sensor Features (5 features)

The system starts with 5 raw sensor measurements at time $t$:

$$
\mathbf{x}_t = [x_1, x_2, x_3, x_4, x_5]^T
$$

Where:
- $x_1$ = Temperature (°C)
- $x_2$ = Vibration (Hz)
- $x_3$ = Humidity (%)
- $x_4$ = Pressure (bar)
- $x_5$ = Energy Consumption (kWh)

### 1. Basic Transformations (15 features)

#### Squared Features
$$
f_{squared}(x_i) = x_i^2, \quad i \in \{1, 2, 3, 4, 5\}
$$

**Purpose**: Capture non-linear relationships and amplify larger values.

**Example**:
$$
\text{temperature\_squared} = T^2
$$

#### Square Root Features
$$
f_{sqrt}(x_i) = \sqrt{x_i}, \quad i \in \{1, 2, 3, 4, 5\}
$$

**Purpose**: Compress the range of large values and reduce impact of outliers.

**Example**:
$$
\text{vibration\_sqrt} = \sqrt{V}
$$

#### Log Transform Features
$$
f_{log}(x_i) = \log(x_i + 1), \quad i \in \{1, 2, 3, 4, 5\}
$$

**Purpose**: Handle skewed distributions and stabilize variance. We add 1 to avoid $\log(0)$.

**Example**:
$$
\text{energy\_log} = \log(E + 1)
$$

---

### 2. Statistical Z-Score Features (5 features)

For each machine $m$ and sensor $i$, compute the z-score:

$$
z_{i,m}(t) = \frac{x_i(t) - \mu_{i,m}}{\sigma_{i,m}}
$$

Where:
- $\mu_{i,m} = \frac{1}{N_m} \sum_{t=1}^{N_m} x_i(t)$ is the mean of sensor $i$ for machine $m$
- $\sigma_{i,m} = \sqrt{\frac{1}{N_m} \sum_{t=1}^{N_m} (x_i(t) - \mu_{i,m})^2}$ is the standard deviation
- $N_m$ is the number of observations for machine $m$

**Purpose**: Normalize sensors relative to each machine's baseline, detecting deviations from normal operating conditions.

**Example**:
$$
\text{temperature\_zscore\_machine} = \frac{T_t - \bar{T}_m}{\sigma_{T,m}}
$$

---

### 3. Rolling Window Features (15 features)

#### Rolling Mean
For window size $w = 5$:

$$
\text{rolling\_mean}_i(t) = \frac{1}{w} \sum_{k=0}^{w-1} x_i(t-k)
$$

**Purpose**: Smooth out short-term fluctuations and identify trends.

**Example**:
$$
\text{temperature\_rolling\_mean\_5} = \frac{1}{5} \sum_{k=0}^{4} T_{t-k}
$$

#### Rolling Standard Deviation
$$
\text{rolling\_std}_i(t) = \sqrt{\frac{1}{w} \sum_{k=0}^{w-1} (x_i(t-k) - \text{rolling\_mean}_i(t))^2}
$$

**Purpose**: Measure variability over the window, detecting instability.

**Example**:
$$
\text{vibration\_rolling\_std\_5} = \sqrt{\frac{1}{5} \sum_{k=0}^{4} (V_{t-k} - \bar{V}_t)^2}
$$

#### Exponential Weighted Moving Average (EWM)
$$
\text{EWM}_i(t) = \alpha \cdot x_i(t) + (1-\alpha) \cdot \text{EWM}_i(t-1)
$$

Where $\alpha = \frac{2}{w+1} = \frac{2}{6} = 0.333$ for span $w = 5$.

**Purpose**: Give more weight to recent observations while considering historical context.

**Recursive form**:
$$
\text{EWM}_i(t) = \begin{cases}
x_i(0) & \text{if } t = 0 \\
0.333 \cdot x_i(t) + 0.667 \cdot \text{EWM}_i(t-1) & \text{if } t > 0
\end{cases}
$$

---

### 4. Rate of Change Features (5 features)

First-order difference:

$$
\Delta x_i(t) = x_i(t) - x_i(t-1)
$$

**Purpose**: Detect sudden changes or spikes in sensor readings.

**Example**:
$$
\text{temperature\_diff} = T_t - T_{t-1}
$$

**Interpretation**:
- $\Delta x_i > 0$: Increasing trend
- $\Delta x_i < 0$: Decreasing trend
- $|\Delta x_i|$ large: Rapid change (potential anomaly)

---

### 5. Deviation Features (6 features)

#### Absolute Deviation from Rolling Mean
$$
\text{dev\_rolling}_{i}(t) = |x_i(t) - \text{rolling\_mean}_i(t)|
$$

**Purpose**: Measure how far current value deviates from recent average.

#### Deviation from Machine Mean
$$
\text{dev\_machine}_{i,m}(t) = |x_i(t) - \mu_{i,m}|
$$

**Purpose**: Compare current reading to machine's historical baseline.

#### Anomaly Score
$$
\text{anomaly\_score}_i(t) = \frac{|x_i(t) - \text{rolling\_mean}_i(t)|}{\text{rolling\_std}_i(t) + \epsilon}
$$

Where $\epsilon = 0.001$ prevents division by zero.

**Purpose**: Normalized deviation score. Values $> 3$ suggest anomalies (3-sigma rule).

**Example**:
$$
\text{temperature\_anomaly\_score} = \frac{|T_t - \bar{T}_{t,5}|}{\sigma_{T,5} + 0.001}
$$

---

### 6. Aggregate Statistics (5 features)

For each machine $m$ over all time:

#### Mean
$$
\mu_{i,m} = \frac{1}{N_m} \sum_{t=1}^{N_m} x_i(t)
$$

#### Standard Deviation
$$
\sigma_{i,m} = \sqrt{\frac{1}{N_m} \sum_{t=1}^{N_m} (x_i(t) - \mu_{i,m})^2}
$$

#### Min/Max
$$
\min_{i,m} = \min_{t \in [1, N_m]} x_i(t), \quad \max_{i,m} = \max_{t \in [1, N_m]} x_i(t)
$$

#### Range
$$
\text{range}_{i,m} = \max_{i,m} - \min_{i,m}
$$

**Purpose**: Capture machine-level operating characteristics.

---

### 7. Interaction Features (3 features)

#### Temperature × Vibration
$$
f_{T \times V} = x_1 \cdot x_2 = T \cdot V
$$

**Purpose**: Capture correlated failure modes (e.g., overheating causes increased vibration).

#### Humidity × Pressure
$$
f_{H \times P} = x_3 \cdot x_4 = H \cdot P
$$

**Purpose**: Environmental interaction effects.

#### Energy × Temperature
$$
f_{E \times T} = x_5 \cdot x_1 = E \cdot T
$$

**Purpose**: Energy efficiency indicator (high energy + high temp = inefficiency).

---

### 8. Temporal Encodings (8 features)

#### Hour Encoding (Cyclical)
$$
\text{hour\_sin} = \sin\left(\frac{2\pi \cdot h}{24}\right), \quad \text{hour\_cos} = \cos\left(\frac{2\pi \cdot h}{24}\right)
$$

Where $h \in [0, 23]$ is the hour of day.

**Purpose**: Encode time of day cyclically (hour 23 is close to hour 0).

#### Day of Week Encoding (Cyclical)
$$
\text{day\_sin} = \sin\left(\frac{2\pi \cdot d}{7}\right), \quad \text{day\_cos} = \cos\left(\frac{2\pi \cdot d}{7}\right)
$$

Where $d \in [0, 6]$ is the day of week.

**Purpose**: Capture weekly patterns (weekday vs weekend operations).

#### Day/Month/Year (One-Hot or Ordinal)
$$
\text{day} \in [1, 31], \quad \text{month} \in [1, 12], \quad \text{year} \in \mathbb{Z}^+
$$

---

### Total Feature Count

| Category | Count | Formula Types |
|----------|-------|---------------|
| Raw sensors | 5 | $x_i$ |
| Basic transforms | 15 | $x_i^2, \sqrt{x_i}, \log(x_i + 1)$ |
| Z-scores | 5 | $(x_i - \mu) / \sigma$ |
| Rolling windows | 15 | Mean, Std, EWM |
| Rate of change | 5 | $x_i(t) - x_i(t-1)$ |
| Deviations | 6 | $\|x_i - \bar{x}\|$, anomaly score |
| Aggregates | 5 | $\mu, \sigma, \min, \max, \text{range}$ |
| Interactions | 3 | $x_i \cdot x_j$ |
| Temporal | 8 | $\sin/\cos$ encodings |
| **TOTAL** | **67** | |

---

## Model Mathematics {#model-mathematics}

### 1. Random Forest

**Ensemble of Decision Trees**:

$$
\hat{y} = \frac{1}{B} \sum_{b=1}^{B} T_b(\mathbf{x})
$$

Where:
- $B = 200$ is the number of trees
- $T_b(\mathbf{x})$ is the prediction of tree $b$
- Each tree is trained on a bootstrap sample of the data

**Splitting Criterion (Gini Impurity)**:
$$
\text{Gini}(S) = 1 - \sum_{c=1}^{C} p_c^2
$$

Where $p_c$ is the proportion of class $c$ in set $S$.

**Feature Importance**:
$$
\text{Importance}(f) = \sum_{t \in \text{trees}} \sum_{n \in \text{nodes}} \Delta\text{Gini}_n \cdot \mathbb{1}[\text{feature}_n = f]
$$

**Hyperparameters Used**:
- `n_estimators=200`
- `max_depth=None` (grow until pure)
- `min_samples_split=2`
- `class_weight='balanced'` → $w_c = \frac{n}{C \cdot n_c}$

---

### 2. XGBoost (Extreme Gradient Boosting)

**Objective Function**:
$$
\mathcal{L}(\theta) = \sum_{i=1}^{n} L(y_i, \hat{y}_i) + \sum_{k=1}^{K} \Omega(f_k)
$$

Where:
- $L$ is the loss function (log loss for classification)
- $\Omega(f_k) = \gamma T + \frac{1}{2}\lambda \|\mathbf{w}\|^2$ is the regularization term
- $T$ is the number of leaves, $\mathbf{w}$ are leaf weights

**Additive Training**:
$$
\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + \eta \cdot f_t(\mathbf{x}_i)
$$

Where $\eta = 0.1$ is the learning rate.

**Log Loss (Binary Classification)**:
$$
L(y, \hat{y}) = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]
$$

**Hyperparameters Used**:
- `n_estimators=100`
- `learning_rate=0.1`
- `max_depth=6`
- `subsample=0.8` (80% row sampling)
- `colsample_bytree=0.8` (80% feature sampling)

---

### 3. LightGBM (Gradient Boosting Decision Tree)

**Gradient-Based One-Side Sampling (GOSS)**:

Keeps instances with large gradients and randomly samples instances with small gradients:

$$
\tilde{g} = \frac{1 - a}{b} \sum_{i \in A_l} g_i + \frac{1}{|B_l|} \sum_{i \in B_l} g_i
$$

Where:
- $A_l$ = top $a \times 100\%$ instances with largest gradients
- $B_l$ = random sample of remaining instances
- $g_i = \frac{\partial L}{\partial \hat{y}_i}$ is the gradient

**Leaf-wise Growth**:
$$
\text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma
$$

Where $G = \sum g_i$, $H = \sum h_i$ (sum of gradients and hessians).

**Hyperparameters Used**:
- `n_estimators=100`
- `learning_rate=0.1`
- `num_leaves=31`
- `class_weight='balanced'`

---

### 4. CatBoost (Categorical Boosting)

**Ordered Boosting**:

Prevents prediction shift by using different permutations:

$$
\hat{y}_i^{(t)} = \text{Model}^{(t-1)}(\mathbf{x}_{\sigma(1)}, \ldots, \mathbf{x}_{\sigma(i-1)}) + f_t(\mathbf{x}_{\sigma(i)})
$$

Where $\sigma$ is a random permutation of training data.

**Symmetric Trees**:

All nodes at the same depth use the same splitting criterion, reducing model complexity.

**Loss Function (Logloss)**:
$$
L = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(p_i) + (1-y_i) \log(1-p_i)]
$$

**Hyperparameters Used**:
- `iterations=100`
- `learning_rate=0.1`
- `depth=6`
- `auto_class_weights='Balanced'`

---

### 5. Gradient Boosting

**Functional Gradient Descent**:

$$
\hat{y}^{(t)} = \hat{y}^{(t-1)} - \eta \nabla_{\hat{y}} L(y, \hat{y}^{(t-1)})
$$

**Deviance (Logistic Loss)**:
$$
L(y, F) = -\sum_{i=1}^{n} [y_i F(\mathbf{x}_i) - \log(1 + e^{F(\mathbf{x}_i)})]
$$

**Hyperparameters Used**:
- `n_estimators=100`
- `learning_rate=0.1`
- `max_depth=5`

---

### 6. Extra Trees (Extremely Randomized Trees)

Similar to Random Forest but with fully randomized splits:

**Split Selection**:
$$
\text{split} = \text{random}(\min(f), \max(f))
$$

Instead of optimizing split points, Extra Trees randomly selects them.

**Prediction**:
$$
\hat{y} = \frac{1}{B} \sum_{b=1}^{B} T_b(\mathbf{x})
$$

**Hyperparameters Used**:
- `n_estimators=200`
- `max_depth=None`
- `class_weight='balanced'`

---

### 7. Isolation Forest

**Anomaly Score**:

$$
s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}
$$

Where:
- $h(x)$ = path length to isolate instance $x$
- $c(n) = 2H(n-1) - \frac{2(n-1)}{n}$ is the average path length
- $H(n)$ is the harmonic number

**Interpretation**:
- $s \approx 1$: Anomaly (short path length)
- $s \approx 0.5$: Normal (average path length)
- $s < 0.5$: Normal (long path length)

**Hyperparameters Used**:
- `n_estimators=100`
- `contamination=0.1` (expect 10% anomalies)

---

### 8. Logistic Regression

**Model**:
$$
P(y=1 | \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}
$$

**Loss Function (Cross-Entropy)**:
$$
L(\mathbf{w}, b) = -\sum_{i=1}^{n} [y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)] + \lambda \|\mathbf{w}\|^2
$$

**Gradient**:
$$
\frac{\partial L}{\partial \mathbf{w}} = \sum_{i=1}^{n} (\hat{y}_i - y_i) \mathbf{x}_i + 2\lambda \mathbf{w}
$$

**Hyperparameters Used**:
- `penalty='l2'`
- `C=1.0` (inverse regularization strength)
- `max_iter=1000`
- `class_weight='balanced'`

---

### 9. AdaBoost

**Weight Update**:

$$
w_i^{(t+1)} = w_i^{(t)} \cdot e^{\alpha_t \cdot \mathbb{1}[h_t(\mathbf{x}_i) \neq y_i]}
$$

Where:
$$
\alpha_t = \frac{1}{2} \log\left(\frac{1 - \epsilon_t}{\epsilon_t}\right)
$$

And $\epsilon_t = \sum_{i: h_t(\mathbf{x}_i) \neq y_i} w_i^{(t)}$ is the weighted error.

**Final Prediction**:
$$
H(\mathbf{x}) = \text{sign}\left(\sum_{t=1}^{T} \alpha_t h_t(\mathbf{x})\right)
$$

**Hyperparameters Used**:
- `n_estimators=100`
- `learning_rate=1.0`
- Base estimator: Decision Tree with `max_depth=1`

---

### 10. Gaussian Naive Bayes

**Bayes' Theorem**:
$$
P(y | \mathbf{x}) = \frac{P(\mathbf{x} | y) P(y)}{P(\mathbf{x})}
$$

**Gaussian Assumption**:
$$
P(x_i | y) = \frac{1}{\sqrt{2\pi\sigma_y^2}} \exp\left(-\frac{(x_i - \mu_y)^2}{2\sigma_y^2}\right)
$$

**Independence Assumption**:
$$
P(\mathbf{x} | y) = \prod_{i=1}^{d} P(x_i | y)
$$

**Prediction**:
$$
\hat{y} = \arg\max_y P(y) \prod_{i=1}^{d} P(x_i | y)
$$

---

### 11. Voting Classifier (Ensemble)

**Hard Voting**:
$$
\hat{y} = \text{mode}(h_1(\mathbf{x}), h_2(\mathbf{x}), \ldots, h_M(\mathbf{x}))
$$

**Soft Voting** (used in our system):
$$
\hat{y} = \arg\max_c \sum_{m=1}^{M} P_m(y=c | \mathbf{x})
$$

**Ensemble Components**:
- Random Forest
- XGBoost
- LightGBM

**Averaging**:
$$
P(y=1 | \mathbf{x}) = \frac{1}{3}[P_{RF}(y=1|\mathbf{x}) + P_{XGB}(y=1|\mathbf{x}) + P_{LGBM}(y=1|\mathbf{x})]
$$

---

### 12. Stacking Classifier (Ensemble)

**Two-Level Architecture**:

**Level 0 (Base Models)**:
$$
\hat{\mathbf{z}}_i = [h_1(\mathbf{x}_i), h_2(\mathbf{x}_i), \ldots, h_M(\mathbf{x}_i)]^T
$$

**Level 1 (Meta-Learner)**:
$$
\hat{y}_i = g(\hat{\mathbf{z}}_i)
$$

Where $g$ is Logistic Regression trained on base model predictions.

**Base Models**:
- Random Forest → $h_1$
- XGBoost → $h_2$
- LightGBM → $h_3$

**Meta-Learner**:
- Logistic Regression

**Training Process**:
1. Train base models on training data
2. Generate predictions $\hat{\mathbf{z}}$ using cross-validation
3. Train meta-learner on $\hat{\mathbf{z}}$ to predict $y$

---

## Regression Models (RUL Prediction)

### Random Forest Regressor

**Prediction**:
$$
\hat{y} = \frac{1}{B} \sum_{b=1}^{B} T_b(\mathbf{x})
$$

**Splitting Criterion (MSE)**:
$$
\text{MSE}(S) = \frac{1}{|S|} \sum_{i \in S} (y_i - \bar{y}_S)^2
$$

---

### XGBoost Regressor

**Loss Function (MSE)**:
$$
L(y, \hat{y}) = \frac{1}{2}(y - \hat{y})^2
$$

**Gradient**:
$$
g_i = \frac{\partial L}{\partial \hat{y}_i} = \hat{y}_i - y_i
$$

**Hessian**:
$$
h_i = \frac{\partial^2 L}{\partial \hat{y}_i^2} = 1
$$

---

### LightGBM Regressor

Similar to XGBoost but with leaf-wise growth and GOSS.

---

### CatBoost Regressor

**Loss Function (RMSE)**:
$$
L = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$

---

### Gradient Boosting Regressor

**Loss Function (Least Squares)**:
$$
L(y, F) = \frac{1}{2} \sum_{i=1}^{n} (y_i - F(\mathbf{x}_i))^2
$$

**Negative Gradient**:
$$
-\frac{\partial L}{\partial F(\mathbf{x}_i)} = y_i - F(\mathbf{x}_i)
$$

---

### Extra Trees Regressor

Similar to Extra Trees classifier but outputs continuous values.

---

## Evaluation Metrics {#evaluation-metrics}

### Classification Metrics

#### Confusion Matrix

$$
\begin{bmatrix}
\text{TN} & \text{FP} \\
\text{FN} & \text{TP}
\end{bmatrix}
$$

Where:
- TP = True Positives
- TN = True Negatives
- FP = False Positives
- FN = False Negatives

#### Accuracy
$$
\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}
$$

#### Precision
$$
\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
$$

**Interpretation**: Of all predicted anomalies, how many were actual anomalies?

#### Recall (Sensitivity)
$$
\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
$$

**Interpretation**: Of all actual anomalies, how many did we detect?

#### F1 Score (Harmonic Mean)
$$
F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2\text{TP}}{2\text{TP} + \text{FP} + \text{FN}}
$$

**Interpretation**: Balanced measure of precision and recall.

#### Class Weights (for Imbalanced Data)
$$
w_c = \frac{n_{\text{samples}}}{n_{\text{classes}} \cdot n_{\text{samples}_c}}
$$

For 9:1 imbalance (90% normal, 10% anomaly):
$$
w_{\text{normal}} = \frac{n}{2 \cdot 0.9n} = 0.556, \quad w_{\text{anomaly}} = \frac{n}{2 \cdot 0.1n} = 5.0
$$

---

### Regression Metrics

#### Mean Absolute Error (MAE)
$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

**Interpretation**: Average absolute prediction error in original units (hours).

#### Root Mean Squared Error (RMSE)
$$
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$

**Interpretation**: Penalizes large errors more than MAE.

#### R² Score (Coefficient of Determination)
$$
R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
$$

Where $\bar{y} = \frac{1}{n} \sum_{i=1}^{n} y_i$ is the mean of actual values.

**Interpretation**:
- $R^2 = 1$: Perfect predictions
- $R^2 = 0$: Model no better than predicting mean
- $R^2 < 0$: Model worse than baseline

---

## Cross-Validation {#cross-validation}

### K-Fold Cross-Validation

Split data into $K = 3$ folds:

$$
\mathcal{D} = \mathcal{D}_1 \cup \mathcal{D}_2 \cup \mathcal{D}_3
$$

For each fold $k$:
- Train on $\mathcal{D} \setminus \mathcal{D}_k$
- Test on $\mathcal{D}_k$

**Mean Performance**:
$$
\bar{F}_1 = \frac{1}{K} \sum_{k=1}^{K} F_{1,k}
$$

**Standard Deviation**:
$$
\sigma_{F_1} = \sqrt{\frac{1}{K} \sum_{k=1}^{K} (F_{1,k} - \bar{F}_1)^2}
$$

### Stratified K-Fold

Maintains class distribution in each fold:

$$
\frac{n_{c,k}}{n_k} \approx \frac{n_c}{n}, \quad \forall k, c
$$

Where:
- $n_{c,k}$ = number of class $c$ samples in fold $k$
- $n_k$ = total samples in fold $k$
- $n_c$ = total class $c$ samples
- $n$ = total samples

---

## Feature Scaling

### StandardScaler

$$
x_i' = \frac{x_i - \mu}{\sigma}
$$

Where:
$$
\mu = \frac{1}{n} \sum_{j=1}^{n} x_j, \quad \sigma = \sqrt{\frac{1}{n} \sum_{j=1}^{n} (x_j - \mu)^2}
$$

**Purpose**: Center features to mean 0 and scale to unit variance.

---

## Summary of Model Usage with Features

| Model | All 67 Features Used | Mathematical Basis |
|-------|---------------------|-------------------|
| Random Forest | ✅ Yes | Bootstrap aggregation + Gini impurity |
| XGBoost | ✅ Yes | Gradient boosting + L1/L2 regularization |
| LightGBM | ✅ Yes | GOSS + Leaf-wise growth |
| CatBoost | ✅ Yes | Ordered boosting + Symmetric trees |
| Gradient Boosting | ✅ Yes | Functional gradient descent |
| Extra Trees | ✅ Yes | Fully randomized splits |
| Isolation Forest | ✅ Yes | Path length isolation |
| Logistic Regression | ✅ Yes | Sigmoid + L2 regularization |
| AdaBoost | ✅ Yes | Exponential loss + Weighted errors |
| Gaussian NB | ✅ Yes | Bayes' theorem + Gaussian likelihood |
| Voting Classifier | ✅ Yes | Soft voting ensemble |
| Stacking Classifier | ✅ Yes | Two-level meta-learning |

**Key Point**: All models use the complete 67-dimensional feature vector $\mathbf{x} \in \mathbb{R}^{67}$ after StandardScaler transformation.

---

## Training Process

For each model $M$ and each target variable $y_t$:

1. **Load Data**: $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{n}$ where $\mathbf{x}_i \in \mathbb{R}^{67}$

2. **Feature Scaling**:
   $$
   \mathbf{x}_i' = \frac{\mathbf{x}_i - \boldsymbol{\mu}}{\boldsymbol{\sigma}}
   $$

3. **K-Fold Split**: $\mathcal{D} = \bigcup_{k=1}^{3} \mathcal{D}_k$

4. **Train on Each Fold**:
   $$
   \theta_k^* = \arg\min_\theta \mathcal{L}(\theta; \mathcal{D} \setminus \mathcal{D}_k)
   $$

5. **Evaluate**:
   $$
   F_{1,k} = \text{f1\_score}(y_{\mathcal{D}_k}, M(\mathbf{X}_{\mathcal{D}_k}; \theta_k^*))
   $$

6. **Aggregate**:
   $$
   \bar{F}_1 = \frac{1}{3} \sum_{k=1}^{3} F_{1,k}
   $$

7. **Select Best Model**:
   $$
   M^* = \arg\max_M \bar{F}_{1,M}
   $$

---

**End of Mathematical Foundation**

This document provides the complete mathematical framework for all feature engineering transformations and machine learning models used in the IoT Anomaly Detection system.
