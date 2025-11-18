# Model Training Script Documentation

## File: `train_models_kfold.py`

### Purpose
Train 12 machine learning models using k-fold cross-validation for 5 prediction tasks in IoT anomaly detection.

---

## Overview

### Prediction Tasks
1. **Anomaly Detection** (Binary Classification)
2. **Maintenance Prediction** (Binary Classification)
3. **Downtime Risk** (Binary Classification)
4. **Failure Type Classification** (Multi-class Classification - 5 classes)
5. **Remaining Useful Life** (Regression)

### Models Trained
- **Classification** (11 models): Random Forest, XGBoost, LightGBM, CatBoost, Gradient Boosting, Extra Trees, Isolation Forest, Logistic Regression, AdaBoost, Gaussian NB, Voting Classifier, Stacking Classifier
- **Regression** (6 models): Random Forest, XGBoost, LightGBM, CatBoost, Gradient Boosting, Extra Trees

**Total**: 49 model configurations (11×4 + 6×1)

---

## Mathematical Framework

### K-Fold Cross-Validation

Split dataset into $K$ folds:
$$
\mathcal{D} = \bigcup_{k=1}^{K} \mathcal{D}_k, \quad \mathcal{D}_i \cap \mathcal{D}_j = \emptyset \text{ for } i \neq j
$$

For each fold $k$:
1. **Train set**: $\mathcal{D}_{\text{train}}^{(k)} = \mathcal{D} \setminus \mathcal{D}_k$
2. **Validation set**: $\mathcal{D}_{\text{val}}^{(k)} = \mathcal{D}_k$
3. **Train model**: $\theta_k^* = \arg\min_\theta \mathcal{L}(\theta; \mathcal{D}_{\text{train}}^{(k)})$
4. **Evaluate**: $\text{score}_k = \text{metric}(y_{\mathcal{D}_k}, f(\mathbf{X}_{\mathcal{D}_k}; \theta_k^*))$

**Aggregate metrics**:
$$
\begin{align}
\bar{\text{score}} &= \frac{1}{K} \sum_{k=1}^{K} \text{score}_k \\
\sigma_{\text{score}} &= \sqrt{\frac{1}{K} \sum_{k=1}^{K} (\text{score}_k - \bar{\text{score}})^2}
\end{align}
$$

---

## Function Documentation

### `get_models_classification()`

**Returns**: Dictionary of classification models

**Model Configurations**:

#### 1. Random Forest
```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
```

**Mathematics**:
$$
\hat{y} = \text{mode}\left(\{T_1(\mathbf{x}), T_2(\mathbf{x}), \ldots, T_{200}(\mathbf{x})\}\right)
$$

Where each tree $T_b$ is trained on bootstrap sample $\mathcal{D}_b^*$.

**Class Weights**:
$$
w_c = \frac{n_{\text{samples}}}{n_{\text{classes}} \cdot n_{c}}
$$

For imbalanced data (e.g., 90% normal, 10% anomaly):
$$
w_{\text{normal}} = \frac{n}{2 \cdot 0.9n} = 0.556, \quad w_{\text{anomaly}} = \frac{n}{2 \cdot 0.1n} = 5.0
$$

---

#### 2. XGBoost
```python
XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)
```

**Mathematics**:
$$
\hat{y}^{(t)} = \hat{y}^{(t-1)} + \eta \cdot f_t(\mathbf{x})
$$

**Objective Function**:
$$
\mathcal{L}^{(t)} = \sum_{i=1}^{n} L(y_i, \hat{y}_i^{(t-1)} + f_t(\mathbf{x}_i)) + \Omega(f_t)
$$

Where:
- $L$ = log loss: $-[y \log(p) + (1-y) \log(1-p)]$
- $\Omega(f_t) = \gamma T + \frac{1}{2}\lambda \|\mathbf{w}\|^2$ (regularization)
- $\eta = 0.1$ (learning rate)

**Second-Order Taylor Expansion**:
$$
\mathcal{L}^{(t)} \approx \sum_{i=1}^{n} [L(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(\mathbf{x}_i) + \frac{1}{2} h_i f_t^2(\mathbf{x}_i)] + \Omega(f_t)
$$

Where:
- $g_i = \frac{\partial L}{\partial \hat{y}^{(t-1)}}$ (gradient)
- $h_i = \frac{\partial^2 L}{\partial (\hat{y}^{(t-1)})^2}$ (hessian)

---

#### 3. LightGBM
```python
LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=31,
    class_weight='balanced',
    random_state=42
)
```

**Key Innovation**: Gradient-Based One-Side Sampling (GOSS)

**GOSS Algorithm**:
1. Sort instances by $|g_i|$ (absolute gradient)
2. Keep top $a \times 100\%$ with largest gradients → set $A$
3. Random sample $(1-a) \times b \times 100\%$ from rest → set $B$
4. Compute gain with reweighted gradients:

$$
\tilde{G} = \sum_{i \in A} g_i + \frac{1-a}{b} \sum_{i \in B} g_i
$$

**Leaf-Wise Growth** (vs. Level-Wise):
$$
\text{Gain} = \frac{1}{2} \left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\right] - \gamma
$$

Grows tree by maximum gain, leading to deeper, more accurate trees.

---

#### 4. CatBoost
```python
CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    auto_class_weights='Balanced',
    verbose=0,
    random_state=42
)
```

**Key Innovation**: Ordered Boosting

**Problem with Standard Boosting**: Prediction shift
$$
F^{(t)}(x_i) \text{ uses same data } x_i \text{ was trained on} \Rightarrow \text{overfitting}
$$

**Ordered Boosting Solution**:
Use $k$ random permutations $\sigma_1, \ldots, \sigma_k$ of training data:

$$
\hat{y}_{\sigma_r(i)}^{(t)} = \text{Model}^{(t-1)}(\mathbf{x}_{\sigma_r(1)}, \ldots, \mathbf{x}_{\sigma_r(i-1)})
$$

Model at step $t$ for instance $i$ only uses instances before $i$ in permutation $\sigma_r$.

**Symmetric Trees**: All nodes at depth $d$ use same feature and split, reducing overfitting.

---

#### 5. Gradient Boosting
```python
GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    random_state=42
)
```

**Mathematics**:
$$
F^{(t)}(\mathbf{x}) = F^{(t-1)}(\mathbf{x}) + \eta \cdot h_t(\mathbf{x})
$$

Where $h_t$ fits the negative gradient:
$$
h_t = \arg\min_h \sum_{i=1}^{n} \left(-\frac{\partial L(y_i, F^{(t-1)}(\mathbf{x}_i))}{\partial F^{(t-1)}(\mathbf{x}_i)} - h(\mathbf{x}_i)\right)^2
$$

**For Log Loss**:
$$
-\frac{\partial L}{\partial F} = y - p, \quad p = \frac{1}{1 + e^{-F}}
$$

---

#### 6. Extra Trees
```python
ExtraTreesClassifier(
    n_estimators=200,
    max_depth=None,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
```

**Key Difference from Random Forest**: Fully randomized splits

**Split Selection**:
Instead of finding optimal split:
$$
s^* = \arg\max_s \text{InformationGain}(S, s)
$$

Extra Trees randomly selects:
$$
s_{\text{random}} \sim \text{Uniform}(\min(f), \max(f))
$$

**Advantage**: Lower variance, faster training, less overfitting.

---

#### 7. Isolation Forest
```python
IsolationForest(
    n_estimators=100,
    contamination=0.1,
    random_state=42
)
```

**Anomaly Score**:
$$
s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}
$$

Where:
- $h(x)$ = path length to isolate $x$
- $c(n) = 2H(n-1) - \frac{2(n-1)}{n}$ (normalization)
- $H(n) = \ln(n) + 0.5772$ (harmonic number)

**Interpretation**:
$$
\begin{cases}
s \approx 1 & \Rightarrow \text{Anomaly (short path)} \\
s \approx 0.5 & \Rightarrow \text{Normal (average path)} \\
s < 0.5 & \Rightarrow \text{Normal (long path)}
\end{cases}
$$

**Threshold**: Contamination = 10% means top 10% of scores are anomalies.

---

#### 8. Logistic Regression
```python
LogisticRegression(
    penalty='l2',
    C=1.0,
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)
```

**Model**:
$$
P(y=1 | \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}
$$

**Loss Function (with L2 regularization)**:
$$
\mathcal{L}(\mathbf{w}, b) = -\sum_{i=1}^{n} [y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)] + \frac{1}{2C} \|\mathbf{w}\|^2
$$

**Gradient**:
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = \sum_{i=1}^{n} (\hat{y}_i - y_i) \mathbf{x}_i + \frac{1}{C} \mathbf{w}
$$

**Optimization**: L-BFGS (Limited-memory BFGS) for efficient high-dimensional optimization.

---

#### 9. AdaBoost
```python
AdaBoostClassifier(
    n_estimators=100,
    learning_rate=1.0,
    random_state=42
)
```

**Weight Update**:
$$
w_i^{(t+1)} = w_i^{(t)} \cdot e^{\alpha_t \mathbb{1}[h_t(\mathbf{x}_i) \neq y_i]}
$$

Where:
$$
\alpha_t = \frac{1}{2} \ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right)
$$

And weighted error:
$$
\epsilon_t = \frac{\sum_{i: h_t(\mathbf{x}_i) \neq y_i} w_i^{(t)}}{\sum_{i=1}^{n} w_i^{(t)}}
$$

**Final Prediction**:
$$
H(\mathbf{x}) = \text{sign}\left(\sum_{t=1}^{T} \alpha_t h_t(\mathbf{x})\right)
$$

**Exponential Loss**:
$$
L(y, f(\mathbf{x})) = e^{-y f(\mathbf{x})}
$$

---

#### 10. Gaussian Naive Bayes
```python
GaussianNB()
```

**Bayes' Theorem**:
$$
P(y | \mathbf{x}) = \frac{P(\mathbf{x} | y) P(y)}{P(\mathbf{x})} \propto P(\mathbf{x} | y) P(y)
$$

**Gaussian Likelihood**:
$$
P(x_i | y=c) = \frac{1}{\sqrt{2\pi\sigma_{c,i}^2}} \exp\left(-\frac{(x_i - \mu_{c,i})^2}{2\sigma_{c,i}^2}\right)
$$

**Independence Assumption**:
$$
P(\mathbf{x} | y) = \prod_{i=1}^{d} P(x_i | y)
$$

**Prediction**:
$$
\hat{y} = \arg\max_c P(y=c) \prod_{i=1}^{d} P(x_i | y=c)
$$

**Log-Space Computation** (numerical stability):
$$
\hat{y} = \arg\max_c \left[\log P(y=c) + \sum_{i=1}^{d} \log P(x_i | y=c)\right]
$$

---

#### 11. Voting Classifier (Ensemble)
```python
VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(...)),
        ('xgb', XGBClassifier(...)),
        ('lgbm', LGBMClassifier(...))
    ],
    voting='soft'
)
```

**Soft Voting** (Probability Averaging):
$$
P(y=c | \mathbf{x}) = \frac{1}{M} \sum_{m=1}^{M} P_m(y=c | \mathbf{x})
$$

**Prediction**:
$$
\hat{y} = \arg\max_c P(y=c | \mathbf{x})
$$

**Advantage**: Reduces variance by averaging diverse models.

---

#### 12. Stacking Classifier (Ensemble)
```python
StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(...)),
        ('xgb', XGBClassifier(...)),
        ('lgbm', LGBMClassifier(...))
    ],
    final_estimator=LogisticRegression(...)
)
```

**Two-Level Architecture**:

**Level 0** (Base Models):
$$
\hat{\mathbf{z}}_i = [h_1(\mathbf{x}_i), h_2(\mathbf{x}_i), h_3(\mathbf{x}_i)]^T
$$

**Level 1** (Meta-Learner):
$$
\hat{y}_i = g(\hat{\mathbf{z}}_i)
$$

**Training Process**:
1. Split data: $\mathcal{D} = \mathcal{D}_{\text{train}} \cup \mathcal{D}_{\text{holdout}}$
2. Train base models on $\mathcal{D}_{\text{train}}$
3. Generate predictions on $\mathcal{D}_{\text{holdout}}$: $\hat{\mathbf{z}}$
4. Train meta-learner on $(\hat{\mathbf{z}}, y)$

**Advantage**: Learns optimal combination of base models.

---

### `get_models_regression()`

**Returns**: Dictionary of regression models

Similar to classification but with:
- **Loss**: MSE instead of log loss
- **Output**: Continuous value instead of class probability
- **Metrics**: MAE, RMSE, R² instead of F1, precision, recall

---

### `evaluate_classification(model, X, y, cv)`

**Purpose**: Evaluate classification model using k-fold cross-validation

**Process**:
```
For each fold k in K:
    1. Split data: train = D \ D_k, val = D_k
    2. Train: model.fit(X_train, y_train)
    3. Predict: y_pred = model.predict(X_val)
    4. Compute metrics: accuracy, precision, recall, F1
    5. Store fold results
Return: mean and std of metrics across folds
```

**Metrics**:

#### F1 Score (Primary Metric)
$$
F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

#### Precision
$$
\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
$$

#### Recall
$$
\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
$$

#### Accuracy
$$
\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}
$$

**Implementation**:
```python
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

scores = {
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1': [],
    'train_time': []
}

for train_idx, val_idx in cv.split(X, y):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_val)

    scores['accuracy'].append(accuracy_score(y_val, y_pred))
    scores['precision'].append(precision_score(y_val, y_pred, average='weighted', zero_division=0))
    scores['recall'].append(recall_score(y_val, y_pred, average='weighted', zero_division=0))
    scores['f1'].append(f1_score(y_val, y_pred, average='weighted', zero_division=0))
    scores['train_time'].append(train_time)

return {
    'accuracy_mean': np.mean(scores['accuracy']),
    'precision_mean': np.mean(scores['precision']),
    'recall_mean': np.mean(scores['recall']),
    'f1_mean': np.mean(scores['f1']),
    'train_time_mean': np.mean(scores['train_time'])
}
```

---

### `evaluate_regression(model, X, y, cv)`

**Purpose**: Evaluate regression model using k-fold cross-validation

**Metrics**:

#### Mean Absolute Error
$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

#### Root Mean Squared Error
$$
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$

#### R² Score
$$
R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
$$

Where $\bar{y} = \frac{1}{n} \sum_{i=1}^{n} y_i$.

---

### `train_and_save_best_model(X, y, models, task_name, output_dir)`

**Purpose**: Train all models, compare performance, save best model

**Algorithm**:
```
1. Initialize results list
2. For each model in models:
    a. Evaluate using k-fold CV
    b. Store metrics (F1/R², precision, recall, time)
    c. Print progress
3. Convert results to DataFrame
4. Sort by primary metric (F1 for classification, R² for regression)
5. Save results CSV
6. Train best model on full dataset
7. Save best model as pickle file
8. Return results DataFrame
```

**Model Selection**:
$$
M^* = \arg\max_{M \in \mathcal{M}} \bar{F}_{1,M}
$$

**Full Dataset Training**:
After selecting best model, retrain on entire dataset:
```python
best_model = models[best_model_name]
best_model.fit(X, y)
joblib.dump(best_model, f"{output_dir}/best_{task_name}_model.pkl")
```

---

## Main Training Pipeline

### `main()`

**Command Line Arguments**:
```bash
python train_models_kfold.py \
    --input data/smart_manufacturing_data.csv \
    --output_dir results \
    --sample 5000 \
    --n_folds 3
```

**Process Flow**:
```
1. Load dataset
2. Sample data (if --sample specified)
3. Compute features (call feature_engineering.compute_features)
4. Standardize features (StandardScaler)
5. Save scaler

For each task:
    6. Extract target variable
    7. Encode labels (if multi-class)
    8. Get appropriate models (classification or regression)
    9. Train and evaluate all models with k-fold CV
    10. Save results CSV
    11. Save best model
    12. Print summary

13. Create final summary JSON
```

---

## Feature Scaling

### StandardScaler

**Transform**:
$$
x_i' = \frac{x_i - \mu_i}{\sigma_i}
$$

Where:
$$
\mu_i = \frac{1}{n} \sum_{j=1}^{n} x_{ij}, \quad \sigma_i = \sqrt{\frac{1}{n} \sum_{j=1}^{n} (x_{ij} - \mu_i)^2}
$$

**Purpose**:
- Center features to mean 0
- Scale to unit variance
- Ensures features on same scale for distance-based algorithms
- Improves convergence for gradient-based algorithms

**Implementation**:
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler for inference
joblib.dump(scaler, "models/scaler.pkl")
```

---

## Results Output

### CSV Format
```
model,accuracy_mean,precision_mean,recall_mean,f1_mean,train_time_mean
1_RandomForest,0.9945,0.9945,0.9945,0.9945,0.312
2_XGBoost,0.9923,0.9923,0.9923,0.9923,0.187
...
```

### JSON Summary
```json
{
  "anomaly_detection": {
    "best_model": "1_RandomForest",
    "f1_score": 0.9998,
    "precision": 0.9998,
    "recall": 0.9998
  },
  "maintenance_prediction": {
    "best_model": "1_RandomForest",
    "f1_score": 0.9821,
    ...
  },
  ...
}
```

---

## Performance Analysis

### Time Complexity
For $n$ samples, $d$ features, $K$ folds:
- **Random Forest**: $O(K \cdot n \cdot d \cdot \log n \cdot B)$ where $B$ = 200 trees
- **XGBoost**: $O(K \cdot n \cdot d \cdot T)$ where $T$ = 100 boosting rounds
- **Logistic Regression**: $O(K \cdot n \cdot d \cdot I)$ where $I$ = iterations

### Space Complexity
- **Models**: $O(B \cdot n)$ for ensemble methods (store trees/nodes)
- **Data**: $O(n \cdot d)$ for features
- **Total**: $O(n \cdot (d + B))$

### Actual Timing (5000 samples, 67 features, 3 folds)
| Model | Time per Fold | Total Time |
|-------|---------------|------------|
| Random Forest | 0.10s | 0.31s |
| XGBoost | 0.06s | 0.19s |
| LightGBM | 0.05s | 0.16s |
| Logistic Regression | 0.02s | 0.07s |
| Gaussian NB | 0.00s | 0.01s |

---

## Error Handling

### Class Imbalance
```python
class_weight='balanced'  # Automatically adjust weights
```

### Multi-class Encoding
```python
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
joblib.dump(encoder, "models/failure_type_label_encoder.pkl")
```

### Missing Best Model
```python
if len(results) == 0:
    print(f"No models successfully trained for {task_name}")
    return None
```

### Division by Zero in Metrics
```python
precision_score(..., zero_division=0)  # Return 0 if no positive predictions
```

---

## Usage Examples

### Basic Training
```bash
python scripts/train_models_kfold.py \
    --input data/smart_manufacturing_data.csv \
    --output_dir results \
    --sample 5000 \
    --n_folds 3
```

### Full Dataset Training
```bash
python scripts/train_models_kfold.py \
    --input data/smart_manufacturing_data.csv \
    --output_dir results \
    --n_folds 5
```

### Fast Testing
```bash
python scripts/train_models_kfold.py \
    --input data/smart_manufacturing_data.csv \
    --output_dir results_test \
    --sample 1000 \
    --n_folds 3
```

---

## Best Practices

1. **Always use stratified k-fold** for classification to maintain class distribution
2. **Save scaler** for inference to ensure same preprocessing
3. **Use balanced class weights** for imbalanced datasets
4. **Monitor training time** to identify inefficient models
5. **Cross-validate** to avoid overfitting and get robust performance estimates
6. **Save best models** for deployment
7. **Document hyperparameters** for reproducibility

---

**End of Training Script Documentation**
