# IoT Anomaly Utils API Documentation

## File: `iot_anomaly_utils.py`

### Purpose
High-level Python API for IoT anomaly detection, providing a clean interface for data loading, feature engineering, model training, prediction, and evaluation.

---

## Class: `IoTAnomalyDetector`

### Overview
Main class encapsulating the entire ML pipeline.

**Attributes**:
- `df_raw`: Raw sensor data (DataFrame)
- `df_features`: Engineered features (DataFrame)
- `X`: Feature matrix (numpy array)
- `y`: Target vector (numpy array)
- `model`: Trained ML model
- `scaler`: StandardScaler for feature normalization
- `feature_cols`: List of feature column names

---

## Method Documentation

### `__init__(self)`

**Purpose**: Initialize empty detector

**Mathematics**: Creates empty placeholder for data pipeline:
$$
\mathcal{D} = \emptyset, \quad \mathbf{X} = \emptyset, \quad \mathbf{y} = \emptyset, \quad M = \text{None}
$$

```python
detector = IoTAnomalyDetector()
```

---

### `load_data(self, file_path: str, sample_size: int = None)`

**Purpose**: Load IoT sensor data from CSV

**Parameters**:
- `file_path`: Path to CSV file
- `sample_size`: Optional random sample size

**Mathematics**:

If `sample_size` is specified:
$$
\mathcal{D}_{\text{sample}} \sim \text{Uniform}(\mathcal{D}_{\text{full}}, n=\text{sample\_size})
$$

**Implementation**:
```python
df = pd.read_csv(file_path)
if sample_size:
    df = df.sample(n=sample_size, random_state=42)
self.df_raw = df
```

**Example**:
```python
detector.load_data("data/smart_manufacturing_data.csv", sample_size=1000)
```

---

### `prepare_features(self, target_col: str)`

**Purpose**: Generate 67 engineered features from raw sensors

**Parameters**:
- `target_col`: Name of target variable to exclude from features

**Mathematics**:

Applies feature engineering transformations:
$$
\mathbf{x}_{\text{raw}} \in \mathbb{R}^5 \xrightarrow{\phi} \mathbf{f} \in \mathbb{R}^{67}
$$

Where $\phi$ includes:
1. Basic transforms: $x^2, \sqrt{x}, \log(x+1)$
2. Z-scores: $(x - \mu)/\sigma$
3. Rolling windows: mean, std, EWM
4. Differences: $\Delta x = x_t - x_{t-1}$
5. Deviations and anomaly scores
6. Interactions: $x_i \cdot x_j$
7. Temporal encodings: $\sin/\cos$

**Implementation**:
```python
from scripts.feature_engineering import compute_features

self.df_features = compute_features(self.df_raw)

# Extract feature columns (exclude target and metadata)
exclude_cols = ['machine_id', 'timestamp', target_col,
                'maintenance_required', 'downtime_risk',
                'predicted_remaining_life', 'failure_type',
                'anomaly_flag']

self.feature_cols = [c for c in self.df_features.columns
                     if c not in exclude_cols]
```

**Example**:
```python
detector.prepare_features("anomaly_flag")
print(f"Generated {len(detector.feature_cols)} features")
```

---

### `prepare_train_test_split(self, target_col: str, test_size: float = 0.2, random_state: int = 42)`

**Purpose**: Split data into training and testing sets

**Parameters**:
- `target_col`: Target variable name
- `test_size`: Fraction for test set (default 0.2 = 20%)
- `random_state`: Random seed for reproducibility

**Mathematics**:

Stratified split maintaining class distribution:
$$
\mathcal{D} = \mathcal{D}_{\text{train}} \cup \mathcal{D}_{\text{test}}
$$

Where:
$$
\frac{|\mathcal{D}_{\text{train}}|}{|\mathcal{D}|} = 1 - p, \quad \frac{|\mathcal{D}_{\text{test}}|}{|\mathcal{D}|} = p
$$

And class proportions are preserved:
$$
\frac{n_{c,\text{train}}}{|\mathcal{D}_{\text{train}}|} \approx \frac{n_{c,\text{test}}}{|\mathcal{D}_{\text{test}}|} \approx \frac{n_c}{|\mathcal{D}|}
$$

**Implementation**:
```python
from sklearn.model_selection import train_test_split

X = self.df_features[self.feature_cols].values
y = self.df_features[target_col].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=test_size,
    random_state=random_state,
    stratify=y if self._is_classification(y) else None
)

# Scale features
self.scaler = StandardScaler()
X_train = self.scaler.fit_transform(X_train)
X_test = self.scaler.transform(X_test)

return X_train, X_test, y_train, y_test
```

**StandardScaler Transform**:
$$
x_i' = \frac{x_i - \mu_i}{\sigma_i}
$$

Where $\mu_i, \sigma_i$ are computed on **training set only** to avoid data leakage.

**Example**:
```python
X_train, X_test, y_train, y_test = detector.prepare_train_test_split(
    target_col="anomaly_flag",
    test_size=0.3
)
```

---

### `train_model(self, X_train, y_train, model_type='random_forest', **kwargs)`

**Purpose**: Train a specific model type

**Parameters**:
- `X_train`: Training features
- `y_train`: Training labels
- `model_type`: Model name (default: 'random_forest')
- `**kwargs`: Additional model hyperparameters

**Available Models**:
1. `'random_forest'`: Random Forest Classifier/Regressor
2. `'xgboost'`: XGBoost Classifier/Regressor
3. `'lightgbm'`: LightGBM Classifier/Regressor
4. `'catboost'`: CatBoost Classifier/Regressor
5. `'logistic'`: Logistic Regression
6. `'gradient_boosting'`: Gradient Boosting Classifier/Regressor

**Mathematics**:

Finds optimal parameters:
$$
\theta^* = \arg\min_\theta \mathcal{L}(\theta; \mathcal{D}_{\text{train}})
$$

**Implementation**:
```python
from sklearn.ensemble import RandomForestClassifier

if model_type == 'random_forest':
    model = RandomForestClassifier(
        n_estimators=kwargs.get('n_estimators', 200),
        max_depth=kwargs.get('max_depth', None),
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

model.fit(X_train, y_train)
self.model = model
```

**Example**:
```python
detector.train_model(
    X_train, y_train,
    model_type='random_forest',
    n_estimators=300,
    max_depth=15
)
```

---

### `predict(self, X)`

**Purpose**: Make predictions on new data

**Parameters**:
- `X`: Feature matrix (numpy array or DataFrame)

**Returns**: Predictions (numpy array)

**Mathematics**:

For classification:
$$
\hat{y} = M(\mathbf{x}) \in \{0, 1, \ldots, C-1\}
$$

For regression:
$$
\hat{y} = M(\mathbf{x}) \in \mathbb{R}
$$

**Implementation**:
```python
def predict(self, X):
    if self.model is None:
        raise ValueError("No model loaded. Train or load a model first.")

    # Ensure X is scaled if scaler exists
    if self.scaler is not None and not self._is_scaled(X):
        X = self.scaler.transform(X)

    return self.model.predict(X)
```

**Example**:
```python
predictions = detector.predict(X_test)
print(f"Predicted anomalies: {predictions.sum()}")
```

---

### `predict_proba(self, X)`

**Purpose**: Get prediction probabilities (classification only)

**Parameters**:
- `X`: Feature matrix

**Returns**: Probability matrix of shape (n_samples, n_classes)

**Mathematics**:
$$
P(y = c | \mathbf{x}) = M_{\text{proba}}(\mathbf{x})_c
$$

Where $\sum_{c=1}^{C} P(y=c|\mathbf{x}) = 1$.

**For Binary Classification**:
$$
P = \begin{bmatrix}
P(y=0|\mathbf{x}_1) & P(y=1|\mathbf{x}_1) \\
P(y=0|\mathbf{x}_2) & P(y=1|\mathbf{x}_2) \\
\vdots & \vdots \\
P(y=0|\mathbf{x}_n) & P(y=1|\mathbf{x}_n)
\end{bmatrix}
$$

**Example**:
```python
probs = detector.predict_proba(X_test)
anomaly_probs = probs[:, 1]  # Probability of class 1 (anomaly)
high_risk = X_test[anomaly_probs > 0.9]  # 90%+ confidence
```

---

### `evaluate_classification(self, y_true, y_pred)`

**Purpose**: Compute classification metrics

**Parameters**:
- `y_true`: True labels
- `y_pred`: Predicted labels

**Returns**: Dictionary with metrics

**Metrics Computed**:

#### 1. Accuracy
$$
\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}
$$

#### 2. Precision
$$
\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
$$

#### 3. Recall
$$
\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
$$

#### 4. F1 Score
$$
F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

**Implementation**:
```python
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score
)

def evaluate_classification(self, y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
```

**Example**:
```python
metrics = detector.evaluate_classification(y_test, predictions)
print(f"F1 Score: {metrics['f1']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
```

---

### `evaluate_regression(self, y_true, y_pred)`

**Purpose**: Compute regression metrics

**Parameters**:
- `y_true`: True values
- `y_pred`: Predicted values

**Returns**: Dictionary with metrics

**Metrics Computed**:

#### 1. MAE (Mean Absolute Error)
$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

#### 2. RMSE (Root Mean Squared Error)
$$
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$

#### 3. R² Score
$$
R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
$$

**Implementation**:
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_regression(self, y_true, y_pred):
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred)
    }
```

**Example**:
```python
metrics = detector.evaluate_regression(y_test, predictions)
print(f"R² Score: {metrics['r2']:.4f}")
print(f"MAE: {metrics['mae']:.2f} hours")
print(f"RMSE: {metrics['rmse']:.2f} hours")
```

---

### `save_model(self, file_path: str)`

**Purpose**: Save trained model to disk

**Parameters**:
- `file_path`: Path for saved model (should end in .pkl)

**Format**: Pickle (joblib)

**Saved Objects**:
- Model parameters $\theta^*$
- Model architecture
- Feature importances (if applicable)

**Implementation**:
```python
import joblib

def save_model(self, file_path):
    if self.model is None:
        raise ValueError("No model to save")

    joblib.dump(self.model, file_path)
    print(f"Model saved to {file_path}")
```

**Example**:
```python
detector.save_model("models/my_anomaly_model.pkl")
```

---

### `load_model(self, file_path: str)`

**Purpose**: Load pre-trained model from disk

**Parameters**:
- `file_path`: Path to saved model file

**Mathematics**:

Restores model state:
$$
M \leftarrow \text{Load}(\theta^*)
$$

**Implementation**:
```python
import joblib

def load_model(self, file_path):
    self.model = joblib.load(file_path)
    print(f"Model loaded from {file_path}")
```

**Example**:
```python
detector.load_model("models/best_anomaly_model.pkl")
predictions = detector.predict(X_new)
```

---

### `plot_confusion_matrix(self, y_true, y_pred, labels=None)`

**Purpose**: Visualize confusion matrix

**Parameters**:
- `y_true`: True labels
- `y_pred`: Predicted labels
- `labels`: Class labels (optional)

**Returns**: Matplotlib figure

**Confusion Matrix Definition**:
$$
C_{ij} = \sum_{k=1}^{n} \mathbb{1}[y_k = i \land \hat{y}_k = j]
$$

**For Binary Classification**:
$$
\begin{bmatrix}
\text{TN} & \text{FP} \\
\text{FN} & \text{TP}
\end{bmatrix}
$$

**Normalized Version**:
$$
C'_{ij} = \frac{C_{ij}}{\sum_{j} C_{ij}}
$$

**Implementation**:
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(self, y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels or ['Normal', 'Anomaly'],
                yticklabels=labels or ['Normal', 'Anomaly'])

    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')

    return fig
```

**Example**:
```python
fig = detector.plot_confusion_matrix(y_test, predictions)
plt.savefig("charts/confusion_matrix.png", dpi=300)
plt.show()
```

---

### `plot_feature_importance(self, top_n=20)`

**Purpose**: Visualize most important features

**Parameters**:
- `top_n`: Number of top features to display

**Returns**: Matplotlib figure

**Feature Importance (Random Forest)**:
$$
I(f) = \sum_{t \in \text{trees}} \sum_{n \in \text{nodes}} \Delta\text{Gini}_n \cdot \mathbb{1}[\text{feature}_n = f]
$$

Normalized:
$$
I'(f) = \frac{I(f)}{\sum_{f'} I(f')}
$$

**For XGBoost**: Uses gain-based importance

**Implementation**:
```python
def plot_feature_importance(self, top_n=20):
    if not hasattr(self.model, 'feature_importances_'):
        raise ValueError("Model does not support feature importances")

    importances = self.model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(top_n), importances[indices], color='skyblue')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([self.feature_cols[i] for i in indices])
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Feature Importances')

    return fig
```

**Example**:
```python
fig = detector.plot_feature_importance(top_n=15)
plt.tight_layout()
plt.show()
```

---

## Utility Functions

### `load_results(results_dir: str) -> Dict[str, pd.DataFrame]`

**Purpose**: Load all training results from directory

**Returns**: Dictionary mapping task names to result DataFrames

**Example**:
```python
from iot_anomaly_utils import load_results

results = load_results("results")
for task, df in results.items():
    print(f"{task}: {len(df)} models")
```

---

### `plot_model_comparison(results: pd.DataFrame, metric: str, title: str)`

**Purpose**: Create bar chart comparing models

**Parameters**:
- `results`: DataFrame with model results
- `metric`: Metric column name (e.g., 'f1_mean')
- `title`: Plot title

**Example**:
```python
from iot_anomaly_utils import plot_model_comparison

fig = plot_model_comparison(
    results['results_anomaly'],
    metric='f1_mean',
    title='Anomaly Detection - F1 Score'
)
```

---

### `quick_predict(data_path: str, model_path: str, target_col: str, sample_size: int = None)`

**Purpose**: End-to-end prediction in one function

**Process**:
```
1. Load data
2. Compute features
3. Load model and scaler
4. Make predictions
5. Return predictions
```

**Example**:
```python
from iot_anomaly_utils import quick_predict

predictions = quick_predict(
    data_path="data/new_data.csv",
    model_path="models/best_anomaly_model.pkl",
    target_col="anomaly_flag",
    sample_size=1000
)

print(f"Detected {predictions.sum()} anomalies")
```

---

## Complete Workflow Example

```python
from iot_anomaly_utils import IoTAnomalyDetector

# 1. Initialize
detector = IoTAnomalyDetector()

# 2. Load data
detector.load_data("data/smart_manufacturing_data.csv", sample_size=5000)

# 3. Prepare features (67 features generated)
detector.prepare_features("anomaly_flag")

# 4. Split data
X_train, X_test, y_train, y_test = detector.prepare_train_test_split(
    target_col="anomaly_flag",
    test_size=0.2
)

# 5. Train model
detector.train_model(X_train, y_train, model_type='random_forest')

# 6. Predict
predictions = detector.predict(X_test)
probabilities = detector.predict_proba(X_test)

# 7. Evaluate
metrics = detector.evaluate_classification(y_test, predictions)
print(f"F1 Score: {metrics['f1']:.4f}")

# 8. Visualize
fig1 = detector.plot_confusion_matrix(y_test, predictions)
fig2 = detector.plot_feature_importance(top_n=20)
plt.show()

# 9. Save
detector.save_model("models/my_model.pkl")
```

---

## Error Handling

### Model Not Trained
```python
if self.model is None:
    raise ValueError("No model loaded. Train or load a model first.")
```

### Feature Mismatch
```python
if X.shape[1] != len(self.feature_cols):
    raise ValueError(f"Expected {len(self.feature_cols)} features, got {X.shape[1]}")
```

### Invalid Target Column
```python
if target_col not in self.df_features.columns:
    raise KeyError(f"Target column '{target_col}' not found")
```

---

## Best Practices

1. **Always call `prepare_features()` before `prepare_train_test_split()`**
2. **Save scaler** along with model for consistent inference
3. **Use same target_col** throughout pipeline
4. **Check feature count** matches when loading model
5. **Validate predictions** before deployment
6. **Handle class imbalance** with balanced weights
7. **Cross-validate** for robust performance estimates

---

**End of Utils API Documentation**
