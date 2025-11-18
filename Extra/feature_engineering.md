# Feature Engineering Script Documentation

## File: `feature_engineering.py`

### Purpose
Generate 67 sophisticated features from 5 raw IoT sensor measurements for anomaly detection in smart manufacturing equipment.

---

## Mathematical Overview

### Input Data
Raw sensor measurements at time $t$:
$$
\mathbf{x}_t = [T_t, V_t, H_t, P_t, E_t]^T
$$

Where:
- $T_t$ = Temperature (°C)
- $V_t$ = Vibration (Hz)
- $H_t$ = Humidity (%)
- $P_t$ = Pressure (bar)
- $E_t$ = Energy Consumption (kWh)

### Output Data
Feature vector with 67 dimensions:
$$
\mathbf{f}_t \in \mathbb{R}^{67}
$$

---

## Function Documentation

### `compute_features(data: pd.DataFrame) -> pd.DataFrame`

**Purpose**: Main function that orchestrates all feature engineering steps.

**Input**:
- `data`: DataFrame with columns `['timestamp', 'machine_id', 'temperature', 'vibration', 'humidity', 'pressure', 'energy_consumption', ...]`

**Output**:
- DataFrame with 67+ feature columns

**Process Flow**:
```
1. Validate input data
2. Sort by machine_id and timestamp
3. Compute basic transformations
4. Compute rolling window features
5. Compute statistical z-scores
6. Compute rate of change
7. Compute deviation features
8. Compute aggregate statistics
9. Compute interaction terms
10. Encode temporal features
11. Return complete feature set
```

**Mathematical Operations**:

#### 1. Basic Transformations
For each sensor $x_i$:
$$
\begin{align}
f_{\text{squared}}(x_i) &= x_i^2 \\
f_{\text{sqrt}}(x_i) &= \sqrt{x_i} \\
f_{\text{log}}(x_i) &= \log(x_i + 1)
\end{align}
$$

**Implementation**:
```python
data[f'{col}_squared'] = data[col] ** 2
data[f'{col}_sqrt'] = np.sqrt(data[col])
data[f'{col}_log'] = np.log1p(data[col])
```

---

#### 2. Statistical Z-Scores (Per Machine)
$$
z_{i,m}(t) = \frac{x_i(t) - \mu_{i,m}}{\sigma_{i,m} + \epsilon}
$$

Where:
- $\mu_{i,m}$ = mean of sensor $i$ for machine $m$
- $\sigma_{i,m}$ = standard deviation
- $\epsilon = 10^{-6}$ prevents division by zero

**Implementation**:
```python
grouped = data.groupby('machine_id')[col]
mean = grouped.transform('mean')
std = grouped.transform('std').fillna(0)
data[f'{col}_zscore_machine'] = (data[col] - mean) / (std + 1e-6)
```

**Purpose**: Normalize sensors relative to each machine's baseline operating conditions. Detects when a machine deviates from its typical behavior.

---

#### 3. Rolling Window Features (Per Machine, Window=5)

**Rolling Mean**:
$$
\bar{x}_{i,w}(t) = \frac{1}{w} \sum_{k=0}^{w-1} x_i(t-k)
$$

**Rolling Standard Deviation**:
$$
\sigma_{i,w}(t) = \sqrt{\frac{1}{w} \sum_{k=0}^{w-1} (x_i(t-k) - \bar{x}_{i,w}(t))^2}
$$

**Exponential Weighted Moving Average**:
$$
\text{EWM}_i(t) = \alpha x_i(t) + (1-\alpha) \text{EWM}_i(t-1)
$$
Where $\alpha = \frac{2}{w+1} = 0.333$ for span=5.

**Implementation**:
```python
data_sorted = data.sort_values(['machine_id', 'timestamp'])
grouped = data_sorted.groupby('machine_id')[col]

# Rolling mean
data[f'{col}_rolling_mean_5'] = grouped.transform(
    lambda x: x.rolling(window=5, min_periods=1).mean()
)

# Rolling std
data[f'{col}_rolling_std_5'] = grouped.transform(
    lambda x: x.rolling(window=5, min_periods=1).std().fillna(0)
)

# EWM
data[f'{col}_ewm'] = grouped.transform(
    lambda x: x.ewm(span=5, adjust=False).mean()
)
```

**Purpose**:
- Rolling mean: Smooths short-term fluctuations
- Rolling std: Detects instability/variability
- EWM: Gives more weight to recent observations

---

#### 4. Rate of Change (First-Order Difference)
$$
\Delta x_i(t) = x_i(t) - x_i(t-1)
$$

**Implementation**:
```python
grouped = data_sorted.groupby('machine_id')[col]
data[f'{col}_diff'] = grouped.transform(
    lambda x: x.diff().fillna(0)
)
```

**Purpose**: Detect sudden changes or spikes. Large $|\Delta x_i|$ suggests rapid transitions that may indicate failures.

---

#### 5. Deviation Features

**Absolute Deviation from Rolling Mean**:
$$
d_{\text{rolling}}(t) = |x_i(t) - \bar{x}_{i,w}(t)|
$$

**Anomaly Score** (Normalized Deviation):
$$
s_{\text{anomaly}}(t) = \frac{|x_i(t) - \bar{x}_{i,w}(t)|}{\sigma_{i,w}(t) + \epsilon}
$$

**Implementation**:
```python
data[f'{col}_dev_rolling'] = np.abs(
    data[col] - data[f'{col}_rolling_mean_5']
)

data[f'{col}_anomaly_score'] = (
    data[f'{col}_dev_rolling'] /
    (data[f'{col}_rolling_std_5'] + 1e-3)
)
```

**Purpose**:
- Deviation: Measures how far current value is from recent average
- Anomaly score: Normalized score (values > 3 are potential anomalies per 3-sigma rule)

---

#### 6. Aggregate Statistics (Per Machine)

For each machine $m$:
$$
\begin{align}
\mu_{i,m} &= \frac{1}{N_m} \sum_{t=1}^{N_m} x_i(t) \\
\sigma_{i,m} &= \sqrt{\frac{1}{N_m} \sum_{t=1}^{N_m} (x_i(t) - \mu_{i,m})^2} \\
\min_{i,m} &= \min_t x_i(t) \\
\max_{i,m} &= \max_t x_i(t) \\
\text{range}_{i,m} &= \max_{i,m} - \min_{i,m}
\end{align}
$$

**Implementation**:
```python
grouped = data.groupby('machine_id')[col]
data[f'{col}_machine_mean'] = grouped.transform('mean')
data[f'{col}_machine_std'] = grouped.transform('std').fillna(0)
data[f'{col}_machine_min'] = grouped.transform('min')
data[f'{col}_machine_max'] = grouped.transform('max')
data[f'{col}_machine_range'] = (
    data[f'{col}_machine_max'] - data[f'{col}_machine_min']
)
```

**Purpose**: Capture machine-level operating characteristics and ranges.

---

#### 7. Interaction Features

**Temperature × Vibration**:
$$
f_{T \times V} = T \cdot V
$$
*Rationale*: Overheating often causes increased vibration due to thermal expansion and bearing degradation.

**Humidity × Pressure**:
$$
f_{H \times P} = H \cdot P
$$
*Rationale*: Environmental interaction effects on machine performance.

**Energy × Temperature**:
$$
f_{E \times T} = E \cdot T
$$
*Rationale*: Energy efficiency indicator (high energy + high temperature = inefficiency or malfunction).

**Implementation**:
```python
data['temp_vibration_interaction'] = (
    data['temperature'] * data['vibration']
)
data['humidity_pressure_interaction'] = (
    data['humidity'] * data['pressure']
)
data['energy_temp_interaction'] = (
    data['energy_consumption'] * data['temperature']
)
```

---

#### 8. Temporal Encodings

**Cyclical Hour Encoding**:
$$
\begin{align}
h_{\sin} &= \sin\left(\frac{2\pi h}{24}\right) \\
h_{\cos} &= \cos\left(\frac{2\pi h}{24}\right)
\end{align}
$$
Where $h \in [0, 23]$.

**Cyclical Day-of-Week Encoding**:
$$
\begin{align}
d_{\sin} &= \sin\left(\frac{2\pi d}{7}\right) \\
d_{\cos} &= \cos\left(\frac{2\pi d}{7}\right)
\end{align}
$$
Where $d \in [0, 6]$.

**Implementation**:
```python
data['timestamp'] = pd.to_datetime(data['timestamp'])
data['hour'] = data['timestamp'].dt.hour
data['day_of_week'] = data['timestamp'].dt.dayofweek
data['day'] = data['timestamp'].dt.day
data['month'] = data['timestamp'].dt.month

# Cyclical encoding
data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
```

**Purpose**:
- Capture time-of-day patterns (shift work, peak hours)
- Capture weekly patterns (weekday vs weekend)
- Cyclical encoding ensures continuity (23:00 is close to 00:00)

---

## Feature Count Breakdown

| Category | Formula | Count | Example |
|----------|---------|-------|---------|
| **Raw** | $x_i$ | 5 | `temperature` |
| **Squared** | $x_i^2$ | 5 | `temperature_squared` |
| **Square Root** | $\sqrt{x_i}$ | 5 | `temperature_sqrt` |
| **Log** | $\log(x_i + 1)$ | 5 | `temperature_log` |
| **Z-scores** | $(x_i - \mu)/\sigma$ | 5 | `temperature_zscore_machine` |
| **Rolling Mean** | $\frac{1}{w}\sum x_i$ | 5 | `temperature_rolling_mean_5` |
| **Rolling Std** | $\sqrt{\frac{1}{w}\sum(x_i - \bar{x})^2}$ | 5 | `temperature_rolling_std_5` |
| **EWM** | $\alpha x_i + (1-\alpha)\text{EWM}_{t-1}$ | 5 | `temperature_ewm` |
| **Diff** | $x_i(t) - x_i(t-1)$ | 5 | `temperature_diff` |
| **Dev Rolling** | $\|x_i - \bar{x}_w\|$ | 5 | `temperature_dev_rolling` |
| **Anomaly Score** | $\|x_i - \bar{x}_w\|/\sigma_w$ | 1 | `temperature_anomaly_score` |
| **Machine Mean** | $\mu_{i,m}$ | 5 | `temperature_machine_mean` |
| **Machine Std** | $\sigma_{i,m}$ | 0 | (included in z-score) |
| **Interactions** | $x_i \cdot x_j$ | 3 | `temp_vibration_interaction` |
| **Temporal** | $\sin/\cos$ encodings | 8 | `hour_sin`, `hour_cos`, ... |
| **Total** | | **67** | |

---

## Usage Examples

### Command Line
```bash
python scripts/feature_engineering.py \
    --input data/smart_manufacturing_data.csv \
    --output features/engineered_features.csv
```

### Python API
```python
from scripts.feature_engineering import compute_features
import pandas as pd

# Load raw data
df_raw = pd.read_csv("data/smart_manufacturing_data.csv")

# Compute features
df_features = compute_features(df_raw)

# Save
df_features.to_csv("features/engineered_features.csv", index=False)
```

---

## Computational Complexity

### Time Complexity
For $n$ samples and $m$ machines:
- **Basic transformations**: $O(n)$
- **Groupby operations**: $O(n \log n)$ (sorting) + $O(n)$ (aggregation)
- **Rolling windows**: $O(n \cdot w) = O(5n) = O(n)$ for window=5
- **Overall**: $O(n \log n)$

### Space Complexity
- **Input**: $O(n \cdot 5)$ for 5 raw features
- **Output**: $O(n \cdot 67)$ for 67 engineered features
- **Temporary**: $O(n)$ for intermediate calculations

For 100,000 samples:
- Input: ~4 MB (5 features × 8 bytes)
- Output: ~54 MB (67 features × 8 bytes)
- Disk (CSV): ~93 MB (with headers and text encoding)

---

## Error Handling

### Missing Values
```python
# Rolling std with no variance
.fillna(0)

# Division by zero in z-scores
(std + 1e-6)

# Log of zero
np.log1p(x)  # log(1 + x)

# First diff at t=0
.diff().fillna(0)
```

### Edge Cases
1. **Single machine**: All machine-level aggregates work correctly
2. **Single timestamp**: Rolling windows use `min_periods=1`
3. **Negative values**: Square root uses absolute values or clipping
4. **Extreme outliers**: Z-scores can handle but log may overflow (handled by log1p)

---

## Performance Optimization

### Vectorization
All operations use NumPy/Pandas vectorized operations:
```python
# BAD (loop)
for i in range(len(data)):
    data.loc[i, 'squared'] = data.loc[i, 'temp'] ** 2

# GOOD (vectorized)
data['squared'] = data['temp'] ** 2
```

### Groupby Efficiency
```python
# Efficient: compute once, transform many
grouped = data.groupby('machine_id')[col]
data['mean'] = grouped.transform('mean')
data['std'] = grouped.transform('std')
```

### Memory Management
```python
# Process in chunks for large datasets
for chunk in pd.read_csv(path, chunksize=10000):
    features = compute_features(chunk)
    features.to_csv(output, mode='a', header=False)
```

---

## Validation

### Feature Sanity Checks
```python
# Check for NaN/Inf
assert not data.isna().any().any()
assert not np.isinf(data.select_dtypes(include=np.number)).any().any()

# Check feature ranges
assert (data['hour_sin'] >= -1).all() and (data['hour_sin'] <= 1).all()
assert (data['hour_cos'] >= -1).all() and (data['hour_cos'] <= 1).all()

# Check no negative sqrt
assert (data['temperature_sqrt'] >= 0).all()
```

---

## References

1. **Rolling Window Features**: Time series analysis, exponential smoothing
2. **Z-Score Normalization**: Standard statistical normalization
3. **Cyclical Encoding**: Temporal feature encoding for periodic patterns
4. **Interaction Terms**: Feature engineering for non-linear relationships
5. **Anomaly Scores**: Statistical process control (3-sigma rule)

---

## Maintenance Notes

### Adding New Features
To add new features:
1. Define mathematical formula
2. Implement vectorized computation
3. Update feature count in documentation
4. Add validation checks
5. Test on sample data

### Modifying Window Sizes
Current window size is 5. To change:
```python
WINDOW_SIZE = 10  # Change from 5 to 10
data[f'{col}_rolling_mean_{WINDOW_SIZE}'] = ...
```

---

**End of Feature Engineering Documentation**
