# Visualization Script Documentation

## File: `create_visualizations.py`

### Purpose
Automatically generate comprehensive comparison charts for model performance analysis.

---

## Charts Generated

### 1. Classification Metrics Comparison (4 charts)
- `comparison_f1_mean.png` - F1 score comparison
- `comparison_precision_mean.png` - Precision comparison
- `comparison_recall_mean.png` - Recall comparison
- `comparison_accuracy_mean.png` - Accuracy comparison

### 2. Regression Metrics Comparison (1 chart)
- `comparison_regression.png` - MAE, RMSE, R² comparison

### 3. Summary Charts (3 charts)
- `best_models_summary.png` - Best model for each task
- `training_time_comparison.png` - Training time analysis
- `best_models_summary.csv` - Summary table

**Total**: 8 visualizations

---

## Function Documentation

### `plot_classification_comparison(results, metric, task_name, output_dir)`

**Purpose**: Create bar chart comparing all models on a classification metric

**Mathematical Representation**:

For each model $M_i$:
$$
\bar{m}_i = \frac{1}{K} \sum_{k=1}^{K} m_{i,k}
$$

Where $m$ is the metric (F1, precision, recall, accuracy).

**Visualization**:
```
Bar chart with:
- X-axis: Model names
- Y-axis: Metric value [0, 1]
- Colors: Gradient (best = green, worst = red)
- Error bars: Standard deviation across folds
```

**Implementation**:
```python
plt.figure(figsize=(14, 6))
plt.bar(models, scores, color=colors, edgecolor='black')
plt.errorbar(models, scores, yerr=std_devs, fmt='none',
             ecolor='black', capsize=5)
plt.ylabel(f'{metric.replace("_", " ").title()}')
plt.title(f'{task_name} - {metric.replace("_", " ").title()} Comparison')
plt.xticks(rotation=45, ha='right')
plt.ylim([0, 1.05])
plt.tight_layout()
```

---

### `plot_regression_comparison(results, task_name, output_dir)`

**Purpose**: Create grouped bar chart for regression metrics

**Metrics Visualized**:
1. **MAE** (Mean Absolute Error):
   $$
   \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
   $$

2. **RMSE** (Root Mean Squared Error):
   $$
   \text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
   $$

3. **R²** (Coefficient of Determination):
   $$
   R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
   $$

**Normalization for Visualization**:
Since metrics have different scales:
- MAE, RMSE: Lower is better → plot as-is
- R²: Higher is better, range [-∞, 1] → plot directly

**Implementation**:
```python
x = np.arange(len(models))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(x - width, mae, width, label='MAE', color='coral')
ax.bar(x, rmse, width, label='RMSE', color='skyblue')
ax.bar(x + width, r2, width, label='R²', color='lightgreen')

ax.set_xlabel('Models')
ax.set_ylabel('Metric Value')
ax.set_title(f'{task_name} - Regression Metrics Comparison')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right')
ax.legend()
plt.tight_layout()
```

---

### `plot_best_models_summary(results_dir, output_dir)`

**Purpose**: Create summary visualization of best models for all tasks

**Process**:
1. Load all results CSVs
2. Extract best model for each task:
   $$
   M_{\text{task}}^* = \arg\max_M \bar{F}_{1,M} \quad \text{or} \quad \arg\max_M R^2_M
   $$
3. Create summary DataFrame
4. Plot bar chart with best scores

**Summary Table Format**:
```
Task                    | Best Model      | F1/R² Score | Train Time (s)
------------------------|-----------------|-------------|---------------
Anomaly Detection       | Random Forest   | 0.9998      | 0.31
Maintenance Prediction  | Random Forest   | 0.9821      | 0.72
Downtime Risk          | Random Forest   | 0.9998      | 0.34
Failure Type           | Gaussian NB     | 0.9300      | 0.00
Remaining Life         | Random Forest   | 0.18 (R²)   | 6.98
```

**Visualization**:
```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Performance chart
ax1.barh(tasks, scores, color=colors)
ax1.set_xlabel('Performance Score')
ax1.set_title('Best Model Performance by Task')

# Training time chart
ax2.barh(tasks, times, color='coral')
ax2.set_xlabel('Training Time (seconds)')
ax2.set_title('Best Model Training Time by Task')
```

---

### `plot_training_time_comparison(results_dir, output_dir)`

**Purpose**: Analyze training time across all models

**Metrics**:
- Mean training time per fold
- Total training time (mean × K folds)
- Time efficiency ratio:
  $$
  \text{Efficiency} = \frac{F_1}{\text{Time}} \quad \text{(higher is better)}
  $$

**Visualization**:
Stacked bar chart showing training time for all models across all tasks.

**Implementation**:
```python
fig, ax = plt.subplots(figsize=(14, 8))

bottom = np.zeros(len(models))
for task, times in task_times.items():
    ax.bar(models, times, label=task, bottom=bottom)
    bottom += times

ax.set_ylabel('Total Training Time (s)')
ax.set_xlabel('Models')
ax.set_title('Training Time Comparison Across All Tasks')
ax.legend(title='Tasks')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
```

---

## Color Schemes

### Performance-Based Gradient
```python
def get_color_gradient(scores):
    """
    Map scores to color gradient:
    - Best (max score): Dark green (#2ca02c)
    - Worst (min score): Light red (#d62728)
    """
    norm = plt.Normalize(vmin=min(scores), vmax=max(scores))
    cmap = plt.cm.RdYlGn  # Red-Yellow-Green colormap
    return [cmap(norm(score)) for score in scores]
```

### Task-Based Colors
```python
TASK_COLORS = {
    'anomaly': '#1f77b4',      # Blue
    'maintenance': '#ff7f0e',  # Orange
    'downtime': '#2ca02c',     # Green
    'failure_type': '#d62728', # Red
    'remaining_life': '#9467bd' # Purple
}
```

---

## Statistical Annotations

### Error Bars
Standard deviation across folds:
$$
\sigma_m = \sqrt{\frac{1}{K} \sum_{k=1}^{K} (m_k - \bar{m})^2}
$$

Displayed as black error bars with caps.

### Value Labels
Display metric value on top of each bar:
```python
for i, (model, score) in enumerate(zip(models, scores)):
    ax.text(i, score + 0.02, f'{score:.4f}',
            ha='center', va='bottom', fontsize=9)
```

---

## Usage

### Command Line
```bash
python scripts/create_visualizations.py \
    --results_dir results \
    --output_dir charts
```

### Python API
```python
from create_visualizations import (
    plot_classification_comparison,
    plot_regression_comparison,
    plot_best_models_summary
)

# Classification task
plot_classification_comparison(
    results=pd.read_csv('results/results_anomaly.csv'),
    metric='f1_mean',
    task_name='Anomaly Detection',
    output_dir='charts'
)

# Regression task
plot_regression_comparison(
    results=pd.read_csv('results/results_remaining_life.csv'),
    task_name='Remaining Useful Life',
    output_dir='charts'
)

# Summary
plot_best_models_summary(
    results_dir='results',
    output_dir='charts'
)
```

---

## Customization Options

### Figure Size
```python
plt.figure(figsize=(width, height))  # Default: (14, 6)
```

### DPI (Resolution)
```python
plt.savefig(path, dpi=300, bbox_inches='tight')  # Default: 300 dpi
```

### Style Theme
```python
plt.style.use('seaborn-v0_8-darkgrid')  # Professional style
```

### Font Sizes
```python
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
```

---

## Output Specifications

### Image Format
- **Format**: PNG (Portable Network Graphics)
- **DPI**: 300 (publication quality)
- **Color**: RGB, 24-bit
- **Transparency**: None (white background)

### File Sizes (Typical)
- Classification comparison: ~100 KB
- Regression comparison: ~120 KB
- Best models summary: ~150 KB
- Training time: ~180 KB

---

## Mathematical Interpretation

### F1 Score Scale
$$
F_1 \in [0, 1]
$$
- $F_1 > 0.9$: Excellent (green)
- $0.8 < F_1 \leq 0.9$: Good (yellow-green)
- $0.7 < F_1 \leq 0.8$: Fair (yellow)
- $F_1 \leq 0.7$: Poor (red)

### R² Scale
$$
R^2 \in (-\infty, 1]
$$
- $R^2 > 0.9$: Excellent
- $0.7 < R^2 \leq 0.9$: Good
- $0.5 < R^2 \leq 0.7$: Moderate
- $R^2 \leq 0.5$: Poor

### Training Time
$$
T_{\text{total}} = K \cdot \bar{t}
$$

Where $K$ = number of folds, $\bar{t}$ = mean time per fold.

---

## Best Practices

1. **Always generate all charts** for complete analysis
2. **Use consistent color schemes** across tasks
3. **Include error bars** to show variance
4. **Label axes clearly** with units
5. **Save high-resolution** images (DPI ≥ 300)
6. **Export summary table** as CSV for reports
7. **Compare apples-to-apples**: same sample size, same K

---

**End of Visualization Script Documentation**
