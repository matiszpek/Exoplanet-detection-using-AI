# API Documentation

## Overview

This document provides technical documentation for the Exoplanet Detection AI system, including detailed information about the codebase, models, and usage patterns.

## Module Documentation

### data_preprocessing.py

#### ExoplanetDataPreprocessor

Main class for handling data preprocessing operations.

**Key Methods:**

- `load_data(filepath, comment='#')`: Load astronomical data from CSV
- `remove_irrelevant_columns(df, columns_to_drop=None)`: Clean dataset by removing unnecessary columns
- `handle_missing_values(df)`: Impute missing values using median strategy
- `prepare_features_target(df, target_column, target_mapping, binary_classification=True)`: Prepare X, y for ML
- `scale_features(X_train, X_test=None)`: Apply StandardScaler normalization
- `detect_outliers(X, threshold=3)`: Identify outliers using Z-score method

**Usage Example:**
```python
from src.data_preprocessing import ExoplanetDataPreprocessor

preprocessor = ExoplanetDataPreprocessor()
df = preprocessor.load_data('data/cumulative_2025.10.04_05.21.55.csv')
df_clean = preprocessor.remove_irrelevant_columns(df)
df_imputed = preprocessor.handle_missing_values(df_clean)
X, y = preprocessor.prepare_features_target(df_imputed)
X_scaled = preprocessor.scale_features(X)
```

### model_training.py

#### ExoplanetClassifier

Comprehensive classification system with multiple ML algorithms.

**Key Methods:**

- `build_models()`: Initialize collection of ML models (RF, Extra Trees, AdaBoost, Stacking)
- `train_models(X_train, y_train, X_test, y_test)`: Train all models and evaluate performance
- `select_best_model(metric='f1_score')`: Choose best performing model
- `save_models(output_dir, feature_names=None)`: Persist models and metadata
- `get_feature_importance(model_name=None, top_n=20)`: Extract feature importance rankings
- `classify_candidates(data_path, model_path=None, threshold=0.5)`: Classify new observations

**Supported Models:**
- Random Forest Classifier (1600 trees, entropy criterion)
- Extra Trees Classifier (200 trees, entropy criterion)  
- AdaBoost Classifier (974 estimators, 0.1 learning rate)
- Gradient Boosting Classifier (1600 estimators, 0.1 learning rate)
- Stacking Classifier (RF + GB with Logistic Regression meta-learner)

**Usage Example:**
```python
from src.model_training import ExoplanetClassifier

classifier = ExoplanetClassifier(random_state=42)
results = classifier.train_models(X_train, y_train, X_test, y_test)
best_name, best_model = classifier.select_best_model('f1_score')
classifier.save_models('models/production')
```

### feature_engineering.py

#### ExoplanetFeatureEngineer

Feature creation and selection utilities.

**Key Methods:**

- `create_derived_features(df)`: Generate derived astronomical features
- `select_features(X, y, method='mutual_info', k=20)`: Select most informative features
- `apply_pca(X, n_components=0.95)`: Dimensionality reduction via PCA
- `create_polynomial_features(X, degree=2)`: Generate interaction terms
- `detect_correlated_features(df, threshold=0.95)`: Find highly correlated feature pairs
- `analyze_feature_distributions(df, target_column)`: Statistical analysis by class

**Derived Features Created:**
- Planet-to-star radius ratio
- Transit duration to period ratio  
- Planet density estimates
- Equilibrium temperature gradients
- Orbital distance estimates
- Signal strength indicators
- Measurement error ratios

**Usage Example:**
```python
from src.feature_engineering import ExoplanetFeatureEngineer

engineer = ExoplanetFeatureEngineer()
df_enhanced = engineer.create_derived_features(df)
X_selected = engineer.select_features(X, y, method='mutual_info', k=30)
correlated_pairs = engineer.detect_correlated_features(df_enhanced)
```

### evaluation.py

#### ExoplanetModelEvaluator

Comprehensive model evaluation and visualization tools.

**Key Methods:**

- `calculate_comprehensive_metrics(y_true, y_pred, y_proba)`: Full metric calculation
- `evaluate_stratified_performance(y_true, y_pred, y_proba, subgroup_labels)`: Performance by subgroup
- `plot_confusion_matrix(y_true, y_pred)`: Confusion matrix visualization
- `plot_roc_curve(y_true, y_proba)`: ROC curve with AUC
- `plot_precision_recall_curve(y_true, y_proba)`: PR curve with AP score
- `plot_threshold_analysis(y_true, y_proba)`: Threshold optimization plots
- `create_evaluation_report(y_true, y_pred, y_proba)`: Comprehensive report generation

**Metrics Calculated:**
- Standard: Accuracy, Precision, Recall, F1-score
- Probabilistic: ROC-AUC, PR-AUC
- Robust: Matthews Correlation Coefficient
- Confusion Matrix: TP, TN, FP, FN
- Derived: Specificity, Sensitivity, FPR, FNR

**Usage Example:**
```python
from src.evaluation import ExoplanetModelEvaluator

evaluator = ExoplanetModelEvaluator()
metrics = evaluator.calculate_comprehensive_metrics(y_test, y_pred, y_proba)
report = evaluator.create_evaluation_report(y_test, y_pred, y_proba, model_name="RandomForest")
fig_roc = evaluator.plot_roc_curve(y_test, y_proba)
```

## Model Specifications

### Binary Classification Models

**Target Classes:**
- Class 1: CONFIRMED (confirmed exoplanets)
- Class 0: CANDIDATE (potential exoplanets)

**Performance Benchmarks:**
- Accuracy: >93%
- F1-Score: >93%
- ROC-AUC: >0.98
- Precision: >92%
- Recall: >94%

### False Positive Detection Models

**Enhanced Approach:**
- Incorporates FALSE POSITIVE samples in training
- 50/50 split of false positives between train/test
- Threshold optimization for operational deployment
- Stratified evaluation by object type

**Target Classes:**
- Class 1: CONFIRMED (true exoplanets)
- Class 0: CANDIDATE + FALSE POSITIVE (non-exoplanets)

## Feature Documentation

### Core Astronomical Features

**Orbital Parameters:**
- `koi_period`: Orbital period (days)
- `koi_time0bk`: Time of first transit (BKJD)
- `koi_impact`: Impact parameter
- `koi_duration`: Transit duration (hours)

**Transit Characteristics:**
- `koi_depth`: Transit depth (ppm)
- `koi_model_snr`: Transit signal-to-noise ratio

**Planetary Properties:**
- `koi_prad`: Planetary radius (Earth radii)
- `koi_teq`: Equilibrium temperature (K)
- `koi_insol`: Insolation flux (Earth flux)

**Stellar Properties:**
- `koi_steff`: Stellar effective temperature (K)
- `koi_slogg`: Stellar surface gravity (log10(cm/s²))
- `koi_srad`: Stellar radius (Solar radii)
- `koi_kepmag`: Kepler magnitude

**Positional Data:**
- `ra`: Right ascension (decimal degrees)
- `dec`: Declination (decimal degrees)

### Derived Features

**Physical Ratios:**
- Planet-to-star radius ratio
- Transit duration to period ratio
- Error ratios for uncertainty quantification

**Calculated Properties:**
- Planet density estimates
- Orbital distance estimates  
- Temperature gradients
- Signal strength indicators

## Performance Optimization

### Computational Efficiency

**Parallel Processing:**
- All ensemble models use `n_jobs=-1` for maximum CPU utilization
- Stacking classifier employs 5-fold cross-validation for meta-learning

**Memory Management:**
- Feature scaling applied to reduce memory footprint
- Model compression (joblib compression level 3) for storage efficiency

**Training Speed:**
- Random Forest: ~2-3 minutes on standard hardware
- Extra Trees: ~1-2 minutes (faster due to random splits)
- AdaBoost: ~3-5 minutes (sequential nature)
- Stacking: ~5-8 minutes (includes base model training + meta-learning)

### Scalability Considerations

**Data Volume:**
- Current implementation handles datasets up to ~100K observations
- Memory usage scales linearly with feature count
- Batch processing available for larger datasets

**Feature Dimensionality:**
- PCA available for dimensionality reduction
- Feature selection reduces computational load
- Polynomial features can be computationally expensive (use judiciously)

## Error Handling

### Common Issues and Solutions

**Data Loading Errors:**
```python
# Handle malformed CSV files
try:
    df = pd.read_csv(filepath, comment='#', engine='python')
except pd.errors.ParserError:
    df = pd.read_csv(filepath, comment='#', engine='python', error_bad_lines=False)
```

**Missing Value Handling:**
```python
# Check for columns with all missing values
missing_cols = df.columns[df.isnull().all()].tolist()
df = df.drop(columns=missing_cols)
```

**Model Training Failures:**
```python
# Handle single-class scenarios
unique_classes = np.unique(y_train)
if len(unique_classes) < 2:
    raise ValueError(f"Training data contains only {len(unique_classes)} class(es)")
```

### Debugging Tools

**Data Validation:**
```python
from sklearn.utils.multiclass import type_of_target
print(f"Target type: {type_of_target(y)}")
print(f"Unique values: {np.unique(y)}")
```

**Model Diagnostics:**
```python
# Check model classes
print(f"Model learned classes: {model.classes_}")
print(f"Feature importance available: {hasattr(model, 'feature_importances_')}")
```

## Deployment Considerations

### Model Persistence

**Saving Models:**
```python
import joblib
joblib.dump(model, 'model.pkl', compress=3)
joblib.dump(scaler, 'scaler.pkl')
```

**Loading Models:**
```python
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
```

### Production Pipeline

**Preprocessing Pipeline:**
1. Load raw astronomical data
2. Remove irrelevant columns
3. Handle missing values (median imputation)
4. Scale features using fitted scaler
5. Apply trained model
6. Post-process predictions (threshold application)

**Monitoring:**
- Track prediction distributions
- Monitor feature drift
- Validate data quality metrics
- Log model performance over time

## Version Compatibility

**Python Requirements:**
- Python 3.8+
- NumPy 1.24+
- Pandas 1.5+
- Scikit-learn 1.3+

**Known Issues:**
- Some visualization functions require matplotlib 3.6+
- PCA transforms may differ slightly between scikit-learn versions
- Random state reproducibility requires identical package versions

## Support and Troubleshooting

**Common Error Messages:**

1. `ValueError: Input contains NaN`: Check data preprocessing steps
2. `ValueError: Sample weights must be positive`: Ensure no zero/negative weights
3. `AttributeError: 'NoneType' object has no attribute`: Check model training completion

**Performance Issues:**

1. Slow training: Reduce n_estimators or use parallel processing
2. High memory usage: Use PCA or feature selection
3. Poor convergence: Check feature scaling and class balance

For additional support, please refer to the project's GitHub issues or create a new issue with detailed error information.