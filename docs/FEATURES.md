# Feature Documentation

## Overview

This document provides comprehensive information about the 36 features used in the exoplanet detection models. These features are derived from NASA's Kepler Objects of Interest (KOI) catalog and represent the most informative astronomical parameters for distinguishing between confirmed exoplanets, candidates, and false positives.

## Feature Files

### Primary Feature Files:
- **`features.json`**: Simple list of feature column names (legacy format)
- **`models/production/feature_columns.json`**: Copy of features for production models
- **`models/production/features_metadata.json`**: Comprehensive feature metadata with descriptions and groupings

### Usage:
```python
from src.feature_management import load_production_features

# Load feature manager
feature_manager = load_production_features("models/production")
features = feature_manager.get_feature_columns()
```

## Complete Feature List (36 features)

### 1. Orbital Parameters (9 features)

| Feature | Description | Units | Error Columns |
|---------|-------------|-------|---------------|
| `koi_period` | Orbital period | days | `koi_period_err1`, `koi_period_err2` |
| `koi_time0bk` | Time of first transit | BKJD | `koi_time0bk_err1`, `koi_time0bk_err2` |
| `koi_impact` | Impact parameter | dimensionless | `koi_impact_err1`, `koi_impact_err2` |

**Physical Meaning**: These parameters describe the orbital motion of the planet around its host star. The period determines how long it takes for one orbit, the time of first transit establishes the timing reference, and the impact parameter indicates how close to the center of the star the planet passes during transit.

### 2. Transit Characteristics (7 features)

| Feature | Description | Units | Error Columns |
|---------|-------------|-------|---------------|
| `koi_duration` | Transit duration | hours | `koi_duration_err1`, `koi_duration_err2` |
| `koi_depth` | Transit depth | ppm | `koi_depth_err1`, `koi_depth_err2` |
| `koi_model_snr` | Signal-to-noise ratio | dimensionless | - |

**Physical Meaning**: These features characterize the transit event itself. Duration tells us how long the planet takes to cross the star's disk, depth indicates how much the star's light dims (related to planet size), and SNR measures the quality of the detection.

### 3. Planetary Properties (7 features)

| Feature | Description | Units | Error Columns |
|---------|-------------|-------|---------------|
| `koi_prad` | Planetary radius | Earth radii | `koi_prad_err1`, `koi_prad_err2` |
| `koi_teq` | Equilibrium temperature | Kelvin | - |
| `koi_insol` | Insolation flux | Earth flux | `koi_insol_err1`, `koi_insol_err2` |

**Physical Meaning**: These are derived planetary properties. Radius indicates the planet's size relative to Earth, equilibrium temperature estimates the planet's temperature based on received stellar radiation, and insolation flux measures how much energy the planet receives compared to Earth.

### 4. Stellar Properties (10 features)

| Feature | Description | Units | Error Columns |
|---------|-------------|-------|---------------|
| `koi_steff` | Stellar effective temperature | Kelvin | `koi_steff_err1`, `koi_steff_err2` |
| `koi_slogg` | Stellar surface gravity | log₁₀(cm/s²) | `koi_slogg_err1`, `koi_slogg_err2` |
| `koi_srad` | Stellar radius | Solar radii | `koi_srad_err1`, `koi_srad_err2` |
| `koi_kepmag` | Kepler magnitude | magnitudes | - |

**Physical Meaning**: These characterize the host star. Effective temperature indicates the star's surface temperature and color, surface gravity relates to the star's mass and radius, stellar radius gives the star's size relative to the Sun, and Kepler magnitude measures the star's brightness.

### 5. Positional Data (2 features)

| Feature | Description | Units |
|---------|-------------|-------|
| `ra` | Right ascension | decimal degrees |
| `dec` | Declination | decimal degrees |

**Physical Meaning**: These are the star's coordinates on the sky, similar to longitude and latitude on Earth. They help identify the star's location for follow-up observations.

### 6. Catalog Information (1 feature)

| Feature | Description | Units |
|---------|-------------|-------|
| `koi_tce_plnt_num` | Planet number in system | integer |

**Physical Meaning**: This indicates which planet in a multi-planet system this candidate represents (1 for the first detected, 2 for the second, etc.).

## Feature Importance Rankings

Based on our trained models, the most important features for exoplanet detection are:

### Top 10 Most Important Features:
1. **`koi_depth`** - Transit depth (how much light is blocked)
2. **`koi_period`** - Orbital period (fundamental orbital characteristic)
3. **`koi_prad`** - Planetary radius (size indicator)
4. **`koi_srad`** - Stellar radius (host star size)
5. **`koi_model_snr`** - Signal-to-noise ratio (detection quality)
6. **`koi_duration`** - Transit duration (geometric constraint)
7. **`koi_steff`** - Stellar temperature (host star type)
8. **`koi_insol`** - Insolation flux (energy received)
9. **`koi_impact`** - Impact parameter (geometric alignment)
10. **`koi_kepmag`** - Kepler magnitude (stellar brightness)

## Data Quality and Missing Values

### Error Columns:
Most primary measurements have associated uncertainty estimates (err1 and err2). These represent the confidence intervals for the measurements and are crucial for understanding data quality.

### Missing Value Handling:
- **Strategy**: Median imputation for missing values
- **Rationale**: Median is robust to outliers, which are common in astronomical data
- **Preprocessing**: Applied during training and must be consistently applied during inference

### Typical Missing Value Rates:
- **Planetary properties**: ~20-30% (derived quantities)
- **Transit characteristics**: ~5-10% (direct measurements)
- **Stellar properties**: ~10-15% (from stellar characterization)
- **Positional data**: <1% (catalog data)

## Feature Engineering Opportunities

### Derived Features (not included in current model):
1. **Planet-to-star radius ratio**: `koi_prad / koi_srad`
2. **Duration-to-period ratio**: `koi_duration / koi_period`
3. **Density estimate**: Based on radius and period
4. **Temperature difference**: `koi_steff - koi_teq`
5. **Orbital distance estimate**: From period and stellar mass

### Interaction Terms:
The current models implicitly capture feature interactions through ensemble methods, but explicit polynomial features could be explored.

## Model Input Requirements

### Data Format:
- **Type**: Pandas DataFrame or NumPy array
- **Shape**: (n_samples, 36)
- **Data Types**: All numeric (float64 preferred)
- **Order**: Features must be in the exact order specified in `features.json`

### Preprocessing Requirements:
1. **Missing value imputation** (median strategy)
2. **Feature scaling** (StandardScaler normalization)
3. **Outlier handling** (optional, models are robust to outliers)

### Validation:
```python
from src.feature_management import validate_model_input

# Validate your data before inference
is_valid = validate_model_input(your_dataframe, "models/production")
```

## Scientific Significance

### Physical Relationships:
- **Transit depth ∝ (R_planet/R_star)²**: Larger planets block more light
- **Transit duration ∝ R_star/P**: Depends on stellar size and orbital period
- **Equilibrium temperature ∝ √(T_star × R_star / orbital_distance)**: Energy balance

### Classification Challenges:
- **False positives**: Often eclipsing binaries with similar transit signatures
- **Grazing transits**: Low impact parameter transits can be confused with noise
- **Stellar activity**: Star spots and flares can mimic planetary signals

### Model Strategy:
Our ensemble approach combines multiple algorithms to capture different aspects of the classification problem:
- **Random Forest**: Captures non-linear relationships and feature interactions
- **Extra Trees**: Reduces overfitting through increased randomization
- **AdaBoost**: Focuses on difficult-to-classify examples
- **Stacking**: Combines predictions optimally using meta-learning

## Version Control

### Feature Evolution:
- **v1.0**: Initial 36-feature set based on KOI catalog
- **Future**: May incorporate TESS-specific features or derived quantities

### Backward Compatibility:
The feature management system maintains backward compatibility with legacy feature files while supporting enhanced metadata for new deployments.

## Troubleshooting

### Common Issues:

1. **Feature count mismatch**: Ensure exactly 36 features are provided
2. **Column order**: Features must be in the specified order
3. **Missing columns**: Check that all required features are present
4. **Data types**: Ensure all features are numeric
5. **Scaling**: Apply the same StandardScaler used during training

### Debugging Tools:
```python
# Check feature availability
feature_manager = load_production_features()
missing = feature_manager.get_missing_columns(your_data)
print(f"Missing features: {missing}")

# Validate data
try:
    feature_manager.validate_input_data(your_data)
    print("✅ Data validation passed")
except ValueError as e:
    print(f"❌ Validation failed: {e}")
```

This comprehensive feature documentation ensures reproducible and reliable model inference for exoplanet detection.