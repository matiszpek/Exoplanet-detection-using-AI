# Project Structure

This document provides an overview of the organized project structure for the Exoplanet Detection using AI system.

## 📁 Directory Layout

```
Exoplanet-detection-using-AI/
├── README.md                           # Main project documentation
├── LICENSE                             # MIT License
├── CONTRIBUTING.md                     # Contribution guidelines
├── requirements.txt                    # Python dependencies
├── features.json                       # Feature list for model inference
│
├── data/                              # Dataset files
│   ├── cumulative_2025.10.04_05.21.55.csv    # Kepler cumulative catalog
│   ├── k2pandc_2025.10.04_05.29.43.csv       # K2 planet candidates
│   └── TOI_2025.10.04_05.29.26.csv           # TESS Objects of Interest
│
├── notebooks/                         # Jupyter notebooks
│   ├── 01_exoplanet_detection_binary.ipynb           # Binary classification
│   ├── 02_false_positive_detection.ipynb             # False positive handling
│   └── archived/                      # Previous versions
│       ├── exoplanet_ml_final_fast.ipynb
│       ├── exoplanet_ml_final.ipynb
│       └── kono/                      # Alternative implementations
│           ├── modelCumulativeKono copy.ipynb
│           └── modelTOIKono.ipynb
│
├── src/                               # Source code modules
│   ├── __init__.py                    # Package initialization
│   ├── data_preprocessing.py          # Data cleaning and preprocessing
│   ├── feature_engineering.py        # Feature creation and selection
│   ├── model_training.py             # Model training utilities
│   └── evaluation.py                 # Model evaluation metrics
│
├── models/                            # Trained models and artifacts
│   ├── production/                    # Production-ready models
│   │   └── sub_models/               # Final trained models
│   │       ├── adaboost.pkl
│   │       ├── extra_trees.pkl
│   │       ├── random_forest.pkl
│   │       └── stacking.pkl
│   ├── experiments/                   # Experimental models
│   │   ├── fps/                      # False positive experiments
│   │   │   └── experimento/          # Specific experiment runs
│   │   └── analysis/                 # Analysis artifacts
│   │       ├── cv_summary.csv
│   │       ├── feature_importances_by_permutation.csv
│   │       ├── feature_importances_by_weight.csv
│   │       └── feature_importances_comparison.csv
│   └── archived/                     # Previous model versions
│       ├── discontinued_models/      # Older model implementations
│       └── sub_models/              # Legacy models
│
├── docs/                             # Documentation
│   ├── API.md                        # Technical API documentation
│   ├── CHANGELOG.md                  # Version history
│   └── DEPLOYMENT.md                 # Deployment instructions
│
└── .git/                             # Git version control
    ├── .gitignore                    # Git ignore rules
    └── .gitattributes               # Git attributes
```

## 📋 File Descriptions

### Core Files

- **README.md**: Comprehensive project overview, installation instructions, and usage guide
- **LICENSE**: MIT License for open-source distribution
- **CONTRIBUTING.md**: Guidelines for contributors
- **requirements.txt**: Python package dependencies with versions
- **features.json**: List of feature columns used by trained models

### Data Directory (`data/`)

Contains astronomical datasets from NASA missions:
- **Kepler Cumulative Catalog**: Complete list of Kepler Objects of Interest
- **K2 Campaign Data**: Targets from K2 extended mission
- **TESS Objects of Interest**: Candidates from TESS mission

### Notebooks Directory (`notebooks/`)

Organized Jupyter notebooks for different aspects of the project:
- **Binary Classification**: Main exoplanet vs candidate classification
- **False Positive Detection**: Enhanced classification including false positives (includes visualization and analysis)
- **Archived**: Historical notebooks and alternative implementations

### Source Code (`src/`)

Modular Python codebase:
- **data_preprocessing.py**: Data loading, cleaning, and preparation utilities
- **feature_engineering.py**: Feature creation, selection, and transformation
- **model_training.py**: Machine learning model training and management
- **evaluation.py**: Comprehensive model evaluation and visualization tools

### Models Directory (`models/`)

Organized model storage:
- **production/**: Finalized models ready for deployment
- **experiments/**: Experimental models and analysis results
- **archived/**: Previous model versions and discontinued approaches

### Documentation (`docs/`)

Technical documentation:
- **API.md**: Detailed API documentation for code modules
- **CHANGELOG.md**: Version history and changes
- **DEPLOYMENT.md**: Instructions for deploying models

## 🔄 Workflow

### Development Workflow

1. **Data Exploration**: Start with the exploration sections in `notebooks/01_exoplanet_detection_binary.ipynb`
2. **Binary Classification**: Implement basic classification in `notebooks/01_exoplanet_detection_binary.ipynb`
3. **Advanced Classification**: Enhance with false positive detection in `notebooks/02_false_positive_detection.ipynb`
4. **Modular Development**: Extract reusable code to `src/` modules
5. **Model Management**: Save final models to `models/production/`

### Data Science Pipeline

```
Raw Data (data/) 
    ↓
Data Preprocessing (src/data_preprocessing.py)
    ↓
Feature Engineering (src/feature_engineering.py)
    ↓
Model Training (src/model_training.py)
    ↓
Model Evaluation (src/evaluation.py)
    ↓
Production Models (models/production/)
```

## 🎯 Best Practices

### Code Organization

- **Modular Design**: Separate concerns into different modules
- **Reusability**: Common functions in `src/` modules
- **Documentation**: Comprehensive docstrings and comments
- **Version Control**: Track changes with meaningful commit messages

### Data Management

- **Raw Data Preservation**: Keep original datasets unchanged
- **Reproducibility**: Fixed random seeds and versioned dependencies
- **Backup**: Multiple copies of important datasets
- **Documentation**: Clear data source attribution

### Model Management

- **Version Control**: Track model versions and performance
- **Artifacts**: Save scalers, feature lists, and metadata
- **Organization**: Separate experimental from production models
- **Documentation**: Record training procedures and hyperparameters

## 🔧 Maintenance

### Regular Tasks

- **Dependency Updates**: Keep packages current with security patches
- **Model Retraining**: Retrain with new data as available
- **Performance Monitoring**: Track model performance over time
- **Documentation Updates**: Keep documentation current with code changes

### Quality Assurance

- **Code Review**: Review all changes before merging
- **Testing**: Run notebooks end-to-end before releases
- **Validation**: Verify model performance on new data
- **Backup**: Regular backups of models and results

This organized structure ensures the project is maintainable, scalable, and suitable for professional development and deployment.