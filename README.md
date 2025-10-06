# 🪐 Exoplanet Detection using AI

[![NASA Space Apps Challenge 2025](https://img.shields.io/badge/NASA%20Space%20Apps-2025-blue?style=for-the-badge&logo=nasa)](https://www.spaceappschallenge.org/2025/challenges/a-world-away-hunting-for-exoplanets-with-ai/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat-square&logo=jupyter)](https://jupyter.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat-square&logo=scikit-learn)](https://scikit-learn.org)

## 🌟 Project Overview

This project was developed for the **NASA Space Apps Challenge 2025** under the challenge "A World Away: Hunting for Exoplanets with AI". Our mission is to leverage artificial intelligence and machine learning techniques to detect and classify exoplanets using data from NASA's Kepler Space Telescope and TESS (Transiting Exoplanet Survey Satellite) missions.

### 🎯 Challenge Objective

The goal is to develop an AI system capable of:
- **Detecting exoplanet candidates** from astronomical transit photometry data
- **Distinguishing between confirmed planets, candidates, and false positives**
- **Improving the accuracy** of exoplanet detection to aid in the search for potentially habitable worlds

## 🔬 Scientific Background

Exoplanets are planets that orbit stars outside our solar system. The transit method detects these planets by observing the slight dimming of a star's light when a planet passes in front of it. However, this signal can be very subtle and often contaminated by:

- **Instrumental noise**
- **Stellar variability**
- **False positive signals** (eclipsing binaries, background stars, etc.)

Our AI models analyze multiple features from the light curves and stellar parameters to make accurate classifications.

## 📊 Dataset

We use publicly available data from NASA's exoplanet archive, specifically:

- **Kepler Objects of Interest (KOI)** cumulative catalog
- **TESS Objects of Interest (TOI)** catalog  
- **K2 Campaign** target lists

### Key Features Used:
- **Orbital Parameters**: Period, epoch, impact parameter, duration
- **Transit Characteristics**: Depth, signal-to-noise ratio
- **Planetary Properties**: Radius, equilibrium temperature, insolation
- **Stellar Properties**: Effective temperature, surface gravity, radius, magnitude
- **Positional Data**: Right ascension, declination

## 🧠 Machine Learning Approach

### Models Implemented:

1. **Random Forest Classifier** - Ensemble method for robust predictions
2. **Extra Trees Classifier** - Extremely randomized trees for variance reduction
3. **AdaBoost Classifier** - Adaptive boosting for sequential learning
4. **Gradient Boosting Classifier** - Gradient-based ensemble method
5. **Stacking Classifier** - Meta-learning approach combining multiple algorithms

### Two-Phase Classification Strategy:

#### Phase 1: Binary Classification (`model.ipynb`)
- **Target**: Distinguish between planets (CONFIRMED) vs non-planets (CANDIDATE)
- **Focus**: High recall for confirmed exoplanets
- **Application**: Initial screening of candidates

#### Phase 2: False Positive Detection (`model2.ipynb`)
- **Target**: Identify false positives among planet candidates
- **Innovation**: Incorporates false positive samples in training to improve discrimination
- **Application**: Refined classification to reduce false discovery rate

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
Jupyter Notebook
Required libraries (see requirements.txt)
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/matiszpek/Exoplanet-detection-using-AI.git
cd Exoplanet-detection-using-AI
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Launch Jupyter Notebook**
```bash
jupyter notebook
```

## 📁 Project Structure

```
├── data/                          # Dataset files
│   ├── cumulative_2025.10.04_05.21.55.csv
│   ├── k2pandc_2025.10.04_05.29.43.csv
│   └── TOI_2025.10.04_05.29.26.csv
├── notebooks/                     # Jupyter notebooks
│   ├── 01_exoplanet_detection_binary.ipynb      # Binary classification model
│   └── 02_false_positive_detection.ipynb        # False positive detection
├── models/                        # Trained models and artifacts
│   ├── production/               # Final production models
│   ├── experiments/             # Experimental models
│   └── archived/               # Previous model versions
├── src/                          # Source code modules
│   ├── data_preprocessing.py    # Data cleaning and preprocessing
│   ├── feature_engineering.py  # Feature extraction and selection
│   ├── model_training.py       # Model training utilities
│   └── evaluation.py           # Model evaluation metrics
├── docs/                        # Documentation
├── features.json               # Feature list for model inference
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## 📓 Notebooks Overview

### 1. Binary Classification (`01_exoplanet_detection_binary.ipynb`)
- **Data exploration** and visualization
- **Feature engineering** and selection
- **Model training** with multiple algorithms
- **Performance evaluation** and comparison
- **Model persistence** for deployment
- **Interactive plots** and data analysis

### 2. False Positive Detection (`02_false_positive_detection.ipynb`)
- **Advanced classification** incorporating false positives
- **Stratified evaluation** by object type
- **Threshold optimization** for operational deployment
- **Detailed performance** analysis by subgroup
- **Interactive visualizations** and model interpretation

## 🎯 Results

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Stacking Classifier | 94.2% | 93.8% | 94.6% | 94.2% | 0.987 |
| Random Forest | 93.7% | 93.1% | 94.3% | 93.7% | 0.985 |
| Extra Trees | 93.5% | 92.9% | 94.1% | 93.5% | 0.984 |
| AdaBoost | 92.8% | 92.2% | 93.4% | 92.8% | 0.981 |

### Key Achievements
- ✅ **High Accuracy**: >94% classification accuracy on test set
- ✅ **Low False Positive Rate**: <6% false positive rate for confirmed planets
- ✅ **Robust Performance**: Consistent results across different stellar types
- ✅ **Feature Insights**: Identified key physical parameters for exoplanet detection

## 🔍 Key Features for Detection

Our analysis identified the most important features for exoplanet classification:

1. **Transit Depth** (`koi_depth`) - Signal strength indicator
2. **Orbital Period** (`koi_period`) - Planetary orbital characteristics  
3. **Planetary Radius** (`koi_prad`) - Physical size of the candidate
4. **Stellar Radius** (`koi_srad`) - Host star characteristics
5. **Signal-to-Noise Ratio** (`koi_model_snr`) - Data quality indicator

## � Feature Management

Our system uses **36 carefully selected astronomical features** for exoplanet detection. The `features.json` and enhanced `features_metadata.json` files contain the exact feature definitions required by trained models.

### Feature Categories:

- **Orbital Parameters** (9 features): Period, epoch, impact parameter, duration + uncertainties
- **Transit Characteristics** (7 features): Depth, signal-to-noise ratio, duration + uncertainties  
- **Planetary Properties** (7 features): Radius, temperature, insolation + uncertainties
- **Stellar Properties** (10 features): Temperature, surface gravity, radius, magnitude + uncertainties
- **Positional Data** (2 features): Right ascension, declination
- **Catalog Info** (1 feature): Planet number in system

### Feature Validation:
```python
from src.feature_management import load_production_features

# Load feature manager
feature_manager = load_production_features("models/production")

# Validate input data
feature_manager.validate_input_data(your_dataframe)

# Get required features
required_features = feature_manager.get_feature_columns()
print(f"Model requires {len(required_features)} features")
```

## �🚀 Usage Examples

### Loading a Trained Model
```python
import joblib
import pandas as pd
from src.feature_management import load_production_features

# Load feature definitions and model
feature_manager = load_production_features('models/production')
model = joblib.load('models/production/sub_models/stacking.pkl')

# Load and prepare new data
new_data = pd.read_csv('new_candidates.csv')

# Validate and prepare features
feature_manager.validate_input_data(new_data)
prepared_data = feature_manager.prepare_input_data(new_data)

# Make predictions
predictions = model.predict(prepared_data)
probabilities = model.predict_proba(prepared_data)
```

### Batch Processing
```python
from src.model_training import ExoplanetClassifier

# Initialize classifier
classifier = ExoplanetClassifier()

# Process multiple candidates
results = classifier.classify_candidates('data/new_observations.csv')
print(f"Found {results['confirmed'].sum()} confirmed exoplanets!")
```

## 🔬 Technical Implementation

### Data Preprocessing
- **Missing value imputation** using median values
- **Feature scaling** with StandardScaler
- **Outlier detection** and handling
- **Class balancing** for training stability

### Model Training
- **Stratified cross-validation** for robust evaluation
- **Hyperparameter optimization** using grid search
- **Ensemble methods** for improved generalization
- **Feature importance analysis** for model interpretation

### Evaluation Metrics
- **Standard classification metrics** (accuracy, precision, recall, F1)
- **ROC curves and AUC** for threshold analysis
- **Precision-recall curves** for imbalanced data
- **Confusion matrices** for detailed error analysis

## 🌟 Future Enhancements

- [ ] **Deep Learning Models**: Implement CNN/RNN for time-series analysis
- [ ] **Multi-class Classification**: Distinguish planet types (terrestrial, gas giant, etc.)
- [ ] **Uncertainty Quantification**: Add prediction confidence intervals
- [ ] **Real-time Processing**: Deploy for live TESS data analysis
- [ ] **Web Interface**: Create user-friendly prediction interface

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NASA** for providing the Kepler and TESS datasets
- **NASA Exoplanet Archive** for data access and documentation
- **Space Apps Challenge** organizers for this inspiring challenge
- **Open Source Community** for the amazing tools and libraries

## 📚 References

1. [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
2. [Kepler Mission](https://www.nasa.gov/mission_pages/kepler/main/index.html)
3. [TESS Mission](https://tess.mit.edu/)
4. [Transit Photometry Method](https://exoplanets.nasa.gov/alien-worlds/ways-to-find-a-planet/)

## 📧 Contact

**Team Members:**
- Matias Ariel Szpektor - [@matiszpek](https://github.com/matiszpek)
- Matias Mayans Kohon - [@matimay](https://github.com/matimay)
- Nicolas Campanario - [@nico-campa123](https://github.com/nico-campa123)

**Project Link:** [https://github.com/matiszpek/Exoplanet-detection-using-AI](https://github.com/matiszpek/Exoplanet-detection-using-AI)

---

