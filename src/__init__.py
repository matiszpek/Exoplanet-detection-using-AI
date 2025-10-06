"""
Exoplanet Detection using AI - Source Code Package

This package provides a comprehensive toolkit for exoplanet detection and classification
using machine learning techniques applied to astronomical data from NASA missions.

Modules:
    data_preprocessing: Data loading, cleaning, and preparation utilities
    feature_engineering: Feature creation, selection, and transformation tools
    model_training: Machine learning model training and management
    evaluation: Model evaluation metrics and visualization tools

Author: Matias Szpektor
Project: NASA Space Apps Challenge 2025 - "A World Away: Hunting for Exoplanets with AI"
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Matias Szpektor"
__email__ = "matirulosz@gmail.com"
__license__ = "MIT"

# Import main classes for easy access
from .data_preprocessing import ExoplanetDataPreprocessor, create_false_positive_dataset
from .feature_engineering import ExoplanetFeatureEngineer, EXOPLANET_FEATURE_GROUPS
from .model_training import ExoplanetClassifier, optimize_threshold
from .evaluation import ExoplanetModelEvaluator, calculate_detection_metrics
from .feature_management import FeatureManager, load_production_features, validate_model_input

__all__ = [
    'ExoplanetDataPreprocessor',
    'create_false_positive_dataset',
    'ExoplanetFeatureEngineer', 
    'EXOPLANET_FEATURE_GROUPS',
    'ExoplanetClassifier',
    'optimize_threshold',
    'ExoplanetModelEvaluator',
    'calculate_detection_metrics',
    'FeatureManager',
    'load_production_features',
    'validate_model_input'
]