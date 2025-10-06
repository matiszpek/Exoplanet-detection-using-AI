"""
Model training utilities for exoplanet detection.

This module provides classes and functions for training, evaluating,
and managing machine learning models for exoplanet classification.
"""

import numpy as np
import pandas as pd
import joblib
import json
import time
from pathlib import Path
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier,
    GradientBoostingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, precision_recall_curve
)
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExoplanetClassifier:
    """
    A comprehensive exoplanet classification system.
    
    This class provides functionality for training multiple machine learning
    models, evaluating their performance, and managing model artifacts.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.feature_names = None
        
    def build_models(self):
        """
        Build a collection of machine learning models for exoplanet detection.
        
        Returns:
            dict: Dictionary of model instances
        """
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=1600,
                criterion='entropy',
                random_state=self.random_state,
                n_jobs=-1
            ),
            'ExtraTrees': ExtraTreesClassifier(
                n_estimators=200,
                criterion='entropy',
                random_state=self.random_state,
                n_jobs=-1
            ),
            'AdaBoost': AdaBoostClassifier(
                n_estimators=974,
                learning_rate=0.1,
                random_state=self.random_state
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=1600,
                learning_rate=0.1,
                random_state=self.random_state
            )
        }
        
        # Create stacking classifier
        meta_learner = LogisticRegression(max_iter=1000, random_state=self.random_state)
        models['Stacking'] = StackingClassifier(
            estimators=[
                ('rf', models['RandomForest']),
                ('gb', models['GradientBoosting'])
            ],
            final_estimator=meta_learner,
            cv=5,
            n_jobs=-1,
            passthrough=False
        )
        
        self.models = models
        logger.info(f"Built {len(models)} models for training")
        return models
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """
        Train all models and evaluate their performance.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            
        Returns:
            pd.DataFrame: Results summary
        """
        if not self.models:
            self.build_models()
        
        results_list = []
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            # Train the model
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_proba = self._safe_predict_proba(model, X_test)
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred, y_proba)
            metrics['training_time'] = training_time
            metrics['model'] = name
            
            # Cross-validation score
            cv_scores = cross_val_score(
                model, X_train, y_train, cv=5, scoring='accuracy'
            )
            metrics['cv_mean'] = np.mean(cv_scores)
            metrics['cv_std'] = np.std(cv_scores)
            
            results_list.append(metrics)
            logger.info(f"{name} - Accuracy: {metrics['accuracy']:.4f}, "
                       f"F1: {metrics['f1_score']:.4f}")
        
        self.results = pd.DataFrame(results_list)
        return self.results
    
    def _safe_predict_proba(self, model, X):
        """
        Safely get prediction probabilities from a model.
        
        Args:
            model: Trained model
            X: Features
            
        Returns:
            np.array: Prediction probabilities
        """
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)
            if proba.shape[1] == 2:
                return proba[:, 1]  # Return positive class probabilities
            else:
                return proba.ravel()
        elif hasattr(model, 'decision_function'):
            scores = model.decision_function(X)
            # Normalize to [0, 1] range
            return (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        else:
            return model.predict(X).astype(float)
    
    def _calculate_metrics(self, y_true, y_pred, y_proba):
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities
            
        Returns:
            dict: Dictionary of metrics
        """
        try:
            roc_auc = roc_auc_score(y_true, y_proba)
        except:
            roc_auc = np.nan
            
        try:
            pr_auc = average_precision_score(y_true, y_proba)
        except:
            pr_auc = np.nan
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
    
    def select_best_model(self, metric='f1_score'):
        """
        Select the best performing model based on a metric.
        
        Args:
            metric (str): Metric to use for selection
            
        Returns:
            tuple: (best_model_name, best_model)
        """
        if self.results is None or self.results.empty:
            raise ValueError("No results available. Train models first.")
        
        best_idx = self.results[metric].idxmax()
        best_model_name = self.results.loc[best_idx, 'model']
        self.best_model = self.models[best_model_name]
        
        logger.info(f"Best model: {best_model_name} "
                   f"({metric}={self.results.loc[best_idx, metric]:.4f})")
        
        return best_model_name, self.best_model
    
    def save_models(self, output_dir, feature_names=None):
        """
        Save trained models and metadata.
        
        Args:
            output_dir (str): Directory to save models
            feature_names (list): List of feature names
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Save individual models
        for name, model in self.models.items():
            model_path = output_path / f"{name.lower()}_model.pkl"
            joblib.dump(model, model_path, compress=3)
            logger.info(f"Saved {name} to {model_path}")
        
        # Save results
        if self.results is not None:
            results_path = output_path / "training_results.csv"
            self.results.to_csv(results_path, index=False)
        
        # Save feature names
        if feature_names:
            features_path = output_path / "feature_names.json"
            with open(features_path, 'w') as f:
                json.dump(feature_names, f)
            self.feature_names = feature_names
        
        # Save metadata
        metadata = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'models_trained': list(self.models.keys()),
            'best_model': getattr(self, 'best_model_name', None),
            'random_state': self.random_state,
            'feature_count': len(feature_names) if feature_names else None
        }
        
        meta_path = output_path / "metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved all artifacts to {output_path}")
    
    def load_model(self, model_path):
        """
        Load a trained model from file.
        
        Args:
            model_path (str): Path to the model file
            
        Returns:
            object: Loaded model
        """
        model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")
        return model
    
    def get_feature_importance(self, model_name=None, top_n=20):
        """
        Get feature importance from tree-based models.
        
        Args:
            model_name (str): Name of the model (uses best if None)
            top_n (int): Number of top features to return
            
        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        if model_name is None:
            if self.best_model is None:
                raise ValueError("No best model selected")
            model = self.best_model
            model_name = "best_model"
        else:
            model = self.models.get(model_name)
            if model is None:
                raise ValueError(f"Model {model_name} not found")
        
        # Handle stacking classifier
        if isinstance(model, StackingClassifier):
            importances_dict = {}
            for name, estimator in model.named_estimators_.items():
                if hasattr(estimator, 'feature_importances_'):
                    importances_dict[name] = estimator.feature_importances_
            
            if importances_dict:
                # Average importance across estimators
                avg_importance = np.mean(list(importances_dict.values()), axis=0)
                feature_names = self.feature_names or [f"feature_{i}" for i in range(len(avg_importance))]
                
                df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': avg_importance
                }).sort_values('importance', ascending=False).head(top_n)
                
                return df
        
        # Handle single models with feature importance
        elif hasattr(model, 'feature_importances_'):
            feature_names = self.feature_names or [f"feature_{i}" for i in range(len(model.feature_importances_))]
            
            df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(top_n)
            
            return df
        
        logger.warning(f"Model {model_name} does not have feature importance")
        return None
    
    def classify_candidates(self, data_path, model_path=None, threshold=0.5):
        """
        Classify new exoplanet candidates.
        
        Args:
            data_path (str): Path to candidate data
            model_path (str): Path to trained model (uses best if None)
            threshold (float): Classification threshold
            
        Returns:
            pd.DataFrame: Classification results
        """
        # Load data
        data = pd.read_csv(data_path)
        
        # Load model
        if model_path:
            model = self.load_model(model_path)
        elif self.best_model:
            model = self.best_model
        else:
            raise ValueError("No model available for classification")
        
        # Prepare features (assuming preprocessing already done)
        X = data[self.feature_names] if self.feature_names else data
        
        # Make predictions
        probabilities = self._safe_predict_proba(model, X)
        predictions = (probabilities >= threshold).astype(int)
        
        # Create results dataframe
        results = data.copy()
        results['predicted_class'] = predictions
        results['probability'] = probabilities
        results['confidence'] = np.abs(probabilities - 0.5) * 2  # Distance from decision boundary
        
        logger.info(f"Classified {len(results)} candidates")
        logger.info(f"Predicted exoplanets: {np.sum(predictions)}")
        
        return results


def optimize_threshold(y_true, y_proba, metric='f1'):
    """
    Optimize classification threshold based on a metric.
    
    Args:
        y_true: True labels
        y_proba: Prediction probabilities
        metric (str): Metric to optimize ('f1', 'precision', 'recall')
        
    Returns:
        tuple: (optimal_threshold, best_score)
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    scores = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        scores.append(score)
    
    best_idx = np.argmax(scores)
    optimal_threshold = thresholds[best_idx]
    best_score = scores[best_idx]
    
    logger.info(f"Optimal threshold for {metric}: {optimal_threshold:.3f} "
               f"(score: {best_score:.4f})")
    
    return optimal_threshold, best_score