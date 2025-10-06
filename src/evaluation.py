"""
Evaluation utilities for exoplanet detection models.

This module provides comprehensive evaluation metrics and visualization
tools for assessing machine learning model performance in exoplanet detection.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
    matthews_corrcoef
)
from sklearn.calibration import calibration_curve
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExoplanetModelEvaluator:
    """
    Comprehensive evaluation class for exoplanet detection models.
    
    This class provides methods for calculating metrics, creating visualizations,
    and performing detailed analysis of model performance.
    """
    
    def __init__(self):
        self.evaluation_results = {}
        
    def calculate_comprehensive_metrics(self, y_true, y_pred, y_proba=None, 
                                      average='binary', pos_label=1):
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities
            average: Averaging method for multiclass
            pos_label: Positive class label
            
        Returns:
            dict: Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average, 
                                       pos_label=pos_label, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average, 
                                 pos_label=pos_label, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average=average, 
                               pos_label=pos_label, zero_division=0),
            'matthews_corrcoef': matthews_corrcoef(y_true, y_pred)
        }
        
        # Add probability-based metrics if available
        if y_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            except:
                metrics['roc_auc'] = np.nan
                
            try:
                metrics['pr_auc'] = average_precision_score(y_true, y_proba)
            except:
                metrics['pr_auc'] = np.nan
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Derived metrics from confusion matrix
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics.update({
                'true_positives': int(tp),
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else np.nan,
                'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else np.nan,
                'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else np.nan,
                'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else np.nan
            })
        
        return metrics
    
    def evaluate_stratified_performance(self, y_true, y_pred, y_proba, 
                                      subgroup_labels, threshold=0.5):
        """
        Evaluate model performance across different subgroups.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels (or None to use threshold)
            y_proba: Prediction probabilities
            subgroup_labels: Array of subgroup identifiers
            threshold: Classification threshold
            
        Returns:
            pd.DataFrame: Stratified performance metrics
        """
        if y_pred is None:
            y_pred = (y_proba >= threshold).astype(int)
        
        results = []
        subgroups = np.unique(subgroup_labels)
        
        for subgroup in subgroups:
            mask = subgroup_labels == subgroup
            if not mask.any():
                continue
                
            y_true_sub = y_true[mask]
            y_pred_sub = y_pred[mask]
            y_proba_sub = y_proba[mask] if y_proba is not None else None
            
            # Calculate metrics for this subgroup
            metrics = self.calculate_comprehensive_metrics(
                y_true_sub, y_pred_sub, y_proba_sub
            )
            
            # Add subgroup information
            metrics['subgroup'] = subgroup
            metrics['sample_size'] = int(mask.sum())
            metrics['positive_rate'] = float(np.mean(y_true_sub))
            
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, 
                             normalize=False, figsize=(8, 6)):
        """
        Plot confusion matrix with customization options.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names for classes
            normalize: Whether to normalize values
            figsize: Figure size
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(cm.shape[0])]
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=ax)
        
        ax.set_title(title)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        plt.tight_layout()
        return fig
    
    def plot_roc_curve(self, y_true, y_proba, title='ROC Curve', figsize=(8, 6)):
        """
        Plot ROC curve with AUC score.
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            title: Plot title
            figsize: Figure size
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        auc_score = roc_auc_score(y_true, y_proba)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc_score:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_precision_recall_curve(self, y_true, y_proba, title='Precision-Recall Curve', 
                                   figsize=(8, 6)):
        """
        Plot precision-recall curve with AP score.
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            title: Plot title
            figsize: Figure size
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        ap_score = average_precision_score(y_true, y_proba)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(recall, precision, linewidth=2, label=f'PR (AP = {ap_score:.3f})')
        
        # Baseline (random classifier)
        baseline = np.mean(y_true)
        ax.axhline(y=baseline, color='k', linestyle='--', linewidth=1, 
                  label=f'Random Classifier (AP = {baseline:.3f})')
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_threshold_analysis(self, y_true, y_proba, figsize=(12, 8)):
        """
        Plot analysis of different classification thresholds.
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            figsize: Figure size
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        thresholds = np.linspace(0.01, 0.99, 99)
        precisions, recalls, f1_scores, accuracies = [], [], [], []
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            precisions.append(precision_score(y_true, y_pred, zero_division=0))
            recalls.append(recall_score(y_true, y_pred, zero_division=0))
            f1_scores.append(f1_score(y_true, y_pred, zero_division=0))
            accuracies.append(accuracy_score(y_true, y_pred))
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot each metric
        metrics = [
            (precisions, 'Precision', axes[0, 0]),
            (recalls, 'Recall', axes[0, 1]),
            (f1_scores, 'F1 Score', axes[1, 0]),
            (accuracies, 'Accuracy', axes[1, 1])
        ]
        
        for values, name, ax in metrics:
            ax.plot(thresholds, values, linewidth=2)
            ax.set_xlabel('Threshold')
            ax.set_ylabel(name)
            ax.set_title(f'{name} vs Threshold')
            ax.grid(True, alpha=0.3)
            
            # Mark best threshold
            best_idx = np.argmax(values)
            best_threshold = thresholds[best_idx]
            best_value = values[best_idx]
            ax.axvline(x=best_threshold, color='red', linestyle='--', alpha=0.7)
            ax.text(best_threshold, best_value, 
                   f'Best: {best_threshold:.3f}', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        plt.tight_layout()
        return fig
    
    def plot_calibration_curve(self, y_true, y_proba, n_bins=10, 
                              title='Calibration Curve', figsize=(8, 6)):
        """
        Plot calibration curve to assess probability calibration.
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            n_bins: Number of bins for calibration
            title: Plot title
            figsize: Figure size
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fraction_pos, mean_pred_value = calibration_curve(
            y_true, y_proba, n_bins=n_bins
        )
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(mean_pred_value, fraction_pos, "s-", linewidth=2, 
               label='Model')
        ax.plot([0, 1], [0, 1], "k:", linewidth=2, label='Perfectly calibrated')
        
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_evaluation_report(self, y_true, y_pred, y_proba=None, 
                               class_names=None, model_name="Model"):
        """
        Create a comprehensive evaluation report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities
            class_names: Names for classes
            model_name: Name of the model
            
        Returns:
            dict: Comprehensive evaluation results
        """
        # Calculate metrics
        metrics = self.calculate_comprehensive_metrics(y_true, y_pred, y_proba)
        
        # Classification report
        report = classification_report(y_true, y_pred, 
                                     target_names=class_names, 
                                     output_dict=True)
        
        # Combine results
        evaluation = {
            'model_name': model_name,
            'metrics': metrics,
            'classification_report': report,
            'sample_size': len(y_true),
            'class_distribution': {
                str(k): int(v) for k, v in zip(*np.unique(y_true, return_counts=True))
            }
        }
        
        # Store for later use
        self.evaluation_results[model_name] = evaluation
        
        logger.info(f"Created evaluation report for {model_name}")
        return evaluation
    
    def compare_models(self, model_results):
        """
        Compare multiple models and create comparison visualizations.
        
        Args:
            model_results: Dictionary of model evaluation results
            
        Returns:
            pd.DataFrame: Comparison dataframe
        """
        comparison_data = []
        
        for model_name, results in model_results.items():
            metrics = results['metrics']
            row = {
                'Model': model_name,
                'Accuracy': metrics.get('accuracy', np.nan),
                'Precision': metrics.get('precision', np.nan),
                'Recall': metrics.get('recall', np.nan),
                'F1 Score': metrics.get('f1_score', np.nan),
                'ROC AUC': metrics.get('roc_auc', np.nan),
                'PR AUC': metrics.get('pr_auc', np.nan),
                'MCC': metrics.get('matthews_corrcoef', np.nan)
            }
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        return comparison_df
    
    def plot_model_comparison(self, comparison_df, figsize=(12, 8)):
        """
        Create visualization comparing multiple models.
        
        Args:
            comparison_df: DataFrame with model comparison data
            figsize: Figure size
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC', 'PR AUC']
        available_metrics = [m for m in metrics_to_plot if m in comparison_df.columns]
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.ravel()
        
        for i, metric in enumerate(available_metrics):
            ax = axes[i]
            comparison_df.plot(x='Model', y=metric, kind='bar', ax=ax, legend=False)
            ax.set_title(f'{metric} Comparison')
            ax.set_xlabel('Model')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(available_metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def analyze_prediction_confidence(self, y_proba, y_true, n_bins=10):
        """
        Analyze prediction confidence distribution.
        
        Args:
            y_proba: Prediction probabilities
            y_true: True labels
            n_bins: Number of confidence bins
            
        Returns:
            pd.DataFrame: Confidence analysis results
        """
        # Calculate confidence (distance from 0.5)
        confidence = np.abs(y_proba - 0.5) * 2
        
        # Create bins
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(confidence, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        # Analyze each bin
        results = []
        for i in range(n_bins):
            mask = bin_indices == i
            if not mask.any():
                continue
            
            bin_accuracy = accuracy_score(y_true[mask], 
                                        (y_proba[mask] >= 0.5).astype(int))
            
            results.append({
                'confidence_bin': f'{bins[i]:.2f}-{bins[i+1]:.2f}',
                'sample_count': int(mask.sum()),
                'accuracy': bin_accuracy,
                'avg_confidence': np.mean(confidence[mask])
            })
        
        return pd.DataFrame(results)


def calculate_detection_metrics(y_true, y_pred, y_proba=None):
    """
    Calculate exoplanet-specific detection metrics.
    
    Args:
        y_true: True labels (1 for exoplanet, 0 for non-exoplanet)
        y_pred: Predicted labels
        y_proba: Prediction probabilities
        
    Returns:
        dict: Exoplanet detection metrics
    """
    # Standard metrics
    detection_rate = recall_score(y_true, y_pred)  # How many real exoplanets we detect
    false_positive_rate = 1 - precision_score(y_true, y_pred)  # How many false alarms
    
    # Exoplanet-specific metrics
    total_detections = np.sum(y_pred)
    true_exoplanets = np.sum(y_true)
    
    metrics = {
        'detection_rate': detection_rate,
        'false_positive_rate': false_positive_rate,
        'total_detections': int(total_detections),
        'true_exoplanets_in_sample': int(true_exoplanets),
        'confirmed_detections': int(np.sum((y_true == 1) & (y_pred == 1))),
        'efficiency': detection_rate / max(1, total_detections / len(y_true))  # Detections per observation
    }
    
    if y_proba is not None:
        # Reliability at different confidence levels
        high_conf_mask = y_proba >= 0.8
        if high_conf_mask.any():
            metrics['high_confidence_precision'] = precision_score(
                y_true[high_conf_mask], y_pred[high_conf_mask]
            )
    
    return metrics