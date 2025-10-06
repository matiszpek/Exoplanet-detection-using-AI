"""
Feature engineering utilities for exoplanet detection.

This module provides functions for creating, selecting, and transforming
features for machine learning models in exoplanet detection.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExoplanetFeatureEngineer:
    """
    Feature engineering class for exoplanet detection data.
    
    This class provides methods for creating new features, selecting
    important features, and transforming existing features to improve
    model performance.
    """
    
    def __init__(self):
        self.feature_selector = None
        self.pca = None
        self.poly_features = None
        self.selected_features = None
        
    def create_derived_features(self, df):
        """
        Create derived features from existing astronomical parameters.
        
        Args:
            df (pd.DataFrame): Input dataframe with basic features
            
        Returns:
            pd.DataFrame: Dataframe with additional derived features
        """
        df_enhanced = df.copy()
        
        # Planet-to-star radius ratio (if both exist)
        if 'koi_prad' in df.columns and 'koi_srad' in df.columns:
            df_enhanced['planet_star_radius_ratio'] = (
                df['koi_prad'] / df['koi_srad']
            ).replace([np.inf, -np.inf], np.nan)
        
        # Transit duration to period ratio
        if 'koi_duration' in df.columns and 'koi_period' in df.columns:
            df_enhanced['duration_period_ratio'] = (
                df['koi_duration'] / df['koi_period']
            ).replace([np.inf, -np.inf], np.nan)
        
        # Planet density estimate (requires radius and period)
        if all(col in df.columns for col in ['koi_prad', 'koi_period', 'koi_srad']):
            # Simplified density calculation
            df_enhanced['planet_density_estimate'] = (
                df['koi_prad']**3 / df['koi_period']**2
            ).replace([np.inf, -np.inf], np.nan)
        
        # Equilibrium temperature gradient
        if 'koi_teq' in df.columns and 'koi_steff' in df.columns:
            df_enhanced['temp_gradient'] = (
                df['koi_steff'] - df['koi_teq']
            )
        
        # Orbital distance estimate (using Kepler's 3rd law approximation)
        if 'koi_period' in df.columns and 'koi_srad' in df.columns:
            df_enhanced['orbital_distance_estimate'] = (
                (df['koi_period'] / 365.25)**(2/3) * df['koi_srad']
            ).replace([np.inf, -np.inf], np.nan)
        
        # Transit depth to radius ratio consistency check
        if 'koi_depth' in df.columns and 'koi_prad' in df.columns and 'koi_srad' in df.columns:
            expected_depth = (df['koi_prad'] / df['koi_srad'])**2 * 1e6  # in ppm
            df_enhanced['depth_radius_consistency'] = (
                np.abs(df['koi_depth'] - expected_depth) / expected_depth
            ).replace([np.inf, -np.inf], np.nan)
        
        # Signal strength indicator
        if 'koi_depth' in df.columns and 'koi_model_snr' in df.columns:
            df_enhanced['signal_strength'] = (
                df['koi_depth'] * df['koi_model_snr']
            ).replace([np.inf, -np.inf], np.nan)
        
        # Error ratios (measurement uncertainty indicators)
        error_columns = [col for col in df.columns if col.endswith('_err1')]
        for err_col in error_columns:
            base_col = err_col.replace('_err1', '')
            if base_col in df.columns:
                df_enhanced[f'{base_col}_error_ratio'] = (
                    df[err_col] / np.abs(df[base_col])
                ).replace([np.inf, -np.inf], np.nan)
        
        logger.info(f"Created {len(df_enhanced.columns) - len(df.columns)} derived features")
        return df_enhanced
    
    def select_features(self, X, y, method='mutual_info', k=20):
        """
        Select the most informative features for classification.
        
        Args:
            X (array-like): Feature matrix
            y (array-like): Target vector
            method (str): Selection method ('mutual_info', 'f_classif', 'variance')
            k (int): Number of features to select
            
        Returns:
            array: Selected features
        """
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        elif method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=k)
        else:
            raise ValueError(f"Unsupported selection method: {method}")
        
        X_selected = selector.fit_transform(X, y)
        self.feature_selector = selector
        self.selected_features = selector.get_support()
        
        logger.info(f"Selected {k} features using {method}")
        return X_selected
    
    def apply_pca(self, X, n_components=0.95, fit=True):
        """
        Apply Principal Component Analysis for dimensionality reduction.
        
        Args:
            X (array-like): Feature matrix
            n_components: Number of components or variance ratio to retain
            fit (bool): Whether to fit the PCA or use existing fit
            
        Returns:
            array: Transformed features
        """
        if fit or self.pca is None:
            self.pca = PCA(n_components=n_components, random_state=42)
            X_pca = self.pca.fit_transform(X)
        else:
            X_pca = self.pca.transform(X)
        
        explained_variance = np.sum(self.pca.explained_variance_ratio_)
        logger.info(f"PCA reduced to {X_pca.shape[1]} components "
                   f"explaining {explained_variance:.3f} of variance")
        
        return X_pca
    
    def create_polynomial_features(self, X, degree=2, interaction_only=True):
        """
        Create polynomial and interaction features.
        
        Args:
            X (array-like): Feature matrix
            degree (int): Polynomial degree
            interaction_only (bool): Only create interaction terms
            
        Returns:
            array: Enhanced feature matrix
        """
        if self.poly_features is None:
            self.poly_features = PolynomialFeatures(
                degree=degree, 
                interaction_only=interaction_only,
                include_bias=False
            )
            X_poly = self.poly_features.fit_transform(X)
        else:
            X_poly = self.poly_features.transform(X)
        
        logger.info(f"Created polynomial features: {X.shape[1]} -> {X_poly.shape[1]}")
        return X_poly
    
    def get_feature_importance_ranking(self, feature_names, importances, top_n=20):
        """
        Create a ranking of feature importance.
        
        Args:
            feature_names (list): List of feature names
            importances (array): Feature importance scores
            top_n (int): Number of top features to return
            
        Returns:
            pd.DataFrame: Ranked features with importance scores
        """
        feature_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return feature_df.head(top_n)
    
    def analyze_feature_distributions(self, df, target_column, features_to_analyze=None):
        """
        Analyze feature distributions by target class.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_column (str): Name of target column
            features_to_analyze (list): Features to analyze (all numeric if None)
            
        Returns:
            dict: Statistical summaries by class
        """
        if features_to_analyze is None:
            features_to_analyze = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_column in features_to_analyze:
                features_to_analyze.remove(target_column)
        
        analysis = {}
        target_classes = df[target_column].unique()
        
        for class_val in target_classes:
            class_data = df[df[target_column] == class_val]
            analysis[class_val] = {}
            
            for feature in features_to_analyze:
                if feature in class_data.columns:
                    analysis[class_val][feature] = {
                        'mean': class_data[feature].mean(),
                        'std': class_data[feature].std(),
                        'median': class_data[feature].median(),
                        'min': class_data[feature].min(),
                        'max': class_data[feature].max(),
                        'count': class_data[feature].count()
                    }
        
        logger.info(f"Analyzed {len(features_to_analyze)} features across {len(target_classes)} classes")
        return analysis
    
    def detect_correlated_features(self, df, threshold=0.95):
        """
        Detect highly correlated features for potential removal.
        
        Args:
            df (pd.DataFrame): Feature dataframe
            threshold (float): Correlation threshold
            
        Returns:
            list: Pairs of highly correlated features
        """
        # Calculate correlation matrix
        corr_matrix = df.corr().abs()
        
        # Find highly correlated pairs
        highly_correlated = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > threshold:
                    highly_correlated.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))
        
        logger.info(f"Found {len(highly_correlated)} highly correlated feature pairs")
        return highly_correlated
    
    def create_binned_features(self, df, feature_column, n_bins=5, strategy='quantile'):
        """
        Create binned categorical features from continuous variables.
        
        Args:
            df (pd.DataFrame): Input dataframe
            feature_column (str): Column to bin
            n_bins (int): Number of bins
            strategy (str): Binning strategy ('uniform', 'quantile')
            
        Returns:
            pd.Series: Binned feature
        """
        if strategy == 'quantile':
            binned = pd.qcut(df[feature_column], q=n_bins, labels=False, duplicates='drop')
        elif strategy == 'uniform':
            binned = pd.cut(df[feature_column], bins=n_bins, labels=False)
        else:
            raise ValueError(f"Unsupported binning strategy: {strategy}")
        
        return binned
    
    def get_feature_statistics(self, X, feature_names=None):
        """
        Get comprehensive statistics for all features.
        
        Args:
            X (array-like): Feature matrix
            feature_names (list): Feature names
            
        Returns:
            pd.DataFrame: Feature statistics
        """
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        stats_df = pd.DataFrame({
            'feature': feature_names,
            'mean': np.mean(X, axis=0),
            'std': np.std(X, axis=0),
            'min': np.min(X, axis=0),
            'max': np.max(X, axis=0),
            'median': np.median(X, axis=0),
            'missing_rate': np.isnan(X).mean(axis=0) if np.any(np.isnan(X)) else 0
        })
        
        return stats_df
    
    def save_feature_engineering_artifacts(self, filepath_prefix):
        """
        Save feature engineering artifacts for reproducibility.
        
        Args:
            filepath_prefix (str): Prefix for saved files
        """
        import joblib
        
        if self.feature_selector:
            joblib.dump(self.feature_selector, f"{filepath_prefix}_feature_selector.pkl")
        
        if self.pca:
            joblib.dump(self.pca, f"{filepath_prefix}_pca.pkl")
        
        if self.poly_features:
            joblib.dump(self.poly_features, f"{filepath_prefix}_poly_features.pkl")
        
        logger.info(f"Saved feature engineering artifacts with prefix: {filepath_prefix}")


# Predefined feature groups for exoplanet data
EXOPLANET_FEATURE_GROUPS = {
    'orbital_parameters': [
        'koi_period', 'koi_period_err1', 'koi_period_err2',
        'koi_time0bk', 'koi_time0bk_err1', 'koi_time0bk_err2',
        'koi_impact', 'koi_impact_err1', 'koi_impact_err2'
    ],
    'transit_characteristics': [
        'koi_duration', 'koi_duration_err1', 'koi_duration_err2',
        'koi_depth', 'koi_depth_err1', 'koi_depth_err2',
        'koi_model_snr'
    ],
    'planetary_properties': [
        'koi_prad', 'koi_prad_err1', 'koi_prad_err2',
        'koi_teq', 'koi_insol', 'koi_insol_err1', 'koi_insol_err2'
    ],
    'stellar_properties': [
        'koi_steff', 'koi_steff_err1', 'koi_steff_err2',
        'koi_slogg', 'koi_slogg_err1', 'koi_slogg_err2',
        'koi_srad', 'koi_srad_err1', 'koi_srad_err2',
        'koi_kepmag'
    ],
    'positional_data': ['ra', 'dec'],
    'catalog_info': ['koi_tce_plnt_num']
}