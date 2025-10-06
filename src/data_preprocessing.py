"""
Data preprocessing utilities for exoplanet detection.

This module provides functions for cleaning, preprocessing, and preparing
astronomical data for machine learning models.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExoplanetDataPreprocessor:
    """
    A comprehensive data preprocessor for exoplanet detection datasets.
    
    This class handles missing values, outliers, feature scaling, and
    data validation for astronomical datasets.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_columns = None
        self.non_numeric_columns = None
        
    def load_data(self, filepath, comment='#'):
        """
        Load astronomical data from CSV file.
        
        Args:
            filepath (str): Path to the CSV file
            comment (str): Comment character to ignore metadata lines
            
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        try:
            df = pd.read_csv(filepath, comment=comment, engine='python')
            logger.info(f"Successfully loaded data with shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def remove_irrelevant_columns(self, df, columns_to_drop=None):
        """
        Remove irrelevant or empty columns from the dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe
            columns_to_drop (list): List of column names to remove
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        if columns_to_drop is None:
            columns_to_drop = [
                'rowid', 'kepid', 'kepoi_name', 'kepler_name', 
                'koi_pdisposition', 'koi_score', 'koi_teq_err1', 
                'koi_teq_err2', 'koi_tce_delivname', 'koi_fpflag_nt', 
                'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec'
            ]
        
        # Only drop columns that exist in the dataframe
        existing_cols = [col for col in columns_to_drop if col in df.columns]
        df_cleaned = df.drop(columns=existing_cols)
        
        logger.info(f"Removed {len(existing_cols)} irrelevant columns")
        return df_cleaned
    
    def handle_missing_values(self, df):
        """
        Handle missing values in the dataset using median imputation.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with imputed values
        """
        # Separate numeric and non-numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
        
        self.non_numeric_columns = non_numeric_cols.tolist()
        
        # Impute missing values in numeric columns
        df_imputed = df.copy()
        if len(numeric_cols) > 0:
            df_imputed[numeric_cols] = df_imputed[numeric_cols].fillna(
                df_imputed[numeric_cols].median()
            )
        
        logger.info(f"Imputed missing values in {len(numeric_cols)} numeric columns")
        return df_imputed
    
    def prepare_features_target(self, df, target_column='koi_disposition', 
                              target_mapping=None, binary_classification=True):
        """
        Prepare features and target variables for machine learning.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_column (str): Name of the target column
            target_mapping (dict): Mapping for target values
            binary_classification (bool): Whether to use binary classification
            
        Returns:
            tuple: (X, y) features and target arrays
        """
        # Default target mapping for binary classification
        if target_mapping is None and binary_classification:
            target_mapping = {'CONFIRMED': 1, 'CANDIDATE': 0}
        
        # Filter data based on target values
        if binary_classification:
            df_filtered = df[df[target_column].isin(target_mapping.keys())]
        else:
            df_filtered = df.copy()
        
        # Prepare features
        X = df_filtered.drop(columns=[target_column])
        
        # Remove non-numeric columns from features
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]
        self.feature_columns = X.columns.tolist()
        
        # Prepare target
        if binary_classification:
            y = df_filtered[target_column].map(target_mapping)
        else:
            y = df_filtered[target_column]
        
        logger.info(f"Prepared features with shape: {X.shape}")
        logger.info(f"Target distribution: {pd.Series(y).value_counts().to_dict()}")
        
        return X, y
    
    def scale_features(self, X_train, X_test=None):
        """
        Scale features using StandardScaler.
        
        Args:
            X_train (array-like): Training features
            X_test (array-like): Test features (optional)
            
        Returns:
            tuple or array: Scaled features
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def detect_outliers(self, X, threshold=3):
        """
        Detect outliers using Z-score method.
        
        Args:
            X (array-like): Feature matrix
            threshold (float): Z-score threshold for outlier detection
            
        Returns:
            np.array: Boolean mask of outliers
        """
        z_scores = np.abs((X - np.mean(X, axis=0)) / np.std(X, axis=0))
        outliers = np.any(z_scores > threshold, axis=1)
        
        logger.info(f"Detected {np.sum(outliers)} outliers ({np.mean(outliers)*100:.1f}%)")
        return outliers
    
    def get_feature_names(self):
        """
        Get the list of feature column names.
        
        Returns:
            list: Feature column names
        """
        return self.feature_columns
    
    def save_preprocessing_artifacts(self, filepath_prefix):
        """
        Save preprocessing artifacts for later use.
        
        Args:
            filepath_prefix (str): Prefix for saved files
        """
        import joblib
        import json
        
        # Save scaler
        joblib.dump(self.scaler, f"{filepath_prefix}_scaler.pkl")
        
        # Save feature names
        if self.feature_columns:
            with open(f"{filepath_prefix}_features.json", 'w') as f:
                json.dump(self.feature_columns, f)
        
        logger.info(f"Saved preprocessing artifacts with prefix: {filepath_prefix}")


def create_false_positive_dataset(X_train, y_train, X_test, y_test, 
                                 df_original, split_ratio=0.5, random_state=42):
    """
    Create enhanced dataset including false positive samples.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data  
        df_original: Original dataframe with all classes
        split_ratio: Ratio to split false positives between train/test
        random_state: Random seed for reproducibility
        
    Returns:
        tuple: Enhanced (X_train, y_train, X_test, y_test)
    """
    # Extract false positive samples
    fp_data = df_original[df_original['koi_disposition'] == 'FALSE POSITIVE']
    
    # Get feature columns (excluding target and non-numeric)
    non_numeric_cols = df_original.select_dtypes(exclude=[np.number]).columns
    non_numeric_cols = [col for col in non_numeric_cols if col != 'koi_disposition']
    
    X_fp = fp_data.drop(columns=['koi_disposition'] + non_numeric_cols.tolist())
    
    # Create labels (false positives are class 0 - not planets)
    y_fp = np.zeros(len(X_fp), dtype=int)
    
    # Split false positives
    from sklearn.model_selection import train_test_split
    X_fp_train, X_fp_test, y_fp_train, y_fp_test = train_test_split(
        X_fp, y_fp, test_size=(1-split_ratio), random_state=random_state
    )
    
    # Combine with original data
    X_train_enhanced = np.vstack([X_train, X_fp_train])
    y_train_enhanced = np.concatenate([y_train, y_fp_train])
    X_test_enhanced = np.vstack([X_test, X_fp_test])
    y_test_enhanced = np.concatenate([y_test, y_fp_test])
    
    logger.info(f"Added {len(X_fp)} false positive samples")
    logger.info(f"Enhanced training set: {X_train_enhanced.shape}")
    logger.info(f"Enhanced test set: {X_test_enhanced.shape}")
    
    return X_train_enhanced, y_train_enhanced, X_test_enhanced, y_test_enhanced