"""
Feature management utilities for exoplanet detection models.

This module provides utilities for loading feature definitions, validating
input data, and ensuring consistency between training and inference.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)


class FeatureManager:
    """
    Manages feature definitions and validation for exoplanet detection models.
    
    This class ensures consistent feature handling between training and inference,
    validates input data, and provides feature metadata.
    """
    
    def __init__(self, metadata_path: Optional[str] = None):
        """
        Initialize FeatureManager with feature metadata.
        
        Args:
            metadata_path: Path to features metadata JSON file
        """
        self.metadata_path = metadata_path
        self.feature_metadata = None
        self.feature_columns = None
        
        if metadata_path and Path(metadata_path).exists():
            self.load_feature_metadata(metadata_path)
    
    def load_feature_metadata(self, metadata_path: str) -> Dict:
        """
        Load feature metadata from JSON file.
        
        Args:
            metadata_path: Path to metadata file
            
        Returns:
            dict: Feature metadata
        """
        try:
            with open(metadata_path, 'r') as f:
                self.feature_metadata = json.load(f)
            
            self.feature_columns = self.feature_metadata.get('feature_columns', [])
            logger.info(f"Loaded {len(self.feature_columns)} feature definitions")
            return self.feature_metadata
            
        except Exception as e:
            logger.error(f"Error loading feature metadata: {e}")
            raise
    
    def get_feature_columns(self) -> List[str]:
        """
        Get the list of required feature columns.
        
        Returns:
            list: Feature column names
        """
        if self.feature_columns is None:
            raise ValueError("Feature metadata not loaded. Call load_feature_metadata() first.")
        
        return self.feature_columns.copy()
    
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """
        Get feature groups (orbital, transit, stellar, etc.).
        
        Returns:
            dict: Feature groups with column lists
        """
        if self.feature_metadata is None:
            raise ValueError("Feature metadata not loaded.")
        
        return self.feature_metadata.get('feature_groups', {})
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """
        Get feature descriptions for documentation.
        
        Returns:
            dict: Feature descriptions
        """
        if self.feature_metadata is None:
            raise ValueError("Feature metadata not loaded.")
        
        return self.feature_metadata.get('feature_descriptions', {})
    
    def validate_input_data(self, data: Union[pd.DataFrame, np.ndarray], 
                           strict: bool = True) -> bool:
        """
        Validate input data against expected features.
        
        Args:
            data: Input data (DataFrame or array)
            strict: Whether to enforce strict column matching
            
        Returns:
            bool: True if validation passes
            
        Raises:
            ValueError: If validation fails
        """
        if self.feature_columns is None:
            raise ValueError("Feature metadata not loaded.")
        
        if isinstance(data, pd.DataFrame):
            return self._validate_dataframe(data, strict)
        elif isinstance(data, np.ndarray):
            return self._validate_array(data, strict)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def _validate_dataframe(self, df: pd.DataFrame, strict: bool) -> bool:
        """Validate DataFrame input."""
        expected_cols = set(self.feature_columns)
        actual_cols = set(df.columns)
        
        # Check for missing columns
        missing_cols = expected_cols - actual_cols
        if missing_cols:
            raise ValueError(f"Missing required columns: {sorted(missing_cols)}")
        
        # Check for extra columns (only in strict mode)
        if strict:
            extra_cols = actual_cols - expected_cols
            if extra_cols:
                logger.warning(f"Extra columns found (will be ignored): {sorted(extra_cols)}")
        
        # Check data types
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        non_numeric = set(self.feature_columns) - set(numeric_cols)
        if non_numeric:
            raise ValueError(f"Non-numeric columns found: {sorted(non_numeric)}")
        
        logger.info(f"DataFrame validation passed: {len(df)} rows, {len(expected_cols)} features")
        return True
    
    def _validate_array(self, arr: np.ndarray, strict: bool) -> bool:
        """Validate array input."""
        expected_features = len(self.feature_columns)
        
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got {arr.ndim}D")
        
        if arr.shape[1] != expected_features:
            raise ValueError(f"Expected {expected_features} features, got {arr.shape[1]}")
        
        # Check for non-numeric data
        if not np.issubdtype(arr.dtype, np.number):
            raise ValueError(f"Array must contain numeric data, got {arr.dtype}")
        
        logger.info(f"Array validation passed: {arr.shape}")
        return True
    
    def prepare_input_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare input data by selecting and ordering required features.
        
        Args:
            data: Input DataFrame
            
        Returns:
            pd.DataFrame: Prepared DataFrame with correct features
        """
        if self.feature_columns is None:
            raise ValueError("Feature metadata not loaded.")
        
        # Validate first
        self.validate_input_data(data, strict=False)
        
        # Select and order features correctly
        prepared_data = data[self.feature_columns].copy()
        
        logger.info(f"Prepared input data: {prepared_data.shape}")
        return prepared_data
    
    def get_missing_columns(self, data: pd.DataFrame) -> List[str]:
        """
        Get list of missing required columns.
        
        Args:
            data: Input DataFrame
            
        Returns:
            list: Missing column names
        """
        if self.feature_columns is None:
            raise ValueError("Feature metadata not loaded.")
        
        expected_cols = set(self.feature_columns)
        actual_cols = set(data.columns)
        
        return sorted(expected_cols - actual_cols)
    
    def create_feature_summary(self) -> pd.DataFrame:
        """
        Create a summary DataFrame of all features.
        
        Returns:
            pd.DataFrame: Feature summary with groups and descriptions
        """
        if self.feature_metadata is None:
            raise ValueError("Feature metadata not loaded.")
        
        feature_groups = self.get_feature_groups()
        feature_descriptions = self.get_feature_descriptions()
        
        # Create reverse mapping of feature to group
        feature_to_group = {}
        for group, features in feature_groups.items():
            for feature in features:
                feature_to_group[feature] = group
        
        # Build summary
        summary_data = []
        for feature in self.feature_columns:
            summary_data.append({
                'feature': feature,
                'group': feature_to_group.get(feature, 'unknown'),
                'description': feature_descriptions.get(feature, 'No description available')
            })
        
        return pd.DataFrame(summary_data)
    
    def export_feature_list(self, output_path: str, format: str = 'json') -> None:
        """
        Export feature list to file.
        
        Args:
            output_path: Output file path
            format: Export format ('json', 'csv', 'txt')
        """
        if self.feature_columns is None:
            raise ValueError("Feature metadata not loaded.")
        
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(self.feature_columns, f, indent=2)
        elif format == 'csv':
            pd.DataFrame({'feature': self.feature_columns}).to_csv(output_path, index=False)
        elif format == 'txt':
            with open(output_path, 'w') as f:
                f.write('\n'.join(self.feature_columns))
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Exported feature list to {output_path}")
    
    def compare_features(self, other_features: List[str]) -> Dict[str, List[str]]:
        """
        Compare current features with another feature list.
        
        Args:
            other_features: List of features to compare
            
        Returns:
            dict: Comparison results with missing, extra, and common features
        """
        if self.feature_columns is None:
            raise ValueError("Feature metadata not loaded.")
        
        current_set = set(self.feature_columns)
        other_set = set(other_features)
        
        return {
            'missing_in_other': sorted(current_set - other_set),
            'extra_in_other': sorted(other_set - current_set),
            'common': sorted(current_set & other_set),
            'total_current': len(current_set),
            'total_other': len(other_set),
            'overlap_percentage': len(current_set & other_set) / max(len(current_set), 1) * 100
        }


def load_production_features(models_dir: str = "models/production") -> FeatureManager:
    """
    Load feature manager with production feature definitions.
    
    Args:
        models_dir: Directory containing production models
        
    Returns:
        FeatureManager: Configured feature manager
    """
    metadata_path = Path(models_dir) / "features_metadata.json"
    
    if not metadata_path.exists():
        # Fallback to legacy features.json
        legacy_path = Path(models_dir) / "feature_columns.json"
        if legacy_path.exists():
            logger.warning("Using legacy feature file. Consider upgrading to features_metadata.json")
            metadata_path = legacy_path
        else:
            raise FileNotFoundError(f"No feature definitions found in {models_dir}")
    
    return FeatureManager(str(metadata_path))


def validate_model_input(data: Union[pd.DataFrame, np.ndarray], 
                        models_dir: str = "models/production") -> bool:
    """
    Convenient function to validate model input data.
    
    Args:
        data: Input data to validate
        models_dir: Directory containing feature definitions
        
    Returns:
        bool: True if validation passes
    """
    feature_manager = load_production_features(models_dir)
    return feature_manager.validate_input_data(data)


# Legacy support for backward compatibility
def load_feature_columns(filepath: str) -> List[str]:
    """
    Load feature columns from JSON file (legacy function).
    
    Args:
        filepath: Path to features JSON file
        
    Returns:
        list: Feature column names
    """
    logger.warning("load_feature_columns is deprecated. Use FeatureManager instead.")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Handle both old and new formats
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'feature_columns' in data:
        return data['feature_columns']
    else:
        raise ValueError("Unrecognized feature file format")


# Default feature columns for backward compatibility
DEFAULT_FEATURE_COLUMNS = [
    "koi_period", "koi_period_err1", "koi_period_err2", 
    "koi_time0bk", "koi_time0bk_err1", "koi_time0bk_err2", 
    "koi_impact", "koi_impact_err1", "koi_impact_err2", 
    "koi_duration", "koi_duration_err1", "koi_duration_err2", 
    "koi_depth", "koi_depth_err1", "koi_depth_err2", 
    "koi_prad", "koi_prad_err1", "koi_prad_err2", 
    "koi_teq", 
    "koi_insol", "koi_insol_err1", "koi_insol_err2", 
    "koi_model_snr", 
    "koi_tce_plnt_num", 
    "koi_steff", "koi_steff_err1", "koi_steff_err2", 
    "koi_slogg", "koi_slogg_err1", "koi_slogg_err2", 
    "koi_srad", "koi_srad_err1", "koi_srad_err2", 
    "ra", "dec", 
    "koi_kepmag"
]