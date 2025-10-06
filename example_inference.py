"""
Example script demonstrating feature management and model inference.

This script shows how to:
1. Load feature definitions
2. Validate input data
3. Prepare data for model inference
4. Load and use trained models
"""

import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Add src to path for imports
sys.path.append('src')

from feature_management import FeatureManager, load_production_features
from data_preprocessing import ExoplanetDataPreprocessor


def main():
    """Demonstrate complete feature management and inference pipeline."""
    
    print("🚀 Exoplanet Detection - Feature Management Demo")
    print("=" * 50)
    
    # 1. Load feature definitions
    print("\n1. Loading feature definitions...")
    try:
        feature_manager = load_production_features("models/production")
        feature_columns = feature_manager.get_feature_columns()
        print(f"✅ Loaded {len(feature_columns)} feature definitions")
        
        # Show feature groups
        feature_groups = feature_manager.get_feature_groups()
        print(f"📊 Feature groups: {list(feature_groups.keys())}")
        
    except Exception as e:
        print(f"❌ Error loading features: {e}")
        return
    
    # 2. Load sample data
    print("\n2. Loading sample data...")
    try:
        data_path = "data/cumulative_2025.10.04_05.21.55.csv"
        if not Path(data_path).exists():
            print(f"❌ Data file not found: {data_path}")
            return
            
        # Load just first 10 rows for demo
        df = pd.read_csv(data_path, comment='#', engine='python', nrows=10)
        print(f"✅ Loaded sample data: {df.shape}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # 3. Validate and prepare data
    print("\n3. Validating and preparing data...")
    try:
        # Check for missing columns
        missing_cols = feature_manager.get_missing_columns(df)
        if missing_cols:
            print(f"⚠️  Missing columns: {missing_cols[:5]}...")  # Show first 5
        
        # Prepare data (this will handle missing columns and order correctly)
        preprocessor = ExoplanetDataPreprocessor()
        df_clean = preprocessor.remove_irrelevant_columns(df)
        df_imputed = preprocessor.handle_missing_values(df_clean)
        
        # Check if we have the required features
        available_features = [col for col in feature_columns if col in df_imputed.columns]
        print(f"✅ Available features: {len(available_features)}/{len(feature_columns)}")
        
        if len(available_features) < len(feature_columns) * 0.8:  # Need at least 80% of features
            print("❌ Insufficient features for reliable prediction")
            return
            
        # Prepare input matrix
        X = df_imputed[available_features].fillna(0)  # Fill any remaining NaNs
        X_scaled = preprocessor.scale_features(X)
        
        print(f"✅ Prepared data for inference: {X_scaled.shape}")
        
    except Exception as e:
        print(f"❌ Error preparing data: {e}")
        return
    
    # 4. Load and use trained model
    print("\n4. Loading trained model...")
    try:
        # Try to load the stacking model
        model_path = "models/production/sub_models/stacking.pkl"
        if not Path(model_path).exists():
            print(f"⚠️  Model not found at {model_path}")
            print("   Available models in production:")
            prod_dir = Path("models/production/sub_models")
            if prod_dir.exists():
                for model_file in prod_dir.glob("*.pkl"):
                    print(f"   - {model_file.name}")
            return
        
        model = joblib.load(model_path)
        print(f"✅ Loaded model: {type(model).__name__}")
        
        # Make predictions
        if X_scaled.shape[1] == len(feature_columns):
            predictions = model.predict(X_scaled)
            probabilities = model.predict_proba(X_scaled)[:, 1]
            
            print(f"✅ Made predictions for {len(predictions)} samples")
            
            # Show results
            print("\n5. Prediction Results:")
            print("-" * 30)
            for i in range(min(5, len(predictions))):  # Show first 5 results
                pred_class = "Confirmed Exoplanet" if predictions[i] == 1 else "Candidate/FP"
                confidence = probabilities[i]
                print(f"Sample {i+1}: {pred_class} (confidence: {confidence:.3f})")
                
        else:
            print(f"❌ Feature mismatch: model expects {len(feature_columns)}, got {X_scaled.shape[1]}")
            
    except Exception as e:
        print(f"❌ Error with model inference: {e}")
        return
    
    # 6. Feature importance analysis
    print("\n6. Feature Analysis:")
    print("-" * 20)
    try:
        # Show feature summary
        feature_summary = feature_manager.create_feature_summary()
        print(f"📋 Feature summary created with {len(feature_summary)} features")
        
        # Show top features by group
        for group in feature_groups:
            group_features = feature_groups[group]
            available_in_group = [f for f in group_features if f in available_features]
            print(f"   {group}: {len(available_in_group)}/{len(group_features)} features available")
        
    except Exception as e:
        print(f"⚠️  Feature analysis error: {e}")
    
    print("\n🎉 Demo completed successfully!")
    print("\nNext steps:")
    print("- Ensure all required features are available in your data")
    print("- Use the feature_manager to validate input before inference")
    print("- Load the appropriate scaler used during training")
    print("- Apply the same preprocessing steps as during training")


if __name__ == "__main__":
    main()