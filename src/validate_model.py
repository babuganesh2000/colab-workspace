"""
Model validation and testing pipeline
"""

import sys
import json
import pickle
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelValidator:
    """Validate trained models against performance thresholds"""
    
    def __init__(self, config_path: str = "config/model_config.json"):
        """Initialize validator with configuration"""
        self.config = self._load_config(config_path)
        self.validation_results = {}
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            return {
                "thresholds": {
                    "min_accuracy": 0.80,
                    "min_r2_score": 0.70,
                    "max_cv_std": 0.05
                }
            }
    
    def load_latest_models(self, model_dir: str = "models") -> dict:
        """Load the latest trained models"""
        logger.info("Loading latest models for validation...")
        
        model_files = list(Path(model_dir).glob("*.pkl"))
        if not model_files:
            raise FileNotFoundError(f"No model files found in {model_dir}")
        
        # Find latest models
        latest_clf = max([f for f in model_files if "classifier" in f.name], key=lambda x: x.stat().st_mtime)
        latest_reg = max([f for f in model_files if "regressor" in f.name], key=lambda x: x.stat().st_mtime)
        
        models = {}
        
        with open(latest_clf, 'rb') as f:
            models['classification'] = pickle.load(f)
            
        with open(latest_reg, 'rb') as f:
            models['regression'] = pickle.load(f)
            
        logger.info(f"Loaded models: {latest_clf.name}, {latest_reg.name}")
        return models
    
    def validate_classification_model(self, model, X_test, y_test, X_train, y_train) -> dict:
        """Validate classification model"""
        logger.info("Validating classification model...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Check thresholds
        min_accuracy = self.config["thresholds"]["min_accuracy"]
        max_cv_std = self.config["thresholds"]["max_cv_std"]
        
        validation_passed = (
            accuracy >= min_accuracy and
            cv_std <= max_cv_std
        )
        
        results = {
            "accuracy": accuracy,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "min_accuracy_threshold": min_accuracy,
            "max_cv_std_threshold": max_cv_std,
            "validation_passed": validation_passed,
            "classification_report": classification_report(y_test, y_pred, output_dict=True)
        }
        
        if validation_passed:
            logger.info(f"✅ Classification validation PASSED - Accuracy: {accuracy:.3f}")
        else:
            logger.error(f"❌ Classification validation FAILED - Accuracy: {accuracy:.3f}")
            
        return results
    
    def validate_regression_model(self, model, X_test, y_test, X_train, y_train) -> dict:
        """Validate regression model"""
        logger.info("Validating regression model...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = np.mean(np.abs(y_test - y_pred))
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Check thresholds
        min_r2 = self.config["thresholds"]["min_r2_score"]
        max_cv_std = self.config["thresholds"]["max_cv_std"]
        
        validation_passed = (
            r2 >= min_r2 and
            cv_std <= max_cv_std
        )
        
        results = {
            "r2_score": r2,
            "rmse": rmse,
            "mae": mae,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "min_r2_threshold": min_r2,
            "max_cv_std_threshold": max_cv_std,
            "validation_passed": validation_passed
        }
        
        if validation_passed:
            logger.info(f"✅ Regression validation PASSED - R²: {r2:.3f}")
        else:
            logger.error(f"❌ Regression validation FAILED - R²: {r2:.3f}")
            
        return results
    
    def run_model_validation(self, models: dict, data_splits: dict) -> bool:
        """Run complete model validation"""
        logger.info("Starting comprehensive model validation...")
        
        all_passed = True
        
        # Validate Classification Model
        clf_results = self.validate_classification_model(
            models['classification'],
            data_splits['classification']['X_test'],
            data_splits['classification']['y_test'],
            data_splits['classification']['X_train'],
            data_splits['classification']['y_train']
        )
        self.validation_results['classification'] = clf_results
        
        if not clf_results['validation_passed']:
            all_passed = False
        
        # Validate Regression Model
        reg_results = self.validate_regression_model(
            models['regression'],
            data_splits['regression']['X_test'],
            data_splits['regression']['y_test'],
            data_splits['regression']['X_train'],
            data_splits['regression']['y_train']
        )
        self.validation_results['regression'] = reg_results
        
        if not reg_results['validation_passed']:
            all_passed = False
        
        return all_passed
    
    def generate_validation_report(self, output_path: str = "validation_report.json"):
        """Generate detailed validation report"""
        logger.info(f"Generating validation report: {output_path}")
        
        with open(output_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)
        
        # Print summary
        print("\n" + "="*50)
        print("MODEL VALIDATION SUMMARY")
        print("="*50)
        
        clf_results = self.validation_results.get('classification', {})
        reg_results = self.validation_results.get('regression', {})
        
        print(f"Classification Model:")
        print(f"  ✓ Accuracy: {clf_results.get('accuracy', 0):.3f} (Threshold: {clf_results.get('min_accuracy_threshold', 0):.3f})")
        print(f"  ✓ CV Std: {clf_results.get('cv_std', 0):.3f} (Max: {clf_results.get('max_cv_std_threshold', 0):.3f})")
        print(f"  ✓ Status: {'PASSED' if clf_results.get('validation_passed', False) else 'FAILED'}")
        
        print(f"\nRegression Model:")
        print(f"  ✓ R² Score: {reg_results.get('r2_score', 0):.3f} (Threshold: {reg_results.get('min_r2_threshold', 0):.3f})")
        print(f"  ✓ RMSE: {reg_results.get('rmse', 0):.2f} mmHg")
        print(f"  ✓ CV Std: {reg_results.get('cv_std', 0):.3f} (Max: {reg_results.get('max_cv_std_threshold', 0):.3f})")
        print(f"  ✓ Status: {'PASSED' if reg_results.get('validation_passed', False) else 'FAILED'}")
        
        overall_status = (
            clf_results.get('validation_passed', False) and 
            reg_results.get('validation_passed', False)
        )
        print(f"\nOverall Validation: {'PASSED ✅' if overall_status else 'FAILED ❌'}")
        print("="*50)

def main():
    """Main validation pipeline"""
    logger.info("Starting model validation pipeline...")
    
    try:
        # Initialize validator
        validator = ModelValidator()
        
        # Load models
        models = validator.load_latest_models()
        
        # Load test data (this would typically come from a data loader)
        # For now, we'll regenerate the data
        sys.path.append('.')
        from src.train_model import HealthcareMLPipeline
        
        pipeline = HealthcareMLPipeline()
        heart_data, bp_data = pipeline.load_data()
        data_splits = pipeline.prepare_data(heart_data, bp_data)
        
        # Run validation
        validation_passed = validator.run_model_validation(models, data_splits)
        
        # Generate report
        validator.generate_validation_report()
        
        if validation_passed:
            logger.info("🎉 All models passed validation!")
            sys.exit(0)
        else:
            logger.error("❌ Model validation failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Validation pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()