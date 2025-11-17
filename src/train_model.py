"""
MLOps Training Pipeline for Healthcare Random Forest Models
Integrates with Google Colab and GitHub Actions
"""

import os
import sys
import json
import pickle
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, r2_score, mean_squared_error
import mlflow
from google.colab import drive, files
from google.cloud import storage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthcareMLPipeline:
    """Main MLOps pipeline for healthcare models"""
    
    def __init__(self, config_path: str = "config/model_config.json"):
        """Initialize pipeline with configuration"""
        self.config = self._load_config(config_path)
        self.models = {}
        self.metrics = {}
        
        # Initialize MLflow
        mlflow.set_experiment(self.config["experiment_name"])
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            return self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """Return default configuration"""
        return {
            "experiment_name": "healthcare_random_forest",
            "models": {
                "classification": {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "min_samples_split": 5,
                    "random_state": 42
                },
                "regression": {
                    "n_estimators": 100,
                    "max_depth": 12,
                    "min_samples_split": 5,
                    "random_state": 42
                }
            },
            "data": {
                "heart_disease_samples": 1000,
                "blood_pressure_samples": 800,
                "test_size": 0.2
            },
            "thresholds": {
                "min_accuracy": 0.80,
                "min_r2_score": 0.70
            }
        }
    
    def mount_drive(self):
        """Mount Google Drive in Colab"""
        if 'google.colab' in sys.modules:
            drive.mount('/content/drive')
            logger.info("Google Drive mounted successfully")
        else:
            logger.info("Not running in Colab, skipping drive mount")
    
    def load_data(self) -> tuple:
        """Load or generate healthcare datasets"""
        logger.info("Loading healthcare datasets...")
        
        # Import the data generation functions from our notebook
        sys.path.append('/content/drive/MyDrive/colab-workspace')
        
        try:
            from random_forest_healthcare import create_heart_disease_dataset, create_blood_pressure_dataset
            
            # Generate datasets
            heart_data = create_heart_disease_dataset(self.config["data"]["heart_disease_samples"])
            bp_data = create_blood_pressure_dataset(self.config["data"]["blood_pressure_samples"])
            
            logger.info(f"Heart disease dataset: {heart_data.shape}")
            logger.info(f"Blood pressure dataset: {bp_data.shape}")
            
            return heart_data, bp_data
            
        except ImportError:
            logger.error("Could not import data generation functions")
            raise
    
    def prepare_data(self, heart_data: pd.DataFrame, bp_data: pd.DataFrame) -> dict:
        """Prepare data for training"""
        logger.info("Preparing data for training...")
        
        data_splits = {}
        
        # Classification data (Heart Disease)
        X_clf = heart_data.drop('heart_disease', axis=1)
        y_clf = heart_data['heart_disease']
        
        X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
            X_clf, y_clf, test_size=self.config["data"]["test_size"], 
            random_state=42, stratify=y_clf
        )
        
        data_splits['classification'] = {
            'X_train': X_train_clf, 'X_test': X_test_clf,
            'y_train': y_train_clf, 'y_test': y_test_clf
        }
        
        # Regression data (Blood Pressure)
        X_reg = bp_data.drop('systolic_bp', axis=1)
        y_reg = bp_data['systolic_bp']
        
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
            X_reg, y_reg, test_size=self.config["data"]["test_size"], random_state=42
        )
        
        data_splits['regression'] = {
            'X_train': X_train_reg, 'X_test': X_test_reg,
            'y_train': y_train_reg, 'y_test': y_test_reg
        }
        
        return data_splits
    
    def train_models(self, data_splits: dict):
        """Train both classification and regression models"""
        logger.info("Training models...")
        
        with mlflow.start_run() as run:
            # Train Classification Model
            clf_config = self.config["models"]["classification"]
            classifier = RandomForestClassifier(**clf_config)
            
            classifier.fit(
                data_splits['classification']['X_train'],
                data_splits['classification']['y_train']
            )
            
            # Evaluate Classification Model
            y_pred_clf = classifier.predict(data_splits['classification']['X_test'])
            clf_accuracy = accuracy_score(data_splits['classification']['y_test'], y_pred_clf)
            
            # Cross-validation
            cv_scores_clf = cross_val_score(
                classifier, data_splits['classification']['X_train'],
                data_splits['classification']['y_train'], cv=5
            )
            
            # Log Classification Metrics
            mlflow.log_params(clf_config)
            mlflow.log_metric("classification_accuracy", clf_accuracy)
            mlflow.log_metric("classification_cv_mean", cv_scores_clf.mean())
            mlflow.log_metric("classification_cv_std", cv_scores_clf.std())
            
            self.models['classification'] = classifier
            self.metrics['classification'] = {
                'accuracy': clf_accuracy,
                'cv_scores': cv_scores_clf
            }
            
            # Train Regression Model
            reg_config = self.config["models"]["regression"]
            regressor = RandomForestRegressor(**reg_config)
            
            regressor.fit(
                data_splits['regression']['X_train'],
                data_splits['regression']['y_train']
            )
            
            # Evaluate Regression Model
            y_pred_reg = regressor.predict(data_splits['regression']['X_test'])
            reg_r2 = r2_score(data_splits['regression']['y_test'], y_pred_reg)
            reg_rmse = np.sqrt(mean_squared_error(data_splits['regression']['y_test'], y_pred_reg))
            
            # Cross-validation
            cv_scores_reg = cross_val_score(
                regressor, data_splits['regression']['X_train'],
                data_splits['regression']['y_train'], cv=5, scoring='r2'
            )
            
            # Log Regression Metrics
            mlflow.log_params(reg_config)
            mlflow.log_metric("regression_r2", reg_r2)
            mlflow.log_metric("regression_rmse", reg_rmse)
            mlflow.log_metric("regression_cv_mean", cv_scores_reg.mean())
            mlflow.log_metric("regression_cv_std", cv_scores_reg.std())
            
            self.models['regression'] = regressor
            self.metrics['regression'] = {
                'r2_score': reg_r2,
                'rmse': reg_rmse,
                'cv_scores': cv_scores_reg
            }
            
            logger.info(f"Classification Accuracy: {clf_accuracy:.3f}")
            logger.info(f"Regression R²: {reg_r2:.3f}")
    
    def validate_models(self) -> bool:
        """Validate model performance against thresholds"""
        logger.info("Validating model performance...")
        
        validation_passed = True
        
        # Validate Classification Model
        clf_accuracy = self.metrics['classification']['accuracy']
        min_accuracy = self.config["thresholds"]["min_accuracy"]
        
        if clf_accuracy < min_accuracy:
            logger.error(f"Classification accuracy {clf_accuracy:.3f} below threshold {min_accuracy}")
            validation_passed = False
        else:
            logger.info(f"Classification validation passed: {clf_accuracy:.3f} >= {min_accuracy}")
        
        # Validate Regression Model
        reg_r2 = self.metrics['regression']['r2_score']
        min_r2 = self.config["thresholds"]["min_r2_score"]
        
        if reg_r2 < min_r2:
            logger.error(f"Regression R² {reg_r2:.3f} below threshold {min_r2}")
            validation_passed = False
        else:
            logger.info(f"Regression validation passed: {reg_r2:.3f} >= {min_r2}")
        
        return validation_passed
    
    def save_models(self, model_dir: str = "models"):
        """Save trained models to disk"""
        logger.info(f"Saving models to {model_dir}")
        
        Path(model_dir).mkdir(exist_ok=True)
        
        # Save models
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        clf_path = f"{model_dir}/heart_disease_classifier_{timestamp}.pkl"
        reg_path = f"{model_dir}/blood_pressure_regressor_{timestamp}.pkl"
        
        with open(clf_path, 'wb') as f:
            pickle.dump(self.models['classification'], f)
        
        with open(reg_path, 'wb') as f:
            pickle.dump(self.models['regression'], f)
        
        # Save metadata
        metadata = {
            'classification': {
                'model_path': clf_path,
                'metrics': self.metrics['classification'],
                'timestamp': timestamp
            },
            'regression': {
                'model_path': reg_path,
                'metrics': self.metrics['regression'],
                'timestamp': timestamp
            }
        }
        
        with open(f"{model_dir}/model_metadata_{timestamp}.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info("Models saved successfully")
        
        return clf_path, reg_path
    
    def upload_to_gcs(self, local_path: str, bucket_name: str, blob_name: str):
        """Upload model to Google Cloud Storage"""
        try:
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(local_path)
            logger.info(f"Uploaded {local_path} to gs://{bucket_name}/{blob_name}")
        except Exception as e:
            logger.error(f"Failed to upload to GCS: {e}")

def main():
    """Main training pipeline"""
    logger.info("Starting MLOps training pipeline...")
    
    try:
        # Initialize pipeline
        pipeline = HealthcareMLPipeline()
        
        # Mount Google Drive if in Colab
        pipeline.mount_drive()
        
        # Load data
        heart_data, bp_data = pipeline.load_data()
        
        # Prepare data
        data_splits = pipeline.prepare_data(heart_data, bp_data)
        
        # Train models
        pipeline.train_models(data_splits)
        
        # Validate models
        if not pipeline.validate_models():
            logger.error("Model validation failed!")
            sys.exit(1)
        
        # Save models
        clf_path, reg_path = pipeline.save_models()
        
        # Upload to cloud storage (if configured)
        bucket_name = os.getenv('GCS_BUCKET_NAME')
        if bucket_name:
            pipeline.upload_to_gcs(clf_path, bucket_name, f"models/{os.path.basename(clf_path)}")
            pipeline.upload_to_gcs(reg_path, bucket_name, f"models/{os.path.basename(reg_path)}")
        
        logger.info("Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()