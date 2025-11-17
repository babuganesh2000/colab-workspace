"""
Comprehensive test suite for healthcare ML models and MLOps pipeline
"""

import pytest
import pandas as pd
import numpy as np
import pickle
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json

# Add src to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from train_model import HealthcareMLPipeline
from validate_model import ModelValidator

class TestDataGeneration:
    """Test data generation functions"""
    
    def test_heart_disease_data_generation(self):
        """Test heart disease dataset generation"""
        pipeline = HealthcareMLPipeline()
        
        # Generate data
        X, y = pipeline._generate_heart_disease_data(n_samples=100)
        
        # Check shapes
        assert X.shape[0] == 100
        assert len(y) == 100
        assert X.shape[1] == 10  # Expected features
        
        # Check data types
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        
        # Check value ranges
        assert X['age'].min() >= 20
        assert X['age'].max() <= 80
        assert set(X['sex'].unique()) <= {0, 1}
        assert set(y.unique()) <= {0, 1}
    
    def test_blood_pressure_data_generation(self):
        """Test blood pressure dataset generation"""
        pipeline = HealthcareMLPipeline()
        
        # Generate data
        X, y = pipeline._generate_blood_pressure_data(n_samples=100)
        
        # Check shapes
        assert X.shape[0] == 100
        assert len(y) == 100
        assert X.shape[1] == 9  # Expected features
        
        # Check data types
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        
        # Check value ranges
        assert X['age'].min() >= 20
        assert X['age'].max() <= 80
        assert y.min() >= 90  # Reasonable systolic BP range
        assert y.max() <= 200

class TestModelTraining:
    """Test model training functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.pipeline = HealthcareMLPipeline()
        
        # Create test data
        self.X_class, self.y_class = self.pipeline._generate_heart_disease_data(n_samples=100)
        self.X_reg, self.y_reg = self.pipeline._generate_blood_pressure_data(n_samples=100)
    
    def test_classification_model_training(self):
        """Test heart disease classification model training"""
        # Train model
        model, metrics = self.pipeline.train_classification_model(
            self.X_class, self.y_class
        )
        
        # Check model exists
        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
        
        # Check metrics
        required_metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        for metric in required_metrics:
            assert metric in metrics
            assert 0 <= metrics[metric] <= 1
    
    def test_regression_model_training(self):
        """Test blood pressure regression model training"""
        # Train model
        model, metrics = self.pipeline.train_regression_model(
            self.X_reg, self.y_reg
        )
        
        # Check model exists
        assert model is not None
        assert hasattr(model, 'predict')
        
        # Check metrics
        required_metrics = ['r2', 'mae', 'mse', 'rmse']
        for metric in required_metrics:
            assert metric in metrics
        
        # R2 should be reasonable
        assert -1 <= metrics['r2'] <= 1
        
        # Error metrics should be positive
        assert metrics['mae'] >= 0
        assert metrics['mse'] >= 0
        assert metrics['rmse'] >= 0
    
    def test_hyperparameter_tuning(self):
        """Test hyperparameter tuning functionality"""
        # Test classification tuning
        best_params, best_score = self.pipeline.tune_hyperparameters(
            self.X_class, self.y_class, model_type='classification'
        )
        
        assert isinstance(best_params, dict)
        assert 'n_estimators' in best_params
        assert 'max_depth' in best_params
        assert isinstance(best_score, (float, np.float64))
        assert 0 <= best_score <= 1

class TestModelValidation:
    """Test model validation functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.pipeline = HealthcareMLPipeline()
        self.validator = ModelValidator()
        
        # Create and train test models
        X_class, y_class = self.pipeline._generate_heart_disease_data(n_samples=200)
        X_reg, y_reg = self.pipeline._generate_blood_pressure_data(n_samples=200)
        
        self.class_model, self.class_metrics = self.pipeline.train_classification_model(
            X_class, y_class
        )
        self.reg_model, self.reg_metrics = self.pipeline.train_regression_model(
            X_reg, y_reg
        )
        
        # Create test data for validation
        self.X_test_class, self.y_test_class = self.pipeline._generate_heart_disease_data(
            n_samples=50
        )
        self.X_test_reg, self.y_test_reg = self.pipeline._generate_blood_pressure_data(
            n_samples=50
        )
    
    def test_classification_validation(self):
        """Test classification model validation"""
        # Validate model
        is_valid, metrics = self.validator.validate_classification_model(
            self.class_model, self.X_test_class, self.y_test_class
        )
        
        assert isinstance(is_valid, bool)
        assert isinstance(metrics, dict)
        
        # Check required metrics
        required_metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        for metric in required_metrics:
            assert metric in metrics
    
    def test_regression_validation(self):
        """Test regression model validation"""
        # Validate model
        is_valid, metrics = self.validator.validate_regression_model(
            self.reg_model, self.X_test_reg, self.y_test_reg
        )
        
        assert isinstance(is_valid, bool)
        assert isinstance(metrics, dict)
        
        # Check required metrics
        required_metrics = ['r2', 'mae', 'mse', 'rmse']
        for metric in required_metrics:
            assert metric in metrics
    
    def test_performance_thresholds(self):
        """Test performance threshold validation"""
        # Test with different thresholds
        thresholds = {
            'accuracy': 0.5,  # Low threshold for test
            'precision': 0.5,
            'recall': 0.5,
            'f1': 0.5,
            'auc': 0.5
        }
        
        is_valid = self.validator._check_classification_thresholds(
            self.class_metrics, thresholds
        )
        
        assert isinstance(is_valid, bool)

class TestDataValidation:
    """Test data validation functionality"""
    
    def test_data_schema_validation(self):
        """Test data schema validation"""
        pipeline = HealthcareMLPipeline()
        
        # Create valid data
        X_valid, _ = pipeline._generate_heart_disease_data(n_samples=10)
        
        # Validate schema
        is_valid = pipeline._validate_data_schema(X_valid, 'classification')
        assert is_valid is True
        
        # Test with missing columns
        X_invalid = X_valid.drop(columns=['age'])
        is_valid = pipeline._validate_data_schema(X_invalid, 'classification')
        assert is_valid is False
    
    def test_data_quality_checks(self):
        """Test data quality validation"""
        pipeline = HealthcareMLPipeline()
        
        # Create test data
        X, y = pipeline._generate_heart_disease_data(n_samples=100)
        
        # Check for missing values
        missing_ratio = X.isnull().sum().sum() / (X.shape[0] * X.shape[1])
        assert missing_ratio == 0  # Generated data should have no missing values
        
        # Check for duplicates
        duplicate_ratio = X.duplicated().sum() / len(X)
        assert duplicate_ratio < 0.1  # Should have low duplicate rate

class TestModelPersistence:
    """Test model saving and loading"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.pipeline = HealthcareMLPipeline()
        self.test_dir = Path("test_models")
        self.test_dir.mkdir(exist_ok=True)
    
    def teardown_method(self):
        """Clean up test files"""
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_model_saving_and_loading(self):
        """Test model persistence"""
        # Train a simple model
        X, y = self.pipeline._generate_heart_disease_data(n_samples=100)
        model, _ = self.pipeline.train_classification_model(X, y)
        
        # Save model
        model_path = self.test_dir / "test_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Load model
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
        
        # Test predictions are the same
        original_pred = model.predict(X.iloc[:10])
        loaded_pred = loaded_model.predict(X.iloc[:10])
        
        np.testing.assert_array_equal(original_pred, loaded_pred)

class TestIntegration:
    """Integration tests for the complete pipeline"""
    
    @patch('src.train_model.mlflow')
    def test_full_pipeline_integration(self, mock_mlflow):
        """Test complete pipeline from data generation to model validation"""
        # Mock MLflow
        mock_mlflow.start_run.return_value.__enter__ = Mock()
        mock_mlflow.start_run.return_value.__exit__ = Mock()
        mock_mlflow.log_params = Mock()
        mock_mlflow.log_metrics = Mock()
        mock_mlflow.sklearn.log_model = Mock()
        
        # Initialize pipeline
        pipeline = HealthcareMLPipeline()
        validator = ModelValidator()
        
        # Run classification pipeline
        class_results = pipeline.run_classification_pipeline()
        assert 'model' in class_results
        assert 'metrics' in class_results
        
        # Run regression pipeline  
        reg_results = pipeline.run_regression_pipeline()
        assert 'model' in reg_results
        assert 'metrics' in reg_results
        
        # Validate models
        X_test, y_test = pipeline._generate_heart_disease_data(n_samples=50)
        is_valid, _ = validator.validate_classification_model(
            class_results['model'], X_test, y_test
        )
        assert isinstance(is_valid, bool)

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_input_data(self):
        """Test handling of invalid input data"""
        pipeline = HealthcareMLPipeline()
        
        # Test with empty data
        with pytest.raises((ValueError, Exception)):
            pipeline.train_classification_model(
                pd.DataFrame(), pd.Series(dtype=int)
            )
        
        # Test with mismatched dimensions
        X = pd.DataFrame({'feature1': [1, 2, 3]})
        y = pd.Series([1, 2])  # Different length
        
        with pytest.raises((ValueError, Exception)):
            pipeline.train_classification_model(X, y)
    
    def test_model_validation_edge_cases(self):
        """Test model validation with edge cases"""
        validator = ModelValidator()
        
        # Test with None model
        with pytest.raises((AttributeError, TypeError)):
            validator.validate_classification_model(
                None, pd.DataFrame(), pd.Series(dtype=int)
            )

# Smoke Tests
class TestSmokeTests:
    """Basic smoke tests to ensure everything loads"""
    
    def test_imports(self):
        """Test that all modules can be imported"""
        try:
            from train_model import HealthcareMLPipeline
            from validate_model import ModelValidator
            assert True
        except ImportError as e:
            pytest.fail(f"Import failed: {e}")
    
    def test_basic_instantiation(self):
        """Test basic class instantiation"""
        pipeline = HealthcareMLPipeline()
        validator = ModelValidator()
        
        assert pipeline is not None
        assert validator is not None
    
    def test_data_generation_smoke(self):
        """Smoke test for data generation"""
        pipeline = HealthcareMLPipeline()
        
        # Should not raise exceptions
        X1, y1 = pipeline._generate_heart_disease_data(n_samples=10)
        X2, y2 = pipeline._generate_blood_pressure_data(n_samples=10)
        
        assert len(X1) == 10
        assert len(X2) == 10
        assert len(y1) == 10
        assert len(y2) == 10

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])