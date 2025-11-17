"""
Model deployment script for staging and production environments
"""

import os
import sys
import json
import pickle
import logging
import argparse
from pathlib import Path
from datetime import datetime
import requests
from google.cloud import storage, run_v2
import docker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelDeployer:
    """Deploy models to different environments"""
    
    def __init__(self, environment: str = "staging"):
        """Initialize deployer for specific environment"""
        self.environment = environment
        self.config = self._load_deployment_config()
        
    def _load_deployment_config(self) -> dict:
        """Load deployment configuration"""
        config_path = f"deployment/{self.environment}_config.json"
        
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            return self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """Return default deployment configuration"""
        return {
            "staging": {
                "gcp_project": os.getenv("GCP_PROJECT_ID", "your-project"),
                "region": "us-central1",
                "service_name": "healthcare-ml-staging",
                "image_name": "healthcare-ml-api",
                "cpu": "1",
                "memory": "2Gi",
                "max_instances": 10
            },
            "production": {
                "gcp_project": os.getenv("GCP_PROJECT_ID", "your-project"),
                "region": "us-central1", 
                "service_name": "healthcare-ml-prod",
                "image_name": "healthcare-ml-api",
                "cpu": "2",
                "memory": "4Gi",
                "max_instances": 100
            }
        }[self.environment]
    
    def create_api_service(self):
        """Create FastAPI service for model serving"""
        logger.info("Creating API service...")
        
        api_code = '''
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging

app = FastAPI(title="Healthcare ML API", version="1.0.0")

# Load models at startup
models = {}

try:
    with open("/app/models/heart_disease_classifier.pkl", "rb") as f:
        models["classification"] = pickle.load(f)
    
    with open("/app/models/blood_pressure_regressor.pkl", "rb") as f:
        models["regression"] = pickle.load(f)
        
    logging.info("Models loaded successfully")
except Exception as e:
    logging.error(f"Failed to load models: {e}")

class HeartDiseaseRequest(BaseModel):
    age: int
    sex: int
    chest_pain_type: int
    resting_bp: int
    cholesterol: int
    fasting_bs: int
    resting_ecg: int
    max_heart_rate: int
    exercise_angina: int
    st_depression: float

class BloodPressureRequest(BaseModel):
    age: int
    bmi: float
    exercise_hours: float
    smoking: int
    alcohol: int
    stress_level: float
    sleep_hours: float
    family_history: int
    sodium_intake: int

class PredictionResponse(BaseModel):
    prediction: float
    probability: float = None
    model_version: str
    timestamp: str

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": len(models)}

@app.post("/predict/heart-disease", response_model=PredictionResponse)
async def predict_heart_disease(request: HeartDiseaseRequest):
    try:
        if "classification" not in models:
            raise HTTPException(status_code=503, detail="Classification model not available")
        
        # Prepare input data
        input_data = pd.DataFrame([request.dict()])
        
        # Make prediction
        prediction = models["classification"].predict(input_data)[0]
        probability = models["classification"].predict_proba(input_data)[0][1]
        
        return PredictionResponse(
            prediction=float(prediction),
            probability=float(probability),
            model_version="1.0.0",
            timestamp=pd.Timestamp.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/blood-pressure", response_model=PredictionResponse)
async def predict_blood_pressure(request: BloodPressureRequest):
    try:
        if "regression" not in models:
            raise HTTPException(status_code=503, detail="Regression model not available")
        
        # Prepare input data
        input_data = pd.DataFrame([request.dict()])
        
        # Make prediction
        prediction = models["regression"].predict(input_data)[0]
        
        return PredictionResponse(
            prediction=float(prediction),
            model_version="1.0.0",
            timestamp=pd.Timestamp.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/info")
async def get_model_info():
    return {
        "classification": {
            "type": "RandomForestClassifier",
            "n_estimators": getattr(models.get("classification"), "n_estimators", None),
            "available": "classification" in models
        },
        "regression": {
            "type": "RandomForestRegressor", 
            "n_estimators": getattr(models.get("regression"), "n_estimators", None),
            "available": "regression" in models
        }
    }
'''
        
        # Write API service
        with open("deployment/api_service.py", "w") as f:
            f.write(api_code)
        
        logger.info("API service created")
    
    def create_dockerfile(self):
        """Create Dockerfile for containerization"""
        logger.info("Creating Dockerfile...")
        
        dockerfile_content = '''
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY deployment/api_service.py .
COPY models/ ./models/

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8080/health || exit 1

# Run the API
CMD ["uvicorn", "api_service:app", "--host", "0.0.0.0", "--port", "8080"]
'''
        
        with open("deployment/Dockerfile", "w") as f:
            f.write(dockerfile_content)
        
        logger.info("Dockerfile created")
    
    def build_and_push_image(self):
        """Build and push Docker image"""
        logger.info("Building and pushing Docker image...")
        
        client = docker.from_env()
        
        # Build image
        image_tag = f"gcr.io/{self.config['gcp_project']}/{self.config['image_name']}:latest"
        
        image, logs = client.images.build(
            path=".",
            dockerfile="deployment/Dockerfile",
            tag=image_tag
        )
        
        # Push to registry
        client.images.push(image_tag)
        
        logger.info(f"Image pushed: {image_tag}")
        return image_tag
    
    def deploy_to_cloud_run(self, image_url: str):
        """Deploy to Google Cloud Run"""
        logger.info(f"Deploying to Cloud Run ({self.environment})...")
        
        client = run_v2.ServicesClient()
        
        service = {
            "template": {
                "containers": [{
                    "image": image_url,
                    "ports": [{"container_port": 8080}],
                    "resources": {
                        "limits": {
                            "cpu": self.config["cpu"],
                            "memory": self.config["memory"]
                        }
                    },
                    "env": [
                        {"name": "ENVIRONMENT", "value": self.environment}
                    ]
                }],
                "scaling": {
                    "max_instance_count": self.config["max_instances"]
                }
            }
        }
        
        request = run_v2.CreateServiceRequest(
            parent=f"projects/{self.config['gcp_project']}/locations/{self.config['region']}",
            service=service,
            service_id=self.config["service_name"]
        )
        
        operation = client.create_service(request=request)
        result = operation.result()
        
        service_url = result.uri
        logger.info(f"Service deployed: {service_url}")
        
        return service_url
    
    def run_deployment_tests(self, service_url: str):
        """Run deployment tests"""
        logger.info("Running deployment tests...")
        
        # Test health endpoint
        health_response = requests.get(f"{service_url}/health")
        if health_response.status_code != 200:
            raise Exception(f"Health check failed: {health_response.status_code}")
        
        # Test heart disease prediction
        heart_test_data = {
            "age": 45,
            "sex": 1,
            "chest_pain_type": 1,
            "resting_bp": 130,
            "cholesterol": 250,
            "fasting_bs": 0,
            "resting_ecg": 0,
            "max_heart_rate": 150,
            "exercise_angina": 0,
            "st_depression": 0.5
        }
        
        heart_response = requests.post(
            f"{service_url}/predict/heart-disease",
            json=heart_test_data
        )
        
        if heart_response.status_code != 200:
            raise Exception(f"Heart disease prediction test failed: {heart_response.status_code}")
        
        # Test blood pressure prediction
        bp_test_data = {
            "age": 45,
            "bmi": 26.5,
            "exercise_hours": 3.0,
            "smoking": 0,
            "alcohol": 1,
            "stress_level": 5.0,
            "sleep_hours": 7.5,
            "family_history": 0,
            "sodium_intake": 2500
        }
        
        bp_response = requests.post(
            f"{service_url}/predict/blood-pressure",
            json=bp_test_data
        )
        
        if bp_response.status_code != 200:
            raise Exception(f"Blood pressure prediction test failed: {bp_response.status_code}")
        
        logger.info("All deployment tests passed!")
        
        return {
            "health": health_response.json(),
            "heart_disease": heart_response.json(),
            "blood_pressure": bp_response.json()
        }

def main():
    """Main deployment pipeline"""
    parser = argparse.ArgumentParser(description="Deploy healthcare ML models")
    parser.add_argument("--environment", choices=["staging", "production"], 
                       default="staging", help="Deployment environment")
    
    args = parser.parse_args()
    
    logger.info(f"Starting deployment to {args.environment}...")
    
    try:
        # Initialize deployer
        deployer = ModelDeployer(args.environment)
        
        # Create deployment artifacts
        deployer.create_api_service()
        deployer.create_dockerfile()
        
        # Build and push Docker image
        image_url = deployer.build_and_push_image()
        
        # Deploy to Cloud Run
        service_url = deployer.deploy_to_cloud_run(image_url)
        
        # Run tests
        test_results = deployer.run_deployment_tests(service_url)
        
        logger.info(f"🎉 Deployment to {args.environment} completed successfully!")
        logger.info(f"Service URL: {service_url}")
        
        # Save deployment info
        deployment_info = {
            "environment": args.environment,
            "service_url": service_url,
            "image_url": image_url,
            "deployment_time": datetime.now().isoformat(),
            "test_results": test_results
        }
        
        with open(f"deployment_{args.environment}_info.json", "w") as f:
            json.dump(deployment_info, f, indent=2)
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()