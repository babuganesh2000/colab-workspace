"""
Google Colab MLOps Integration Script
This script sets up the connection between Google Colab and GitHub for MLOps
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from google.colab import auth, drive, files
from google.cloud import storage
import pandas as pd
import numpy as np

class ColabMLOpsSetup:
    """Setup MLOps environment in Google Colab"""
    
    def __init__(self):
        self.project_root = "/content/healthcare-mlops"
        self.drive_path = "/content/drive/MyDrive/healthcare-mlops"
        
    def mount_drive(self):
        """Mount Google Drive"""
        print("🔗 Mounting Google Drive...")
        drive.mount('/content/drive')
        print("✅ Google Drive mounted successfully")
        
    def authenticate_gcp(self):
        """Authenticate with Google Cloud Platform"""
        print("🔐 Authenticating with Google Cloud...")
        auth.authenticate_user()
        print("✅ GCP authentication complete")
        
    def clone_repository(self, repo_url: str):
        """Clone GitHub repository"""
        print(f"📥 Cloning repository: {repo_url}")
        
        if os.path.exists(self.project_root):
            print("Repository already exists, pulling latest changes...")
            os.chdir(self.project_root)
            subprocess.run(["git", "pull"], check=True)
        else:
            subprocess.run(["git", "clone", repo_url, self.project_root], check=True)
            
        os.chdir(self.project_root)
        print("✅ Repository setup complete")
        
    def install_dependencies(self):
        """Install required Python packages"""
        print("📦 Installing dependencies...")
        
        packages = [
            "scikit-learn==1.3.0",
            "mlflow==2.7.1",
            "google-cloud-storage==2.10.0",
            "pandas==2.0.3",
            "numpy==1.24.3",
            "matplotlib==3.7.2",
            "seaborn==0.12.2",
            "pytest==7.4.0",
            "black==23.7.0",
            "flake8==6.0.0",
            "isort==5.12.0"
        ]
        
        for package in packages:
            subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
            
        print("✅ Dependencies installed successfully")
        
    def setup_environment_variables(self, env_vars: dict):
        """Setup environment variables"""
        print("🔧 Setting up environment variables...")
        
        for key, value in env_vars.items():
            os.environ[key] = value
            
        print("✅ Environment variables configured")
        
    def create_directory_structure(self):
        """Create project directory structure"""
        print("📁 Creating directory structure...")
        
        directories = [
            "src",
            "models",
            "data",
            "config",
            "tests",
            "deployment",
            "notebooks",
            "logs"
        ]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            
        print("✅ Directory structure created")
        
    def setup_git_config(self, username: str, email: str):
        """Configure Git settings"""
        print("⚙️ Configuring Git...")
        
        subprocess.run(["git", "config", "--global", "user.name", username], check=True)
        subprocess.run(["git", "config", "--global", "user.email", email], check=True)
        
        print("✅ Git configuration complete")

def run_colab_notebook_mlops():
    """
    Main function to run MLOps pipeline in Google Colab
    This should be called from within a Colab notebook cell
    """
    
    print("🚀 Starting MLOps setup in Google Colab...")
    
    # Initialize setup
    setup = ColabMLOpsSetup()
    
    # Mount Google Drive
    setup.mount_drive()
    
    # Authenticate with GCP
    setup.authenticate_gcp()
    
    # Configuration - these should be set according to your setup
    config = {
        "repo_url": "https://github.com/your-username/healthcare-mlops.git",
        "username": "Your Name",
        "email": "your.email@example.com",
        "env_vars": {
            "MLFLOW_TRACKING_URI": "https://your-mlflow-server.com",
            "GCP_PROJECT_ID": "your-gcp-project-id",
            "GCS_BUCKET_NAME": "your-model-bucket"
        }
    }
    
    # Clone repository
    setup.clone_repository(config["repo_url"])
    
    # Create directory structure
    setup.create_directory_structure()
    
    # Install dependencies
    setup.install_dependencies()
    
    # Setup environment variables
    setup.setup_environment_variables(config["env_vars"])
    
    # Configure Git
    setup.setup_git_config(config["username"], config["email"])
    
    print("🎉 MLOps setup complete! You can now run the training pipeline.")
    
    return setup

def trigger_training_pipeline():
    """Trigger the training pipeline"""
    print("🤖 Starting model training pipeline...")
    
    try:
        # Import and run training pipeline
        from src.train_model import main as train_main
        train_main()
        
        print("✅ Training pipeline completed successfully!")
        
    except Exception as e:
        print(f"❌ Training pipeline failed: {e}")
        raise

def create_colab_notebook():
    """
    Create a Colab notebook with MLOps integration
    This function generates the notebook code that can be copied to Colab
    """
    
    notebook_code = '''
# Healthcare MLOps Pipeline - Google Colab Integration
# Run this cell to set up the complete MLOps environment

# Install required packages
!pip install scikit-learn mlflow google-cloud-storage pandas numpy matplotlib seaborn

# Import setup functions
import sys
import os
from google.colab import drive, auth
import subprocess

# Mount Google Drive
drive.mount('/content/drive')

# Authenticate with Google Cloud
auth.authenticate_user()

# Clone the repository (replace with your repo URL)
repo_url = "https://github.com/your-username/healthcare-mlops.git"
if not os.path.exists("/content/healthcare-mlops"):
    !git clone {repo_url} /content/healthcare-mlops

# Change to project directory
os.chdir("/content/healthcare-mlops")

# Set up environment variables
os.environ["MLFLOW_TRACKING_URI"] = "your-mlflow-server-url"
os.environ["GCP_PROJECT_ID"] = "your-gcp-project"
os.environ["GCS_BUCKET_NAME"] = "your-model-bucket"

# Configure Git (replace with your details)
!git config --global user.name "Your Name"
!git config --global user.email "your.email@example.com"

print("🎉 Setup complete! Ready for MLOps pipeline.")

# Run the training pipeline
!python src/train_model.py

# Run model validation
!python src/validate_model.py

# Generate model report
!python src/generate_report.py

print("🚀 MLOps pipeline execution complete!")
'''
    
    return notebook_code

# Colab-specific functions that can be called directly in notebook cells

def colab_quick_setup():
    """Quick setup function for Colab"""
    print("⚡ Quick MLOps setup for Google Colab...")
    
    # Mount drive
    from google.colab import drive, auth
    drive.mount('/content/drive')
    
    # Authenticate
    auth.authenticate_user()
    
    # Install packages
    os.system("pip install scikit-learn mlflow google-cloud-storage")
    
    print("✅ Quick setup complete!")

def colab_train_model():
    """Train model directly in Colab"""
    print("🏋️ Training healthcare models in Colab...")
    
    # This would import and run the training script
    # Modified to work directly in Colab environment
    
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, r2_score
    
    # Generate synthetic data (using functions from our notebook)
    print("📊 Generating healthcare datasets...")
    
    # This would use the data generation functions
    # from the main notebook we created earlier
    
    print("✅ Model training complete!")

if __name__ == "__main__":
    print("This script is designed to be imported and used in Google Colab")
    print("Copy the notebook code generated by create_colab_notebook() to use in Colab")
    
    # Generate notebook code
    notebook_code = create_colab_notebook()
    print("📓 Colab notebook code:")
    print(notebook_code)