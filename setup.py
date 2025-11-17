#!/usr/bin/env python3
"""
Healthcare ML MLOps Setup Script
This script helps set up the complete MLOps pipeline for healthcare ML models
"""

import os
import sys
import subprocess
import json
import shutil
from pathlib import Path
import argparse

def run_command(command, check=True, shell=True):
    """Run shell command and return result"""
    try:
        result = subprocess.run(
            command, 
            shell=shell, 
            check=check, 
            capture_output=True, 
            text=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(f"Error output: {e.stderr}")
        if check:
            sys.exit(1)
        return None

def check_requirements():
    """Check if required tools are installed"""
    print("🔍 Checking requirements...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or python_version.minor < 8:
        print("❌ Python 3.8+ is required")
        sys.exit(1)
    print(f"✅ Python {python_version.major}.{python_version.minor} found")
    
    # Check pip
    try:
        import pip
        print("✅ pip found")
    except ImportError:
        print("❌ pip not found")
        sys.exit(1)
    
    # Check git
    git_version = run_command("git --version", check=False)
    if git_version:
        print(f"✅ {git_version}")
    else:
        print("⚠️  git not found - needed for version control")
    
    # Check Docker (optional)
    docker_version = run_command("docker --version", check=False)
    if docker_version:
        print(f"✅ {docker_version}")
    else:
        print("⚠️  Docker not found - needed for containerization")
    
    # Check Google Cloud SDK (optional)
    gcloud_version = run_command("gcloud --version", check=False)
    if gcloud_version:
        print("✅ Google Cloud SDK found")
    else:
        print("⚠️  Google Cloud SDK not found - needed for cloud deployment")

def setup_python_environment():
    """Set up Python virtual environment"""
    print("\n🐍 Setting up Python environment...")
    
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("📁 Virtual environment already exists")
        response = input("Do you want to recreate it? (y/N): ")
        if response.lower() == 'y':
            shutil.rmtree(venv_path)
        else:
            print("Using existing virtual environment")
            return
    
    # Create virtual environment
    print("Creating virtual environment...")
    run_command(f"{sys.executable} -m venv venv")
    
    # Determine activation script based on OS
    if os.name == 'nt':  # Windows
        activate_script = "venv\\Scripts\\activate"
        pip_path = "venv\\Scripts\\pip"
    else:  # Unix/Linux/Mac
        activate_script = "source venv/bin/activate"
        pip_path = "venv/bin/pip"
    
    print(f"✅ Virtual environment created")
    print(f"To activate: {activate_script}")
    
    # Install requirements
    print("Installing Python dependencies...")
    run_command(f"{pip_path} install --upgrade pip")
    run_command(f"{pip_path} install -r requirements.txt")
    
    print("✅ Python dependencies installed")

def setup_configuration():
    """Set up configuration files"""
    print("\n⚙️ Setting up configuration...")
    
    # Create .env file if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        print("Creating .env file...")
        env_content = """# Healthcare ML MLOps Configuration

# Google Cloud Configuration
GCP_PROJECT_ID=your-healthcare-ml-project
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/service-account-key.json

# MLflow Configuration
MLFLOW_TRACKING_URI=sqlite:///mlflow.db

# Cloud Storage
GCS_BUCKET_NAME=healthcare-ml-bucket

# Monitoring Configuration
PROMETHEUS_PORT=8000

# Alert Configuration
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
EMAIL_PASSWORD=your-email-password
PAGERDUTY_INTEGRATION_KEY=your-pagerduty-key

# API Configuration
API_KEY=your-secure-api-key

# Environment
ENVIRONMENT=development
LOG_LEVEL=INFO
"""
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("✅ .env file created - please update with your values")
    else:
        print("✅ .env file already exists")
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    print("✅ Models directory created")
    
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    print("✅ Logs directory created")

def setup_git_hooks():
    """Set up git hooks for code quality"""
    print("\n🪝 Setting up git hooks...")
    
    if not Path(".git").exists():
        print("Initializing git repository...")
        run_command("git init")
        run_command("git add .")
        run_command('git commit -m "Initial commit"')
    
    # Install pre-commit hooks
    if Path("venv").exists():
        if os.name == 'nt':
            pip_path = "venv\\Scripts\\pip"
            precommit_path = "venv\\Scripts\\pre-commit"
        else:
            pip_path = "venv/bin/pip"
            precommit_path = "venv/bin/pre-commit"
        
        # Create pre-commit config
        precommit_config = """repos:
  - repo: https://github.com/psf/black
    rev: 23.11.0
    hooks:
      - id: black
        language_version: python3.9
  
  - repo: https://github.com/pycqa/flake8
    rev: 6.1.0
    hooks:
      - id: flake8
        args: [--max-line-length=88, --extend-ignore=E203]
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.1
    hooks:
      - id: mypy
        additional_dependencies: [types-requests, types-PyYAML]
  
  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: pytest
        language: system
        types: [python]
        pass_filenames: false
        always_run: true
        args: [tests/, -v]
"""
        
        with open(".pre-commit-config.yaml", "w") as f:
            f.write(precommit_config)
        
        run_command(f"{pip_path} install pre-commit")
        run_command(f"{precommit_path} install")
        print("✅ Git hooks installed")
    else:
        print("⚠️  Virtual environment not found - skipping git hooks")

def run_initial_tests():
    """Run initial tests to verify setup"""
    print("\n🧪 Running initial tests...")
    
    if Path("venv").exists():
        if os.name == 'nt':
            pytest_path = "venv\\Scripts\\pytest"
        else:
            pytest_path = "venv/bin/pytest"
        
        try:
            run_command(f"{pytest_path} tests/ -v --tb=short")
            print("✅ All tests passed!")
        except:
            print("⚠️  Some tests failed - check the output above")
    else:
        print("⚠️  Virtual environment not found - skipping tests")

def create_docker_setup():
    """Create Docker setup files"""
    print("\n🐳 Creating Docker setup...")
    
    # Create .dockerignore
    dockerignore_content = """venv/
__pycache__/
*.pyc
*.pyo
*.pyd
.env
.git/
.pytest_cache/
.coverage
.mypy_cache/
logs/
models/*.pkl
*.log
.DS_Store
"""
    
    with open(".dockerignore", "w") as f:
        f.write(dockerignore_content)
    
    # Create docker-compose for local development
    docker_compose_content = """version: '3.8'

services:
  healthcare-ml-api:
    build:
      context: .
      dockerfile: deployment/Dockerfile
    ports:
      - "8080:8080"
    environment:
      - ENVIRONMENT=development
      - LOG_LEVEL=INFO
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    depends_on:
      - mlflow-server
      - prometheus
  
  mlflow-server:
    image: python:3.9-slim
    ports:
      - "5000:5000"
    command: >
      bash -c "pip install mlflow && 
               mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db"
    volumes:
      - ./mlflow.db:/mlflow.db
      - ./mlartifacts:/mlartifacts
  
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
  
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana

volumes:
  grafana-storage:
"""
    
    with open("docker-compose.yml", "w") as f:
        f.write(docker_compose_content)
    
    print("✅ Docker configuration created")

def generate_readme():
    """Generate comprehensive README"""
    print("\n📚 Generating README...")
    
    readme_content = """# Healthcare ML MLOps Pipeline

A comprehensive MLOps pipeline for healthcare machine learning models with Random Forest algorithms for heart disease prediction and blood pressure estimation.

## 🎯 Features

- **Machine Learning Models**: Random Forest classification and regression
- **MLOps Pipeline**: Complete CI/CD pipeline with GitHub Actions
- **Google Colab Integration**: Seamless development in Colab
- **Model Monitoring**: Real-time monitoring with Prometheus and alerting
- **Data Drift Detection**: Automatic drift detection and alerts
- **API Deployment**: FastAPI-based model serving
- **Cloud Deployment**: Google Cloud Run deployment
- **Comprehensive Testing**: Unit, integration, and performance tests

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Run the setup script
python setup.py --full-setup

# Or manually:
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

### 2. Configure Environment

Update the `.env` file with your configuration:

```bash
GCP_PROJECT_ID=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
```

### 3. Run Training Pipeline

```bash
# Train models locally
python src/train_model.py

# Or in Google Colab
python src/colab_integration.py
```

### 4. Deploy Models

```bash
# Deploy to staging
python src/deploy_model.py --environment staging

# Deploy to production
python src/deploy_model.py --environment production
```

## 📁 Project Structure

```
├── .github/workflows/     # GitHub Actions CI/CD
├── config/               # Configuration files
├── deployment/           # Deployment configurations
├── src/                  # Source code
│   ├── train_model.py    # Training pipeline
│   ├── validate_model.py # Model validation
│   ├── deploy_model.py   # Deployment script
│   └── monitor_model.py  # Monitoring system
├── tests/                # Test suite
├── models/               # Trained models
└── logs/                 # Application logs
```

## 🧠 Machine Learning Models

### Heart Disease Classification
- **Algorithm**: Random Forest Classifier
- **Features**: Age, sex, chest pain type, blood pressure, cholesterol, etc.
- **Target**: Binary classification (disease/no disease)
- **Performance**: >80% accuracy threshold

### Blood Pressure Regression
- **Algorithm**: Random Forest Regressor
- **Features**: Age, BMI, exercise, lifestyle factors
- **Target**: Systolic blood pressure prediction
- **Performance**: R² > 0.70 threshold

## 🔄 MLOps Pipeline

### CI/CD Workflow
1. **Code Quality**: Linting, formatting, type checking
2. **Testing**: Unit tests, integration tests, smoke tests
3. **Training**: Automated model training and validation
4. **Deployment**: Staging and production deployment
5. **Monitoring**: Performance and drift monitoring

### Model Lifecycle
1. **Development**: Jupyter notebooks and local development
2. **Training**: Automated training with hyperparameter tuning
3. **Validation**: Performance and data quality validation
4. **Deployment**: Containerized API deployment
5. **Monitoring**: Real-time monitoring and alerting

## 🐳 Docker Deployment

### Local Development
```bash
docker-compose up -d
```

### Production Deployment
```bash
docker build -f deployment/Dockerfile -t healthcare-ml-api .
docker run -p 8080:8080 healthcare-ml-api
```

## 📊 Monitoring

### Metrics Tracked
- **Performance**: Accuracy, precision, recall, F1-score
- **Latency**: Prediction response times
- **Throughput**: Requests per second
- **Data Drift**: Feature distribution changes
- **System Health**: CPU, memory, error rates

### Alerts
- Model performance degradation
- High prediction latency
- Data drift detection
- System resource exhaustion
- API error rate spikes

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test categories
pytest tests/test_ml_pipeline.py::TestModelTraining -v
```

## 🔧 Configuration

### Model Configuration (`config/model_config.yaml`)
- Hyperparameters and tuning ranges
- Performance thresholds
- Data validation rules
- Feature engineering settings

### Deployment Configuration
- Environment-specific settings
- Resource allocation
- Scaling policies
- Security configurations

## 📈 Usage Examples

### Making Predictions

```python
import requests

# Heart disease prediction
response = requests.post(
    "https://your-api-endpoint/predict/heart-disease",
    json={
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
)

print(response.json())
```

## 🛡️ Security

- API key authentication
- HTTPS encryption
- Input validation and sanitization
- Rate limiting
- Audit logging
- HIPAA-compliant data handling

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and ensure quality checks pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

**Import Errors in Colab**
```python
!pip install scikit-learn pandas numpy matplotlib seaborn
```

**Authentication Issues**
```bash
gcloud auth application-default login
```

**Docker Build Issues**
```bash
docker system prune -a
```

### Support

- 📧 Email: support@yourcompany.com
- 📱 Slack: #ml-ops-support
- 🐛 Issues: GitHub Issues

## 🎉 Acknowledgments

- Healthcare data simulation methods
- MLOps best practices
- Open source ML libraries
- Google Colab platform

---

Built with ❤️ for healthcare AI
"""
    
    with open("README.md", "w") as f:
        f.write(readme_content)
    
    print("✅ README.md generated")

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="Healthcare ML MLOps Setup")
    parser.add_argument("--full-setup", action="store_true", 
                       help="Run complete setup including all components")
    parser.add_argument("--skip-tests", action="store_true",
                       help="Skip running initial tests")
    parser.add_argument("--skip-docker", action="store_true",
                       help="Skip Docker setup")
    
    args = parser.parse_args()
    
    print("🏥 Healthcare ML MLOps Setup")
    print("=" * 50)
    
    # Always check requirements
    check_requirements()
    
    # Setup Python environment
    setup_python_environment()
    
    # Setup configuration
    setup_configuration()
    
    if args.full_setup:
        # Setup git hooks
        setup_git_hooks()
        
        # Create Docker setup
        if not args.skip_docker:
            create_docker_setup()
        
        # Generate README
        generate_readme()
        
        # Run tests
        if not args.skip_tests:
            run_initial_tests()
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Update .env file with your configuration")
    print("2. Run 'python src/train_model.py' to train models")
    print("3. Check README.md for detailed usage instructions")
    print("4. Open random_forest_healthcare.ipynb in Google Colab to get started")

if __name__ == "__main__":
    main()