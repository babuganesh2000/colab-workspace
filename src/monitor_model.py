"""
Monitoring and alerting system for healthcare ML models
"""

import time
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import requests
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from google.cloud import monitoring_v3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Prometheus metrics
PREDICTION_COUNTER = Counter('ml_predictions_total', 'Total predictions made', ['model_type', 'outcome'])
PREDICTION_LATENCY = Histogram('ml_prediction_duration_seconds', 'Prediction latency', ['model_type'])
MODEL_ACCURACY = Gauge('ml_model_accuracy', 'Model accuracy', ['model_type'])
DATA_DRIFT_SCORE = Gauge('ml_data_drift_score', 'Data drift score', ['feature'])
ERROR_COUNTER = Counter('ml_errors_total', 'Total errors', ['error_type'])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthcareMLMonitor:
    """Monitor healthcare ML models in production"""
    
    def __init__(self, config_path: str = "config/monitoring_config.json"):
        """Initialize monitoring system"""
        self.config = self._load_config(config_path)
        self.metrics_history = []
        self.alert_thresholds = self.config.get('alert_thresholds', {})
        
        # Initialize cloud monitoring client
        if self.config.get('enable_cloud_monitoring', False):
            self.cloud_client = monitoring_v3.MetricServiceClient()
            self.project_name = f"projects/{self.config['gcp_project']}"
        
        # Start Prometheus metrics server
        if self.config.get('enable_prometheus', True):
            start_http_server(self.config.get('prometheus_port', 8000))
            logger.info("Prometheus metrics server started on port 8000")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load monitoring configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default monitoring configuration"""
        return {
            "enable_prometheus": True,
            "enable_cloud_monitoring": False,
            "prometheus_port": 8000,
            "alert_thresholds": {
                "accuracy_drop": 0.05,
                "latency_threshold": 1.0,
                "error_rate_threshold": 0.01,
                "drift_threshold": 0.1
            },
            "alert_channels": {
                "email": {
                    "enabled": False,
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "sender_email": "",
                    "sender_password": "",
                    "recipients": []
                },
                "slack": {
                    "enabled": False,
                    "webhook_url": ""
                }
            }
        }
    
    def track_prediction(self, model_type: str, prediction_time: float, 
                        outcome: str = "success", error_type: str = None):
        """Track prediction metrics"""
        # Update Prometheus metrics
        PREDICTION_COUNTER.labels(model_type=model_type, outcome=outcome).inc()
        PREDICTION_LATENCY.labels(model_type=model_type).observe(prediction_time)
        
        if error_type:
            ERROR_COUNTER.labels(error_type=error_type).inc()
        
        # Store in history
        metric = {
            'timestamp': datetime.now().isoformat(),
            'model_type': model_type,
            'prediction_time': prediction_time,
            'outcome': outcome,
            'error_type': error_type
        }
        self.metrics_history.append(metric)
        
        # Check thresholds
        self._check_latency_threshold(model_type, prediction_time)
    
    def track_model_accuracy(self, model_type: str, accuracy: float):
        """Track model accuracy"""
        MODEL_ACCURACY.labels(model_type=model_type).set(accuracy)
        
        # Check for accuracy drop
        self._check_accuracy_threshold(model_type, accuracy)
    
    def track_data_drift(self, feature_name: str, drift_score: float):
        """Track data drift for features"""
        DATA_DRIFT_SCORE.labels(feature=feature_name).set(drift_score)
        
        # Check drift threshold
        if drift_score > self.alert_thresholds.get('drift_threshold', 0.1):
            self._send_alert(
                title=f"Data Drift Alert: {feature_name}",
                message=f"Feature {feature_name} has drift score {drift_score:.3f}, "
                       f"exceeding threshold {self.alert_thresholds['drift_threshold']}"
            )
    
    def _check_latency_threshold(self, model_type: str, prediction_time: float):
        """Check if prediction latency exceeds threshold"""
        threshold = self.alert_thresholds.get('latency_threshold', 1.0)
        
        if prediction_time > threshold:
            self._send_alert(
                title=f"High Latency Alert: {model_type}",
                message=f"Prediction latency {prediction_time:.3f}s exceeds threshold {threshold}s"
            )
    
    def _check_accuracy_threshold(self, model_type: str, current_accuracy: float):
        """Check for accuracy degradation"""
        # Get recent accuracy values
        recent_accuracies = [
            m for m in self.metrics_history[-100:] 
            if m.get('model_type') == model_type and 'accuracy' in m
        ]
        
        if len(recent_accuracies) > 5:
            baseline_accuracy = np.mean([m['accuracy'] for m in recent_accuracies[-20:-5]])
            accuracy_drop = baseline_accuracy - current_accuracy
            
            threshold = self.alert_thresholds.get('accuracy_drop', 0.05)
            
            if accuracy_drop > threshold:
                self._send_alert(
                    title=f"Accuracy Drop Alert: {model_type}",
                    message=f"Accuracy dropped by {accuracy_drop:.3f} "
                           f"(from {baseline_accuracy:.3f} to {current_accuracy:.3f})"
                )
    
    def calculate_error_rate(self, model_type: str, time_window_minutes: int = 60) -> float:
        """Calculate error rate over time window"""
        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
        
        recent_metrics = [
            m for m in self.metrics_history
            if (datetime.fromisoformat(m['timestamp']) > cutoff_time and 
                m.get('model_type') == model_type)
        ]
        
        if not recent_metrics:
            return 0.0
        
        error_count = sum(1 for m in recent_metrics if m.get('outcome') == 'error')
        total_count = len(recent_metrics)
        
        error_rate = error_count / total_count
        
        # Check error rate threshold
        threshold = self.alert_thresholds.get('error_rate_threshold', 0.01)
        if error_rate > threshold:
            self._send_alert(
                title=f"High Error Rate Alert: {model_type}",
                message=f"Error rate {error_rate:.3f} exceeds threshold {threshold} "
                       f"({error_count}/{total_count} over {time_window_minutes} minutes)"
            )
        
        return error_rate
    
    def detect_data_drift(self, reference_data: pd.DataFrame, 
                         current_data: pd.DataFrame) -> Dict[str, float]:
        """Detect data drift using statistical tests"""
        from scipy import stats
        
        drift_scores = {}
        
        for column in reference_data.columns:
            if column in current_data.columns:
                # Use Kolmogorov-Smirnov test for drift detection
                ks_statistic, p_value = stats.ks_2samp(
                    reference_data[column].dropna(),
                    current_data[column].dropna()
                )
                
                # Higher KS statistic indicates more drift
                drift_scores[column] = ks_statistic
                
                # Track drift metric
                self.track_data_drift(column, ks_statistic)
        
        return drift_scores
    
    def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {},
            'metrics': {},
            'alerts': []
        }
        
        # Calculate summary statistics
        recent_metrics = self.metrics_history[-1000:]  # Last 1000 predictions
        
        if recent_metrics:
            # Success rate
            success_count = sum(1 for m in recent_metrics if m.get('outcome') == 'success')
            total_count = len(recent_metrics)
            success_rate = success_count / total_count if total_count > 0 else 0
            
            # Average latency by model
            latencies_by_model = {}
            for metric in recent_metrics:
                model_type = metric.get('model_type')
                if model_type and 'prediction_time' in metric:
                    if model_type not in latencies_by_model:
                        latencies_by_model[model_type] = []
                    latencies_by_model[model_type].append(metric['prediction_time'])
            
            avg_latencies = {
                model: np.mean(times) for model, times in latencies_by_model.items()
            }
            
            report['summary'] = {
                'total_predictions': total_count,
                'success_rate': success_rate,
                'error_rate': 1 - success_rate,
                'average_latencies': avg_latencies
            }
        
        # Add current metric values
        report['metrics'] = {
            'classification_accuracy': self._get_current_accuracy('classification'),
            'regression_accuracy': self._get_current_accuracy('regression'),
            'drift_scores': self._get_current_drift_scores()
        }
        
        return report
    
    def _get_current_accuracy(self, model_type: str) -> Optional[float]:
        """Get current accuracy for model type"""
        recent_metrics = [
            m for m in self.metrics_history[-50:]
            if m.get('model_type') == model_type and 'accuracy' in m
        ]
        
        if recent_metrics:
            return recent_metrics[-1]['accuracy']
        return None
    
    def _get_current_drift_scores(self) -> Dict[str, float]:
        """Get current drift scores"""
        drift_scores = {}
        
        # This would typically come from your drift detection pipeline
        # For now, return empty dict
        return drift_scores
    
    def _send_alert(self, title: str, message: str):
        """Send alert through configured channels"""
        logger.warning(f"ALERT: {title} - {message}")
        
        # Email alerts
        if self.config['alert_channels']['email']['enabled']:
            self._send_email_alert(title, message)
        
        # Slack alerts
        if self.config['alert_channels']['slack']['enabled']:
            self._send_slack_alert(title, message)
    
    def _send_email_alert(self, title: str, message: str):
        """Send email alert"""
        try:
            email_config = self.config['alert_channels']['email']
            
            msg = MIMEMultipart()
            msg['From'] = email_config['sender_email']
            msg['To'] = ', '.join(email_config['recipients'])
            msg['Subject'] = f"[Healthcare ML Alert] {title}"
            
            body = f"""
Healthcare ML Monitoring Alert

Title: {title}
Time: {datetime.now().isoformat()}
Message: {message}

This is an automated alert from the Healthcare ML monitoring system.
"""
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['sender_email'], email_config['sender_password'])
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent: {title}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def _send_slack_alert(self, title: str, message: str):
        """Send Slack alert"""
        try:
            webhook_url = self.config['alert_channels']['slack']['webhook_url']
            
            payload = {
                "text": f"🚨 Healthcare ML Alert: {title}",
                "attachments": [
                    {
                        "color": "danger",
                        "fields": [
                            {
                                "title": "Message",
                                "value": message,
                                "short": False
                            },
                            {
                                "title": "Time",
                                "value": datetime.now().isoformat(),
                                "short": True
                            }
                        ]
                    }
                ]
            }
            
            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()
            
            logger.info(f"Slack alert sent: {title}")
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")

# Example usage and integration
class ModelMonitoringMiddleware:
    """Middleware for FastAPI to integrate monitoring"""
    
    def __init__(self, monitor: HealthcareMLMonitor):
        self.monitor = monitor
    
    async def __call__(self, request, call_next):
        """Process request with monitoring"""
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # Track successful prediction
            prediction_time = time.time() - start_time
            model_type = self._extract_model_type(request.url.path)
            
            self.monitor.track_prediction(
                model_type=model_type,
                prediction_time=prediction_time,
                outcome="success"
            )
            
            return response
            
        except Exception as e:
            # Track error
            prediction_time = time.time() - start_time
            model_type = self._extract_model_type(request.url.path)
            
            self.monitor.track_prediction(
                model_type=model_type,
                prediction_time=prediction_time,
                outcome="error",
                error_type=type(e).__name__
            )
            
            raise
    
    def _extract_model_type(self, path: str) -> str:
        """Extract model type from request path"""
        if "heart-disease" in path:
            return "classification"
        elif "blood-pressure" in path:
            return "regression"
        else:
            return "unknown"

if __name__ == "__main__":
    # Example usage
    monitor = HealthcareMLMonitor()
    
    # Simulate some predictions
    for i in range(100):
        monitor.track_prediction(
            model_type="classification",
            prediction_time=np.random.normal(0.1, 0.02),
            outcome="success"
        )
    
    # Generate health report
    report = monitor.generate_health_report()
    print(json.dumps(report, indent=2))