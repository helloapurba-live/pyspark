"""
==============================================================================
MLOPS COMPONENTS
==============================================================================
This module implements MLOps best practices for production deployment.

What is MLOps?
MLOps = Machine Learning Operations
It's like DevOps, but for machine learning! It includes:
1. Experiment tracking (which model version performed best?)
2. Model versioning (keep track of all models)
3. Logging (record everything that happens)
4. Monitoring (is the model still working well?)
5. Model registry (organized storage of models)

Think of it as the "infrastructure" that makes ML systems production-ready!

Author: Your Friendly AI Teacher
Date: 2025-11-13
==============================================================================
"""

import json
import os
import pickle
from datetime import datetime
from pathlib import Path
import logging
from pyspark.sql.functions import *
from pyspark.sql.types import *


class ExperimentTracker:
    """
    Tracks experiments and their results.

    Think of this as a "lab notebook" for data scientists!
    Every experiment (model training run) gets recorded with:
    - Timestamp
    - Configuration used
    - Metrics achieved
    - Artifacts produced
    """

    def __init__(self, experiment_name, base_path):
        """
        Initialize experiment tracker.

        Args:
            experiment_name: Name of the experiment
            base_path: Base path for storing experiment data
        """
        self.experiment_name = experiment_name
        self.base_path = Path(base_path)
        self.experiments = []

        # Create directory if it doesn't exist
        self.base_path.mkdir(parents=True, exist_ok=True)

        print(f"🔬 Experiment Tracker initialized: {experiment_name}")

    def log_experiment(self, run_id, config, metrics, artifacts=None):
        """
        Log an experiment run.

        This is like taking notes in your lab notebook!

        Args:
            run_id: Unique identifier for this run
            config: Configuration used
            metrics: Metrics achieved
            artifacts: Paths to saved models/data
        """
        experiment = {
            'run_id': run_id,
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'metrics': metrics,
            'artifacts': artifacts or {}
        }

        self.experiments.append(experiment)

        # Save to file
        experiment_file = self.base_path / f"{run_id}.json"
        with open(experiment_file, 'w') as f:
            json.dump(experiment, f, indent=2, default=str)

        print(f"   ✓ Experiment logged: {run_id}")

    def get_best_experiment(self, metric_name):
        """
        Get the best experiment based on a metric.

        This is like finding the "winner" of all your experiments!

        Args:
            metric_name: Name of metric to optimize

        Returns:
            Best experiment dictionary
        """
        if not self.experiments:
            return None

        best_exp = max(
            self.experiments,
            key=lambda x: self._get_nested_metric(x['metrics'], metric_name)
        )

        return best_exp

    def _get_nested_metric(self, metrics, metric_name):
        """Get a potentially nested metric value."""
        try:
            if '.' in metric_name:
                parts = metric_name.split('.')
                value = metrics
                for part in parts:
                    value = value.get(part, 0)
                return value
            return metrics.get(metric_name, 0)
        except:
            return 0

    def list_experiments(self):
        """List all experiments."""
        return self.experiments


class ModelRegistry:
    """
    Manages model versions and deployment.

    Think of this as a "library" for your models!
    - Store different versions
    - Tag models (dev, staging, production)
    - Track which model is currently deployed
    """

    def __init__(self, registry_path):
        """
        Initialize model registry.

        Args:
            registry_path: Path for model registry
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)

        self.metadata_file = self.registry_path / 'registry_metadata.json'
        self.models = self._load_metadata()

        print(f"📚 Model Registry initialized: {registry_path}")

    def register_model(self, model_name, version, model_obj, metrics, tags=None):
        """
        Register a new model version.

        This is like adding a book to the library!

        Args:
            model_name: Name of the model
            version: Version identifier
            model_obj: The actual model object (or path to it)
            metrics: Performance metrics
            tags: Tags (e.g., 'production', 'champion')
        """
        model_id = f"{model_name}_v{version}"
        model_path = self.registry_path / f"{model_id}.pkl"

        # Save model if it's an object
        if model_obj is not None:
            try:
                with open(model_path, 'wb') as f:
                    pickle.dump(model_obj, f)
            except Exception as e:
                print(f"   ⚠️  Could not serialize model: {e}")
                model_path = "not_serializable"

        # Register metadata
        model_metadata = {
            'model_name': model_name,
            'version': version,
            'model_id': model_id,
            'registered_at': datetime.now().isoformat(),
            'model_path': str(model_path),
            'metrics': metrics,
            'tags': tags or [],
            'status': 'registered'
        }

        self.models[model_id] = model_metadata
        self._save_metadata()

        print(f"   ✓ Model registered: {model_id}")
        return model_id

    def promote_model(self, model_id, stage):
        """
        Promote model to a stage (dev/staging/production).

        This is like moving a book to the "featured" section!

        Args:
            model_id: Model identifier
            stage: Stage to promote to ('dev', 'staging', 'production')
        """
        if model_id not in self.models:
            print(f"   ⚠️  Model not found: {model_id}")
            return False

        # Remove stage from other models
        for mid, metadata in self.models.items():
            if stage in metadata['tags']:
                metadata['tags'].remove(stage)

        # Add stage to this model
        if stage not in self.models[model_id]['tags']:
            self.models[model_id]['tags'].append(stage)

        self.models[model_id]['status'] = stage
        self._save_metadata()

        print(f"   ✓ Model promoted to {stage}: {model_id}")
        return True

    def get_production_model(self):
        """Get the current production model."""
        for model_id, metadata in self.models.items():
            if 'production' in metadata['tags']:
                return metadata
        return None

    def _load_metadata(self):
        """Load registry metadata from file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_metadata(self):
        """Save registry metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.models, f, indent=2, default=str)


class ModelMonitor:
    """
    Monitors model performance in production.

    Think of this as a "health monitor" for your model!
    It watches for:
    - Performance degradation
    - Data drift (is the data changing?)
    - Concept drift (are fraud patterns changing?)
    - Anomalies in predictions
    """

    def __init__(self, spark, config):
        """
        Initialize model monitor.

        Args:
            spark: PySpark session
            config: Configuration dictionary
        """
        self.spark = spark
        self.config = config
        self.baseline_stats = None

        print("📡 Model Monitor initialized!")

    def set_baseline(self, df, feature_cols):
        """
        Set baseline statistics from training data.

        This establishes what "normal" looks like!

        Args:
            df: Training data DataFrame
            feature_cols: List of feature column names
        """
        print("📊 Computing baseline statistics...")

        baseline = {}

        for col_name in feature_cols:
            if col_name in df.columns:
                stats = df.select(
                    mean(col(col_name)).alias('mean'),
                    stddev(col(col_name)).alias('stddev'),
                    min(col(col_name)).alias('min'),
                    max(col(col_name)).alias('max')
                ).collect()[0]

                baseline[col_name] = {
                    'mean': float(stats['mean']) if stats['mean'] else 0,
                    'stddev': float(stats['stddev']) if stats['stddev'] else 0,
                    'min': float(stats['min']) if stats['min'] else 0,
                    'max': float(stats['max']) if stats['max'] else 0
                }

        self.baseline_stats = baseline
        print(f"   ✓ Baseline set for {len(baseline)} features")

    def detect_data_drift(self, df, feature_cols):
        """
        Detect data drift in production data.

        Data drift = When the distribution of input data changes
        Example: If average transaction amount was $100 in training,
                but now it's $500, that's drift!

        Args:
            df: Production data DataFrame
            feature_cols: List of feature columns

        Returns:
            Dictionary with drift detection results
        """
        if self.baseline_stats is None:
            print("   ⚠️  No baseline set! Call set_baseline() first.")
            return {}

        print("🔍 Detecting data drift...")

        drift_results = {}
        threshold = self.config['mlops']['monitoring']['drift_threshold']

        for col_name in feature_cols:
            if col_name not in df.columns or col_name not in self.baseline_stats:
                continue

            # Get current statistics
            current_stats = df.select(
                mean(col(col_name)).alias('mean'),
                stddev(col(col_name)).alias('stddev')
            ).collect()[0]

            baseline = self.baseline_stats[col_name]
            current_mean = float(current_stats['mean']) if current_stats['mean'] else 0
            current_stddev = float(current_stats['stddev']) if current_stats['stddev'] else 0

            # Calculate drift score (normalized difference in means)
            if baseline['stddev'] > 0:
                drift_score = abs(current_mean - baseline['mean']) / baseline['stddev']
            else:
                drift_score = 0

            # Check if drift is significant
            has_drift = drift_score > threshold

            drift_results[col_name] = {
                'baseline_mean': baseline['mean'],
                'current_mean': current_mean,
                'drift_score': drift_score,
                'has_drift': has_drift
            }

            if has_drift:
                print(f"   ⚠️  Drift detected in {col_name}: score={drift_score:.3f}")

        # Summary
        num_drifted = sum(1 for r in drift_results.values() if r['has_drift'])
        if num_drifted > 0:
            print(f"   ⚠️  Total features with drift: {num_drifted}/{len(drift_results)}")
        else:
            print(f"   ✓ No significant drift detected")

        return drift_results

    def monitor_prediction_quality(self, predictions_df, score_col, label_col='is_suspicious'):
        """
        Monitor prediction quality on new data.

        This checks if the model is still performing well!

        Args:
            predictions_df: DataFrame with predictions
            score_col: Column with anomaly scores
            label_col: Column with true labels (if available)

        Returns:
            Dictionary with quality metrics
        """
        print("📈 Monitoring prediction quality...")

        metrics = {}

        # Distribution of predictions
        score_stats = predictions_df.select(
            mean(col(score_col)).alias('mean_score'),
            stddev(col(score_col)).alias('stddev_score'),
            min(col(score_col)).alias('min_score'),
            max(col(score_col)).alias('max_score')
        ).collect()[0]

        metrics['prediction_distribution'] = {
            'mean': float(score_stats['mean_score']),
            'stddev': float(score_stats['stddev_score']),
            'min': float(score_stats['min_score']),
            'max': float(score_stats['max_score'])
        }

        print(f"   Prediction score range: [{metrics['prediction_distribution']['min']:.3f}, "
              f"{metrics['prediction_distribution']['max']:.3f}]")
        print(f"   Mean score: {metrics['prediction_distribution']['mean']:.3f}")

        # If we have labels, calculate performance
        if label_col in predictions_df.columns:
            threshold = predictions_df.approxQuantile(score_col, [0.85], 0.01)[0]

            pred_df = predictions_df.withColumn(
                'predicted',
                when(col(score_col) > threshold, 1).otherwise(0)
            )

            tp = pred_df.filter((col('predicted') == 1) & (col(label_col) == 1)).count()
            fp = pred_df.filter((col('predicted') == 1) & (col(label_col) == 0)).count()
            tn = pred_df.filter((col('predicted') == 0) & (col(label_col) == 0)).count()
            fn = pred_df.filter((col('predicted') == 0) & (col(label_col) == 1)).count()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            metrics['performance'] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'true_positives': tp,
                'false_positives': fp
            }

            print(f"   Current F1-Score: {f1:.3f}")

        return metrics


class MLOpsLogger:
    """
    Centralized logging for MLOps.

    Think of this as the "diary" of your ML system!
    Everything gets recorded:
    - Training runs
    - Predictions
    - Errors
    - System events
    """

    def __init__(self, log_file, log_level='INFO'):
        """
        Initialize MLOps logger.

        Args:
            log_file: Path to log file
            log_level: Logging level
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )

        self.logger = logging.getLogger('AML_MLOps')
        self.logger.info("="*80)
        self.logger.info("MLOps Logger initialized")
        self.logger.info("="*80)

    def log_training_start(self, config):
        """Log the start of training."""
        self.logger.info("Training started")
        self.logger.info(f"Configuration: {json.dumps(config, indent=2, default=str)}")

    def log_training_complete(self, metrics):
        """Log training completion."""
        self.logger.info("Training completed")
        self.logger.info(f"Metrics: {json.dumps(metrics, indent=2, default=str)}")

    def log_prediction(self, num_predictions, num_anomalies):
        """Log prediction batch."""
        self.logger.info(f"Predictions made: {num_predictions}, Anomalies: {num_anomalies}")

    def log_error(self, error_msg, exception=None):
        """Log an error."""
        self.logger.error(f"Error: {error_msg}")
        if exception:
            self.logger.exception(exception)

    def log_model_deployment(self, model_id, stage):
        """Log model deployment."""
        self.logger.info(f"Model deployed: {model_id} to {stage}")

    def log_drift_detection(self, drift_results):
        """Log drift detection results."""
        num_drifted = sum(1 for r in drift_results.values() if r.get('has_drift', False))
        if num_drifted > 0:
            self.logger.warning(f"Data drift detected in {num_drifted} features")
            self.logger.warning(f"Drift details: {json.dumps(drift_results, indent=2, default=str)}")
        else:
            self.logger.info("No significant data drift detected")


def main():
    """
    Main function demonstrating MLOps components.
    """
    print("\n" + "="*80)
    print("🚀 MLOPS COMPONENTS")
    print("="*80)

    # Initialize components
    tracker = ExperimentTracker("aml_fraud_detection", "models/experiments")
    registry = ModelRegistry("models/registry")
    logger = MLOpsLogger("logs/mlops.log")

    # Example: Log an experiment
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    config = {'algorithm': 'kmeans', 'k': 10}
    metrics = {'f1_score': 0.85, 'precision': 0.82, 'recall': 0.88}

    tracker.log_experiment(run_id, config, metrics)

    # Example: Register a model
    model_id = registry.register_model(
        model_name='kmeans_fraud_detector',
        version='1.0',
        model_obj=None,  # In real use, pass actual model
        metrics=metrics,
        tags=['champion']
    )

    # Example: Promote to production
    registry.promote_model(model_id, 'production')

    # Example: Logging
    logger.log_training_complete(metrics)

    print("\n✅ MLOps components demonstration complete!")


if __name__ == "__main__":
    main()
