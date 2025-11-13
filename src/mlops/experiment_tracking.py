"""
=============================================================================
MLOPS - EXPERIMENT TRACKING AND MODEL MANAGEMENT
=============================================================================
This module handles MLOps concerns:
- Experiment tracking (log all training runs)
- Model registry (save and version models)
- Metrics logging (track performance over time)
- Artifact management (save plots, data, configs)

Think of this as your "lab notebook" - it records everything you do
so you can reproduce experiments and track what works!

Why MLOps matters:
- Remember what you tried and what worked
- Compare different models fairly
- Reproduce results months later
- Share findings with team
- Deploy the best model confidently
"""

import mlflow
import mlflow.spark
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import json
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    Manages experiment tracking using MLflow.

    MLflow is like a "flight recorder" for ML experiments - it tracks:
    - Parameters (what settings you used)
    - Metrics (how well the model performed)
    - Artifacts (model files, plots, data)
    - Code version (what code produced this model)
    """

    def __init__(self, experiment_name: str, tracking_uri: str = "file:./experiments"):
        """
        Initialize experiment tracker.

        Parameters:
        -----------
        experiment_name : str
            Name of the experiment (e.g., "AML_Fraud_Detection")
        tracking_uri : str
            Where to store experiment data
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri

        # Set MLflow tracking URI
        mlflow.set_tracking_uri(tracking_uri)

        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=f"{tracking_uri}/{experiment_name}"
            )
            logger.info(f"📝 Created new experiment: {experiment_name}")
        except:
            self.experiment = mlflow.get_experiment_by_name(experiment_name)
            self.experiment_id = self.experiment.experiment_id
            logger.info(f"📝 Using existing experiment: {experiment_name}")

        mlflow.set_experiment(experiment_name)

        self.client = MlflowClient()
        self.current_run = None

        logger.info(f"✅ Experiment tracker initialized")
        logger.info(f"   📍 Tracking URI: {tracking_uri}")
        logger.info(f"   🆔 Experiment ID: {self.experiment_id}")

    def start_run(self, run_name: str, tags: Optional[Dict] = None) -> str:
        """
        Start a new training run.

        A "run" is one training attempt - like one experiment in a lab.

        Parameters:
        -----------
        run_name : str
            Name for this run (e.g., "random_forest_v1")
        tags : Dict, optional
            Tags to organize runs (e.g., {"model_type": "tree", "version": "1.0"})

        Returns:
        --------
        str : Run ID
        """
        # Add timestamp to run name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_run_name = f"{run_name}_{timestamp}"

        # Start MLflow run
        self.current_run = mlflow.start_run(run_name=full_run_name, tags=tags or {})

        logger.info(f"🚀 Started run: {full_run_name}")
        logger.info(f"   🆔 Run ID: {self.current_run.info.run_id}")

        return self.current_run.info.run_id

    def log_params(self, params: Dict[str, Any]):
        """
        Log hyperparameters.

        Parameters are the "settings" you used for training.
        Example: learning_rate=0.01, num_trees=100

        Parameters:
        -----------
        params : Dict
            Dictionary of parameter name -> value
        """
        logger.info("📋 Logging parameters...")

        for param_name, param_value in params.items():
            # MLflow requires string values
            mlflow.log_param(param_name, str(param_value))

        logger.info(f"   ✓ Logged {len(params)} parameters")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log performance metrics.

        Metrics are the "scores" that tell you how well your model performs.
        Example: accuracy=0.95, f1_score=0.87

        Parameters:
        -----------
        metrics : Dict
            Dictionary of metric name -> value
        step : int, optional
            Step number (useful for tracking over training epochs)
        """
        logger.info("📊 Logging metrics...")

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value, step=step)

        logger.info(f"   ✓ Logged {len(metrics)} metrics")

        # Print metrics for user to see
        for metric_name, metric_value in metrics.items():
            logger.info(f"      {metric_name}: {metric_value:.4f}")

    def log_model(self, model: Any, model_name: str, signature=None):
        """
        Log trained model.

        This saves the model so you can use it later!

        Parameters:
        -----------
        model : Any
            Trained model object
        model_name : str
            Name to save model as
        signature : ModelSignature, optional
            Input/output signature
        """
        logger.info(f"💾 Logging model: {model_name}...")

        try:
            # Log Spark ML model
            mlflow.spark.log_model(
                model,
                model_name,
                signature=signature
            )
            logger.info(f"   ✅ Model logged successfully")

        except Exception as e:
            logger.warning(f"   ⚠️  Could not log model: {str(e)}")

    def log_artifact(self, artifact_path: str, artifact_name: Optional[str] = None):
        """
        Log artifact (file).

        Artifacts are additional files like plots, datasets, configs.

        Parameters:
        -----------
        artifact_path : str
            Path to file to log
        artifact_name : str, optional
            Name to save as (default: use original filename)
        """
        logger.info(f"📎 Logging artifact: {artifact_path}...")

        mlflow.log_artifact(artifact_path, artifact_name)

        logger.info(f"   ✅ Artifact logged")

    def log_dict(self, dictionary: Dict, filename: str):
        """
        Log dictionary as JSON artifact.

        Useful for saving configurations, results, etc.

        Parameters:
        -----------
        dictionary : Dict
            Dictionary to save
        filename : str
            Filename to save as (e.g., "config.json")
        """
        logger.info(f"📝 Logging dictionary as {filename}...")

        # Save to temp file
        temp_path = f"/tmp/{filename}"
        with open(temp_path, 'w') as f:
            json.dump(dictionary, f, indent=2)

        # Log as artifact
        mlflow.log_artifact(temp_path)

        logger.info(f"   ✅ Dictionary logged")

    def log_figure(self, figure, filename: str):
        """
        Log matplotlib/plotly figure.

        Parameters:
        -----------
        figure : Figure
            Matplotlib or Plotly figure
        filename : str
            Filename to save as
        """
        logger.info(f"📊 Logging figure: {filename}...")

        mlflow.log_figure(figure, filename)

        logger.info(f"   ✅ Figure logged")

    def end_run(self, status: str = "FINISHED"):
        """
        End the current run.

        Parameters:
        -----------
        status : str
            Status of run: FINISHED, FAILED, or KILLED
        """
        if self.current_run:
            mlflow.end_run(status)
            logger.info(f"🏁 Run ended with status: {status}")
            self.current_run = None
        else:
            logger.warning("⚠️  No active run to end")

    def get_best_run(self, metric_name: str, mode: str = "max") -> Dict:
        """
        Get the best run based on a metric.

        This helps you find your best model!

        Parameters:
        -----------
        metric_name : str
            Metric to optimize (e.g., "f1_score")
        mode : str
            "max" to maximize, "min" to minimize

        Returns:
        --------
        Dict : Best run information
        """
        logger.info(f"🔍 Finding best run by {metric_name} ({mode})...")

        # Search all runs
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=[f"metrics.{metric_name} {'DESC' if mode == 'max' else 'ASC'}"],
            max_results=1
        )

        if not runs:
            logger.warning("⚠️  No runs found")
            return None

        best_run = runs[0]

        logger.info(f"   ✅ Best run: {best_run.info.run_name}")
        logger.info(f"   📊 {metric_name}: {best_run.data.metrics.get(metric_name, 'N/A')}")

        return {
            'run_id': best_run.info.run_id,
            'run_name': best_run.info.run_name,
            'metrics': best_run.data.metrics,
            'params': best_run.data.params
        }

    def compare_runs(self, run_ids: List[str], metrics: List[str]) -> pd.DataFrame:
        """
        Compare multiple runs.

        Parameters:
        -----------
        run_ids : List[str]
            List of run IDs to compare
        metrics : List[str]
            Metrics to compare

        Returns:
        --------
        DataFrame : Comparison table
        """
        import pandas as pd

        logger.info(f"📊 Comparing {len(run_ids)} runs...")

        comparison_data = []

        for run_id in run_ids:
            run = self.client.get_run(run_id)

            row = {
                'run_id': run_id,
                'run_name': run.info.run_name,
                'start_time': datetime.fromtimestamp(run.info.start_time / 1000)
            }

            # Add metrics
            for metric in metrics:
                row[metric] = run.data.metrics.get(metric, None)

            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)

        logger.info(f"   ✅ Comparison complete")

        return df


class ModelRegistry:
    """
    Manages model versioning and deployment.

    Think of this as a "library" for your models - you can:
    - Save models with version numbers
    - Tag models for different stages (staging, production)
    - Load specific versions later
    - Track model lineage (what data/code created this model)
    """

    def __init__(self, registry_path: str = "./model_registry"):
        """
        Initialize model registry.

        Parameters:
        -----------
        registry_path : str
            Where to store model files
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)

        self.metadata_file = self.registry_path / "registry_metadata.json"
        self.metadata = self._load_metadata()

        logger.info(f"📚 Model Registry initialized")
        logger.info(f"   📍 Registry path: {registry_path}")

    def _load_metadata(self) -> Dict:
        """Load registry metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"models": {}}

    def _save_metadata(self):
        """Save registry metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def register_model(self, model: Any, model_name: str, version: str,
                      metrics: Dict, params: Dict, tags: Optional[Dict] = None):
        """
        Register a new model version.

        Parameters:
        -----------
        model : Any
            Trained model
        model_name : str
            Name of model (e.g., "random_forest_fraud_detector")
        version : str
            Version number (e.g., "1.0.0")
        metrics : Dict
            Performance metrics
        params : Dict
            Hyperparameters
        tags : Dict, optional
            Additional tags
        """
        logger.info(f"📝 Registering model: {model_name} v{version}...")

        # Create model directory
        model_dir = self.registry_path / model_name / version
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = model_dir / "model"
        try:
            model.save(str(model_path))
            logger.info(f"   💾 Model saved to {model_path}")
        except Exception as e:
            logger.error(f"   ❌ Error saving model: {str(e)}")
            return

        # Save metadata
        model_metadata = {
            'version': version,
            'registered_at': datetime.now().isoformat(),
            'metrics': metrics,
            'params': params,
            'tags': tags or {},
            'path': str(model_path),
            'stage': 'none'  # none, staging, production
        }

        # Update registry metadata
        if model_name not in self.metadata['models']:
            self.metadata['models'][model_name] = {'versions': {}}

        self.metadata['models'][model_name]['versions'][version] = model_metadata

        self._save_metadata()

        logger.info(f"   ✅ Model registered successfully")

    def load_model(self, model_name: str, version: str = "latest"):
        """
        Load a model from registry.

        Parameters:
        -----------
        model_name : str
            Name of model
        version : str
            Version to load ("latest", "production", or specific version)

        Returns:
        --------
        Loaded model
        """
        logger.info(f"📦 Loading model: {model_name} ({version})...")

        if model_name not in self.metadata['models']:
            logger.error(f"   ❌ Model {model_name} not found in registry")
            return None

        versions = self.metadata['models'][model_name]['versions']

        if version == "latest":
            # Get most recent version
            version = max(versions.keys(), key=lambda v: versions[v]['registered_at'])
        elif version == "production":
            # Get production version
            prod_versions = [v for v, meta in versions.items() if meta['stage'] == 'production']
            if not prod_versions:
                logger.error(f"   ❌ No production version found")
                return None
            version = prod_versions[0]

        if version not in versions:
            logger.error(f"   ❌ Version {version} not found")
            return None

        model_path = versions[version]['path']

        try:
            from pyspark.ml import PipelineModel
            model = PipelineModel.load(model_path)
            logger.info(f"   ✅ Model loaded from {model_path}")
            return model
        except Exception as e:
            logger.error(f"   ❌ Error loading model: {str(e)}")
            return None

    def promote_model(self, model_name: str, version: str, stage: str):
        """
        Promote model to a stage (staging/production).

        Parameters:
        -----------
        model_name : str
            Name of model
        version : str
            Version to promote
        stage : str
            Stage to promote to ('staging' or 'production')
        """
        logger.info(f"🚀 Promoting {model_name} v{version} to {stage}...")

        if model_name not in self.metadata['models']:
            logger.error(f"   ❌ Model {model_name} not found")
            return

        versions = self.metadata['models'][model_name]['versions']

        if version not in versions:
            logger.error(f"   ❌ Version {version} not found")
            return

        # Demote any existing model at this stage
        for v, meta in versions.items():
            if meta['stage'] == stage:
                meta['stage'] = 'archived'
                logger.info(f"   📦 Archived previous {stage} version: {v}")

        # Promote new version
        versions[version]['stage'] = stage
        versions[version]['promoted_at'] = datetime.now().isoformat()

        self._save_metadata()

        logger.info(f"   ✅ Model promoted to {stage}")

    def list_models(self) -> Dict:
        """
        List all registered models.

        Returns:
        --------
        Dict : All models and versions
        """
        return self.metadata['models']


def main():
    """
    Test the MLOps module.
    """
    print("\n" + "="*80)
    print("🔬 TESTING MLOPS - EXPERIMENT TRACKING")
    print("="*80 + "\n")

    # Initialize tracker
    tracker = ExperimentTracker("AML_Fraud_Detection_Test")

    # Start a run
    tracker.start_run("test_run", tags={"model": "test", "version": "1.0"})

    # Log parameters
    tracker.log_params({
        "learning_rate": 0.01,
        "num_trees": 100,
        "max_depth": 10
    })

    # Log metrics
    tracker.log_metrics({
        "accuracy": 0.95,
        "f1_score": 0.87,
        "auc": 0.92
    })

    # End run
    tracker.end_run()

    logger.info("\n✅ MLOps test complete!")


if __name__ == "__main__":
    import pandas as pd
    from typing import List
    main()
