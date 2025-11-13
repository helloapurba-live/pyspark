"""
==============================================================================
END-TO-END AML FRAUD DETECTION PIPELINE
==============================================================================
This is the MAIN PIPELINE that orchestrates everything!

Think of this as the "conductor" of an orchestra - it coordinates all the
different components to work together in harmony.

Pipeline Flow:
1. Initialize Spark and load config
2. Generate synthetic banking data (or load existing)
3. Engineer features from raw data
4. Train multiple clustering algorithms
5. Evaluate and compare all models
6. Select best model
7. Deploy to production (model registry)
8. Monitor performance
9. Generate reports

Author: Your Friendly AI Teacher
Date: 2025-11-13
==============================================================================
"""

import yaml
import sys
from pathlib import Path
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

# Import our custom modules
from data_generator import BankingDataGenerator
from feature_engineering import FeatureEngineer
from clustering_algorithms import ClusteringAlgorithms
from model_evaluation import ModelEvaluator
from mlops_components import (
    ExperimentTracker,
    ModelRegistry,
    ModelMonitor,
    MLOpsLogger
)


class AMLFraudDetectionPipeline:
    """
    End-to-end pipeline for AML fraud detection.

    This is the "master controller" that runs the entire workflow!
    """

    def __init__(self, config_path='../configs/config.yaml'):
        """
        Initialize the pipeline.

        Args:
            config_path: Path to configuration file
        """
        print("\n" + "="*80)
        print("🏦 AML FRAUD DETECTION PIPELINE")
        print("="*80)
        print(f"🚀 Initializing pipeline...")

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        print(f"   ✓ Configuration loaded from {config_path}")

        # Initialize Spark
        self.spark = self._initialize_spark()

        # Initialize MLOps components
        self.run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_tracker = ExperimentTracker(
            self.config['mlops']['experiment_tracking']['experiment_name'],
            "../models/experiments"
        )
        self.model_registry = ModelRegistry("../models/registry")
        self.logger = MLOpsLogger(
            f"../{self.config['mlops']['logging']['log_file']}",
            self.config['mlops']['logging']['level']
        )
        self.monitor = ModelMonitor(self.spark, self.config)

        print(f"   ✓ MLOps components initialized")
        print(f"   ✓ Run ID: {self.run_id}")
        print("\n" + "="*80)

    def _initialize_spark(self):
        """
        Initialize Spark session with optimized configuration.

        Spark is like the "engine" that powers our big data processing!
        """
        print("   ⚙️  Initializing Spark...")

        spark_config = self.config['spark']

        builder = SparkSession.builder \
            .appName(spark_config['app_name'])

        # Add configurations
        for key, value in spark_config['config'].items():
            builder = builder.config(key, value)

        spark = builder.getOrCreate()
        spark.sparkContext.setLogLevel("WARN")  # Reduce log noise

        print(f"      Spark version: {spark.version}")
        print(f"      Master: {spark_config['master']}")

        return spark

    def run_full_pipeline(self):
        """
        Run the complete end-to-end pipeline.

        This is where the magic happens! 🎩✨
        """
        try:
            self.logger.log_training_start(self.config)

            # Stage 1: Data Generation
            print("\n" + "="*80)
            print("📊 STAGE 1: DATA GENERATION")
            print("="*80)
            raw_data = self._run_data_generation()

            # Stage 2: Feature Engineering
            print("\n" + "="*80)
            print("🔧 STAGE 2: FEATURE ENGINEERING")
            print("="*80)
            feature_data, feature_pipeline = self._run_feature_engineering(raw_data)

            # Stage 3: Model Training
            print("\n" + "="*80)
            print("🤖 STAGE 3: MODEL TRAINING")
            print("="*80)
            clustering_results = self._run_clustering(feature_data)

            # Stage 4: Model Evaluation
            print("\n" + "="*80)
            print("📊 STAGE 4: MODEL EVALUATION")
            print("="*80)
            evaluation_results = self._run_evaluation(clustering_results)

            # Stage 5: Model Selection and Registration
            print("\n" + "="*80)
            print("🏆 STAGE 5: MODEL SELECTION")
            print("="*80)
            best_model = self._select_and_register_best_model(
                clustering_results,
                evaluation_results
            )

            # Stage 6: Monitoring Setup
            print("\n" + "="*80)
            print("📡 STAGE 6: MONITORING SETUP")
            print("="*80)
            self._setup_monitoring(feature_data)

            # Stage 7: Generate Reports
            print("\n" + "="*80)
            print("📄 STAGE 7: GENERATE REPORTS")
            print("="*80)
            self._generate_reports(evaluation_results)

            # Log success
            self.logger.log_training_complete(evaluation_results)

            # Pipeline complete!
            print("\n" + "="*80)
            print("✅ PIPELINE COMPLETE!")
            print("="*80)
            self._print_summary(best_model, evaluation_results)

            return {
                'success': True,
                'run_id': self.run_id,
                'best_model': best_model,
                'evaluation_results': evaluation_results
            }

        except Exception as e:
            self.logger.log_error("Pipeline failed", e)
            print(f"\n❌ Pipeline failed: {str(e)}")
            raise

        finally:
            # Always stop Spark
            self.spark.stop()

    def _run_data_generation(self):
        """
        Stage 1: Generate or load banking transaction data.
        """
        data_path = f"../{self.config['data']['raw_data_path']}"

        # Check if data already exists
        if Path(data_path).exists():
            print("📂 Loading existing data...")
            raw_data = self.spark.read.parquet(data_path)
            print(f"   ✓ Loaded {raw_data.count()} transactions")
        else:
            print("🎲 Generating new synthetic data...")
            generator = BankingDataGenerator(self.spark, self.config)
            raw_data = generator.generate_transaction_data()
            generator.save_data(raw_data, data_path)

        # Show sample
        print("\n📋 Sample of raw data:")
        raw_data.show(5)

        return raw_data

    def _run_feature_engineering(self, raw_data):
        """
        Stage 2: Engineer features from raw data.
        """
        engineer = FeatureEngineer(self.spark, self.config)

        # Engineer features
        feature_data = engineer.engineer_features(raw_data)

        # Prepare for clustering
        final_data, pipeline = engineer.prepare_for_clustering(feature_data)

        # Cache for performance (we'll use this data multiple times)
        if self.config['spark']['optimization']['cache_intermediate']:
            print("\n💾 Caching feature data for performance...")
            final_data.cache()
            print("   ✓ Data cached")

        # Save features
        features_path = f"../{self.config['data']['features_path']}"
        engineer.save_features(final_data, features_path)

        # Show sample
        print("\n📋 Sample of engineered features:")
        final_data.select('transaction_id', 'amount', 'composite_risk_score', 'is_suspicious').show(5)

        return final_data, pipeline

    def _run_clustering(self, feature_data):
        """
        Stage 3: Train multiple clustering algorithms.
        """
        clustering = ClusteringAlgorithms(self.spark, self.config)
        results = clustering.train_all_algorithms(feature_data)

        print(f"\n✅ Trained {len(results)} algorithms successfully!")

        return results

    def _run_evaluation(self, clustering_results):
        """
        Stage 4: Evaluate and compare all models.
        """
        evaluator = ModelEvaluator(self.spark, self.config)
        evaluation_results = evaluator.evaluate_all_models(clustering_results)

        # Save evaluation results
        results_path = "../results/evaluation_results.json"
        Path("../results").mkdir(parents=True, exist_ok=True)
        evaluator.save_evaluation_results(results_path)

        # Generate evaluation report
        report_path = "../results/evaluation_report.txt"
        evaluator.generate_evaluation_report(report_path)

        return evaluation_results

    def _select_and_register_best_model(self, clustering_results, evaluation_results):
        """
        Stage 5: Select best model and register it.
        """
        print("\n🔍 Selecting best model...")

        # Find best model based on F1-score
        best_algo = None
        best_f1 = 0

        for algo_name, metrics in evaluation_results.items():
            fraud_metrics = metrics.get('fraud_detection', {})
            f1 = fraud_metrics.get('f1_score', 0)

            if f1 > best_f1:
                best_f1 = f1
                best_algo = algo_name

        if best_algo is None:
            print("   ⚠️  Could not determine best model")
            return None

        print(f"   🏆 Best model: {best_algo}")
        print(f"   📊 F1-Score: {best_f1:.4f}")

        # Register model
        model_obj = clustering_results[best_algo].get('model')
        model_metrics = evaluation_results[best_algo]

        model_id = self.model_registry.register_model(
            model_name=f'aml_fraud_{best_algo}',
            version=self.run_id,
            model_obj=model_obj,
            metrics=model_metrics,
            tags=['champion', 'latest']
        )

        # Promote to production (in a real system, you'd test in staging first!)
        self.model_registry.promote_model(model_id, 'production')
        self.logger.log_model_deployment(model_id, 'production')

        # Log experiment
        self.experiment_tracker.log_experiment(
            run_id=self.run_id,
            config=self.config,
            metrics=model_metrics,
            artifacts={'model_id': model_id}
        )

        return {
            'algorithm': best_algo,
            'model_id': model_id,
            'f1_score': best_f1,
            'metrics': model_metrics
        }

    def _setup_monitoring(self, feature_data):
        """
        Stage 6: Set up monitoring for production.
        """
        print("\n📊 Setting up monitoring baseline...")

        # Set baseline from training data
        numerical_features = [
            'amount', 'transaction_hour', 'customer_avg_amount',
            'txn_count_24h', 'composite_risk_score'
        ]

        self.monitor.set_baseline(feature_data, numerical_features)

        # Simulate checking for drift (on same data, should show no drift)
        print("\n🔍 Running drift detection test...")
        drift_results = self.monitor.detect_data_drift(feature_data, numerical_features)

        self.logger.log_drift_detection(drift_results)

        print("   ✓ Monitoring setup complete")

    def _generate_reports(self, evaluation_results):
        """
        Stage 7: Generate comprehensive reports.
        """
        print("\n📄 Generating reports...")

        # Summary report
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("AML FRAUD DETECTION - EXECUTIVE SUMMARY")
        report_lines.append("="*80)
        report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Run ID: {self.run_id}\n")

        report_lines.append("\n📊 ALGORITHMS EVALUATED:")
        report_lines.append("-"*80)

        for algo_name, metrics in evaluation_results.items():
            fraud_metrics = metrics.get('fraud_detection', {})
            report_lines.append(f"\n{algo_name.upper()}:")
            report_lines.append(f"  F1-Score: {fraud_metrics.get('f1_score', 0):.4f}")
            report_lines.append(f"  Precision: {fraud_metrics.get('precision', 0):.4f}")
            report_lines.append(f"  Recall: {fraud_metrics.get('recall', 0):.4f}")

        # Write report
        report_path = Path("../results/executive_summary.txt")
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))

        print(f"   ✓ Executive summary saved to: {report_path}")

        # Also print to console
        print("\n" + '\n'.join(report_lines))

    def _print_summary(self, best_model, evaluation_results):
        """
        Print pipeline summary.
        """
        print(f"\n📌 PIPELINE SUMMARY:")
        print(f"   Run ID: {self.run_id}")
        print(f"   Data processed: {self.config['data']['num_transactions']} transactions")
        print(f"   Algorithms trained: {len(evaluation_results)}")

        if best_model:
            print(f"\n🏆 PRODUCTION MODEL:")
            print(f"   Algorithm: {best_model['algorithm']}")
            print(f"   Model ID: {best_model['model_id']}")
            print(f"   F1-Score: {best_model['f1_score']:.4f}")

        print(f"\n📂 OUTPUT FILES:")
        print(f"   Raw data: {self.config['data']['raw_data_path']}")
        print(f"   Features: {self.config['data']['features_path']}")
        print(f"   Evaluation: results/evaluation_results.json")
        print(f"   Report: results/evaluation_report.txt")
        print(f"   Logs: {self.config['mlops']['logging']['log_file']}")

        print("\n" + "="*80)
        print("🎉 Thank you for using the AML Fraud Detection Pipeline!")
        print("="*80)


def main():
    """
    Main entry point for the pipeline.
    """
    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                                                                          ║
    ║            🏦 AML FRAUD DETECTION PIPELINE 🏦                           ║
    ║                                                                          ║
    ║  A comprehensive PySpark 4 solution for Anti-Money Laundering           ║
    ║  fraud detection using 12+ clustering algorithms and MLOps best         ║
    ║  practices.                                                              ║
    ║                                                                          ║
    ║  Features:                                                               ║
    ║  ✓ Synthetic banking transaction data generation                        ║
    ║  ✓ Mixed data processing (tabular + text)                               ║
    ║  ✓ 12+ clustering & anomaly detection algorithms                        ║
    ║  ✓ Comprehensive feature engineering                                    ║
    ║  ✓ Model evaluation & comparison                                        ║
    ║  ✓ Full MLOps pipeline (tracking, versioning, monitoring)               ║
    ║  ✓ Production-ready deployment                                          ║
    ║                                                                          ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)

    # Check if config path provided
    config_path = sys.argv[1] if len(sys.argv) > 1 else '../configs/config.yaml'

    # Initialize and run pipeline
    pipeline = AMLFraudDetectionPipeline(config_path)
    results = pipeline.run_full_pipeline()

    # Exit with success code
    sys.exit(0 if results['success'] else 1)


if __name__ == "__main__":
    main()
