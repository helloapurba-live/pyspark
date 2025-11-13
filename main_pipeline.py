#!/usr/bin/env python3
"""
=============================================================================
AML FRAUD DETECTION - END-TO-END MLOPS PIPELINE
=============================================================================
This is the MAIN PIPELINE that orchestrates everything!

Think of this as the "conductor" of an orchestra - it coordinates all
the different components to create a complete ML system.

WHAT THIS PIPELINE DOES:
1. 📊 Generates realistic banking transaction data
2. 🔧 Engineers features (tabular + text)
3. 🤖 Trains 10+ different ML algorithms
4. 📈 Evaluates and compares all models
5. 🏆 Selects the best model
6. 💾 Saves and versions the best model
7. 📝 Logs everything for reproducibility
8. 🚀 Prepares for deployment

This is a COMPLETE MLOps pipeline following best practices!
"""

import sys
import os
from pathlib import Path
import yaml
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Import our custom modules
from data_generation.generate_banking_data import BankingDataGenerator
from feature_engineering.feature_pipeline import FeatureEngineer
from models.ml_algorithms import MLModelFactory
from models.model_evaluation import ModelEvaluator
from mlops.experiment_tracking import ExperimentTracker, ModelRegistry

# PySpark imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./logs/pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AMLFraudDetectionPipeline:
    """
    Complete end-to-end MLOps pipeline for AML fraud detection.

    This is the master class that coordinates everything!
    """

    def __init__(self, config_path: str = './config/pipeline_config.yaml'):
        """
        Initialize the pipeline.

        Parameters:
        -----------
        config_path : str
            Path to configuration file
        """
        logger.info("\n" + "="*80)
        logger.info("🏦 AML FRAUD DETECTION - MLOPS PIPELINE")
        logger.info("="*80 + "\n")

        # Load configuration
        logger.info("📋 Loading configuration...")
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        logger.info(f"   ✓ Configuration loaded from {config_path}")

        # Initialize Spark
        logger.info("⚡ Initializing Spark...")
        self.spark = self._initialize_spark()
        logger.info("   ✓ Spark session created")

        # Initialize MLOps components
        logger.info("🔬 Initializing MLOps components...")
        self.experiment_tracker = ExperimentTracker(
            experiment_name=self.config['project']['name'],
            tracking_uri=self.config['mlops']['experiment_tracking']['uri']
        )
        self.model_registry = ModelRegistry(
            registry_path=self.config['mlops']['model_registry']['path']
        )
        logger.info("   ✓ MLOps components ready")

        # Initialize evaluator
        self.evaluator = ModelEvaluator()

        # Storage for results
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.trained_models = {}
        self.evaluation_results = {}

        logger.info("\n✅ Pipeline initialized successfully!\n")

    def _initialize_spark(self) -> SparkSession:
        """
        Initialize Spark session with optimized configuration.

        Returns:
        --------
        SparkSession : Configured Spark session
        """
        builder = SparkSession.builder \
            .appName(self.config['spark']['app_name']) \
            .master(self.config['spark']['master'])

        # Apply Spark configurations
        for key, value in self.config['spark']['configs'].items():
            builder = builder.config(key, value)

        # Set memory configurations
        builder = builder.config("spark.executor.memory",
                               self.config['spark']['executor_memory'])
        builder = builder.config("spark.driver.memory",
                               self.config['spark']['driver_memory'])

        return builder.getOrCreate()

    # =========================================================================
    # STEP 1: DATA GENERATION
    # =========================================================================

    def generate_data(self):
        """
        Generate synthetic banking transaction data.

        This creates our training dataset with both legitimate and fraudulent
        transactions, including tabular and text features!
        """
        logger.info("\n" + "="*80)
        logger.info("📊 STEP 1: GENERATING BANKING TRANSACTION DATA")
        logger.info("="*80 + "\n")

        # Create data generator
        generator = BankingDataGenerator(
            num_transactions=self.config['data']['num_transactions'],
            fraud_ratio=self.config['data']['fraud_ratio'],
            random_seed=self.config['data']['random_seed']
        )

        # Generate data
        data = generator.generate_data()

        # Save raw data
        output_path = './data/raw/banking_transactions.csv'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        generator.save_data(data, output_path)

        # Load into Spark
        logger.info("⚡ Loading data into Spark...")
        self.raw_data = self.spark.read.csv(output_path, header=True, inferSchema=True)

        logger.info(f"   ✓ Loaded {self.raw_data.count():,} transactions")
        logger.info(f"   ✓ Features: {len(self.raw_data.columns)}")

        # Show sample
        logger.info("\n📋 Sample of data:")
        self.raw_data.show(5, truncate=False)

        logger.info("\n✅ Data generation complete!\n")

    # =========================================================================
    # STEP 2: DATA SPLITTING
    # =========================================================================

    def split_data(self):
        """
        Split data into train/validation/test sets.

        This ensures we can:
        - Train models on one set
        - Tune hyperparameters on another
        - Get unbiased final evaluation on a third
        """
        logger.info("\n" + "="*80)
        logger.info("✂️  STEP 2: SPLITTING DATA")
        logger.info("="*80 + "\n")

        train_ratio = self.config['data']['train_split']
        val_ratio = self.config['data']['validation_split']
        test_ratio = self.config['data']['test_split']

        logger.info(f"   Split ratios: Train={train_ratio}, Val={val_ratio}, Test={test_ratio}")

        # Split data
        train, val, test = self.raw_data.randomSplit(
            [train_ratio, val_ratio, test_ratio],
            seed=self.config['data']['random_seed']
        )

        self.train_data_raw = train
        self.val_data_raw = val
        self.test_data_raw = test

        logger.info(f"\n   ✓ Training set:   {train.count():,} transactions")
        logger.info(f"   ✓ Validation set: {val.count():,} transactions")
        logger.info(f"   ✓ Test set:       {test.count():,} transactions")

        # Check class distribution
        train_fraud_rate = train.filter(col("is_fraud") == 1).count() / train.count()
        logger.info(f"\n   📊 Training set fraud rate: {train_fraud_rate:.2%}")

        logger.info("\n✅ Data splitting complete!\n")

    # =========================================================================
    # STEP 3: FEATURE ENGINEERING
    # =========================================================================

    def engineer_features(self):
        """
        Apply feature engineering to transform raw data into ML-ready features.

        This includes:
        - Text vectorization (TF-IDF)
        - Categorical encoding (one-hot)
        - Numerical scaling (standardization)
        - Advanced feature creation
        """
        logger.info("\n" + "="*80)
        logger.info("🔧 STEP 3: FEATURE ENGINEERING")
        logger.info("="*80 + "\n")

        # Initialize feature engineer
        engineer = FeatureEngineer(self.spark, self.config['feature_engineering'])

        # Fit on training data and transform
        logger.info("📚 Fitting feature pipeline on training data...")
        self.train_data, self.feature_pipeline = engineer.fit_transform(self.train_data_raw)

        # Transform validation and test data
        logger.info("🔄 Transforming validation data...")
        self.val_data = engineer.transform(self.val_data_raw, self.feature_pipeline)

        logger.info("🔄 Transforming test data...")
        self.test_data = engineer.transform(self.test_data_raw, self.feature_pipeline)

        # Cache data for faster access
        self.train_data.cache()
        self.val_data.cache()
        self.test_data.cache()

        logger.info(f"\n   ✓ Feature engineering complete!")
        logger.info(f"   ✓ Training samples: {self.train_data.count():,}")

        # Show sample of features
        logger.info("\n📋 Sample of engineered features:")
        self.train_data.select("transaction_id", "is_fraud", "features").show(3, truncate=False)

        logger.info("\n✅ Feature engineering complete!\n")

    # =========================================================================
    # STEP 4: MODEL TRAINING
    # =========================================================================

    def train_models(self):
        """
        Train all configured ML models.

        This is where the magic happens! We train multiple algorithms
        and see which one works best for fraud detection.
        """
        logger.info("\n" + "="*80)
        logger.info("🤖 STEP 4: TRAINING ML MODELS")
        logger.info("="*80 + "\n")

        # Initialize model factory
        factory = MLModelFactory(self.spark, self.config['models'])

        # Get enabled models
        models_config = self.config['models']

        # =====================================================================
        # Train each model with experiment tracking
        # =====================================================================

        model_definitions = [
            ('Logistic Regression', factory.create_logistic_regression,
             models_config.get('logistic_regression', {}).get('enabled', True)),

            ('Decision Tree', factory.create_decision_tree,
             models_config.get('decision_tree', {}).get('enabled', True)),

            ('Random Forest', factory.create_random_forest,
             models_config.get('random_forest', {}).get('enabled', True)),

            ('Gradient Boosted Trees', factory.create_gbt,
             models_config.get('gbt', {}).get('enabled', True)),

            ('Naive Bayes', factory.create_naive_bayes,
             models_config.get('naive_bayes', {}).get('enabled', True)),

            ('Linear SVM', factory.create_linear_svm,
             models_config.get('linear_svm', {}).get('enabled', True)),

            ('Neural Network (MLP)', factory.create_mlp,
             models_config.get('mlp', {}).get('enabled', True)),
        ]

        for model_name, model_creator, enabled in model_definitions:
            if not enabled:
                logger.info(f"⏭️  Skipping {model_name} (disabled in config)")
                continue

            logger.info(f"\n{'─'*70}")
            logger.info(f"🎯 Training: {model_name}")
            logger.info(f"{'─'*70}\n")

            try:
                # Start MLflow run
                run_id = self.experiment_tracker.start_run(
                    run_name=model_name.lower().replace(' ', '_'),
                    tags={'model_type': model_name, 'stage': 'training'}
                )

                # Create and train model
                model = model_creator()

                logger.info("🏋️  Training model...")
                start_time = datetime.now()

                trained_model = model.fit(self.train_data)

                training_time = (datetime.now() - start_time).total_seconds()

                logger.info(f"   ✅ Training complete in {training_time:.2f} seconds")

                # Log parameters
                # Extract model parameters
                params = {param.name: trained_model.getOrDefault(param)
                         for param in trained_model.params}
                self.experiment_tracker.log_params(params)

                # Log training time
                self.experiment_tracker.log_metrics({'training_time_seconds': training_time})

                # Make predictions on validation set
                logger.info("📊 Evaluating on validation set...")
                val_predictions = trained_model.transform(self.val_data)

                # Evaluate
                metrics = self.evaluator.evaluate_model(val_predictions, model_name)

                # Log metrics
                self.experiment_tracker.log_metrics(metrics)

                # Log model
                self.experiment_tracker.log_model(trained_model, f"model_{model_name}")

                # Save trained model and results
                self.trained_models[model_name] = trained_model
                self.evaluation_results[model_name] = metrics

                # End MLflow run
                self.experiment_tracker.end_run(status="FINISHED")

                logger.info(f"✅ {model_name} complete!")

            except Exception as e:
                logger.error(f"❌ Error training {model_name}: {str(e)}")
                self.experiment_tracker.end_run(status="FAILED")
                continue

        logger.info("\n" + "="*80)
        logger.info(f"✅ TRAINED {len(self.trained_models)} MODELS SUCCESSFULLY!")
        logger.info("="*80 + "\n")

    # =========================================================================
    # STEP 5: MODEL COMPARISON AND SELECTION
    # =========================================================================

    def compare_and_select_best_model(self):
        """
        Compare all models and select the best one.

        We use F1-score as the primary metric because it balances
        precision (avoiding false alarms) and recall (catching fraud).
        """
        logger.info("\n" + "="*80)
        logger.info("🏆 STEP 5: MODEL COMPARISON AND SELECTION")
        logger.info("="*80 + "\n")

        # Compare models
        comparison_df = self.evaluator.compare_models(self.evaluation_results)

        # Select best model based on F1 score
        best_model_name = comparison_df.index[0]
        self.best_model = self.trained_models[best_model_name]
        self.best_model_name = best_model_name
        self.best_model_metrics = self.evaluation_results[best_model_name]

        logger.info(f"\n🥇 BEST MODEL: {best_model_name}")
        logger.info(f"   F1-Score:  {self.best_model_metrics['f1']:.4f}")
        logger.info(f"   AUC-ROC:   {self.best_model_metrics['auc_roc']:.4f}")
        logger.info(f"   Precision: {self.best_model_metrics['precision']:.4f}")
        logger.info(f"   Recall:    {self.best_model_metrics['recall']:.4f}")

        logger.info("\n✅ Model selection complete!\n")

        return best_model_name

    # =========================================================================
    # STEP 6: FINAL EVALUATION ON TEST SET
    # =========================================================================

    def final_evaluation(self):
        """
        Evaluate best model on unseen test set.

        This gives us an unbiased estimate of real-world performance!
        """
        logger.info("\n" + "="*80)
        logger.info("📊 STEP 6: FINAL EVALUATION ON TEST SET")
        logger.info("="*80 + "\n")

        logger.info(f"🎯 Evaluating {self.best_model_name} on test set...")

        # Make predictions
        test_predictions = self.best_model.transform(self.test_data)

        # Evaluate
        test_metrics = self.evaluator.evaluate_model(test_predictions, self.best_model_name)

        # Save test predictions
        output_path = './data/processed/test_predictions.parquet'
        test_predictions.write.mode('overwrite').parquet(output_path)
        logger.info(f"\n💾 Test predictions saved to {output_path}")

        # Generate plots
        logger.info("\n📊 Generating evaluation plots...")

        os.makedirs('./reports/figures', exist_ok=True)

        # Confusion Matrix
        self.evaluator.plot_confusion_matrix(
            test_predictions,
            self.best_model_name,
            save_path=f'./reports/figures/confusion_matrix_{self.best_model_name.lower().replace(" ", "_")}.png'
        )

        # ROC Curve
        self.evaluator.plot_roc_curve(
            test_predictions,
            self.best_model_name,
            save_path=f'./reports/figures/roc_curve_{self.best_model_name.lower().replace(" ", "_")}.png'
        )

        # Precision-Recall Curve
        self.evaluator.plot_precision_recall_curve(
            test_predictions,
            self.best_model_name,
            save_path=f'./reports/figures/pr_curve_{self.best_model_name.lower().replace(" ", "_")}.png'
        )

        logger.info("   ✓ Plots saved to ./reports/figures/")

        logger.info("\n✅ Final evaluation complete!\n")

        return test_metrics

    # =========================================================================
    # STEP 7: MODEL REGISTRATION
    # =========================================================================

    def register_best_model(self):
        """
        Register the best model in the model registry.

        This allows us to version, track, and deploy models!
        """
        logger.info("\n" + "="*80)
        logger.info("📚 STEP 7: MODEL REGISTRATION")
        logger.info("="*80 + "\n")

        version = f"1.0.{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.model_registry.register_model(
            model=self.best_model,
            model_name=self.best_model_name.lower().replace(' ', '_'),
            version=version,
            metrics=self.best_model_metrics,
            params={},
            tags={'stage': 'production_candidate'}
        )

        logger.info(f"✅ Model registered as version {version}\n")

    # =========================================================================
    # STEP 8: RUN COMPLETE PIPELINE
    # =========================================================================

    def run(self):
        """
        Run the complete end-to-end pipeline!

        This executes all steps in order.
        """
        try:
            logger.info("\n" + "🚀 "*30)
            logger.info("STARTING COMPLETE MLOPS PIPELINE")
            logger.info("🚀 "*30 + "\n")

            pipeline_start_time = datetime.now()

            # Execute all steps
            self.generate_data()
            self.split_data()
            self.engineer_features()
            self.train_models()
            self.compare_and_select_best_model()
            self.final_evaluation()
            self.register_best_model()

            # Calculate total time
            total_time = (datetime.now() - pipeline_start_time).total_seconds()

            # Final summary
            logger.info("\n" + "="*80)
            logger.info("🎉 PIPELINE COMPLETE!")
            logger.info("="*80)
            logger.info(f"\n⏱️  Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
            logger.info(f"🏆 Best model: {self.best_model_name}")
            logger.info(f"📊 F1-Score: {self.best_model_metrics['f1']:.4f}")
            logger.info(f"📊 AUC-ROC: {self.best_model_metrics['auc_roc']:.4f}")
            logger.info(f"\n📁 Results saved to:")
            logger.info(f"   - Data: ./data/")
            logger.info(f"   - Models: ./model_registry/")
            logger.info(f"   - Experiments: ./experiments/")
            logger.info(f"   - Reports: ./reports/")
            logger.info(f"   - Logs: ./logs/")
            logger.info("\n" + "="*80 + "\n")

        except Exception as e:
            logger.error(f"\n❌ PIPELINE FAILED: {str(e)}")
            raise

        finally:
            # Stop Spark
            logger.info("🛑 Stopping Spark...")
            self.spark.stop()
            logger.info("   ✓ Spark stopped\n")


def main():
    """
    Main entry point.
    """
    # Create necessary directories
    for directory in ['data/raw', 'data/processed', 'logs', 'reports/figures',
                     'experiments', 'model_registry']:
        os.makedirs(directory, exist_ok=True)

    # Initialize and run pipeline
    pipeline = AMLFraudDetectionPipeline(config_path='./config/pipeline_config.yaml')
    pipeline.run()


if __name__ == "__main__":
    main()
