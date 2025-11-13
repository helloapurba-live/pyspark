"""
============================================================================
MAIN ML PIPELINE - COMPLETE END-TO-END WORKFLOW
============================================================================

🎉 Welcome to the MAIN PIPELINE! 🎉

This is the conductor of our orchestra - it coordinates all the pieces:
- Data loading
- Feature engineering
- Model training (10+ algorithms!)
- Evaluation
- Comparison
- Reporting

Think of this as the "Run Everything" button!

WHAT HAPPENS WHEN YOU RUN THIS:
--------------------------------
1. 🔧 Initialize Spark
2. 📂 Load transaction data
3. ✨ Engineer features (tabular + text)
4. ✂️  Split into train/validation/test
5. 🤖 Train 10+ ML models
6. 📊 Evaluate all models
7. 🏆 Compare performance
8. 💾 Save everything
9. 📝 Generate report

MLOPS BEST PRACTICES:
--------------------
✅ Reproducibility (fixed random seeds)
✅ Data versioning (save everything)
✅ Experiment tracking (log all metrics)
✅ Model versioning (save all models)
✅ Comprehensive evaluation (multiple metrics)
✅ Automated pipeline (no manual steps)
✅ Error handling (graceful failures)
✅ Logging (visibility into process)

Let's run the complete pipeline! 🚀
============================================================================
"""

import os
import sys
from datetime import datetime
import logging
import json
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Add src directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from features.feature_engineering import FeatureEngineer
from models.model_trainer import ModelTrainer
from evaluation.model_evaluator import ModelEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MLPipeline:
    """
    Complete end-to-end ML pipeline for AML fraud detection.

    This class orchestrates the entire machine learning workflow!
    """

    def __init__(
        self,
        app_name: str = "Banking_AML_Detection",
        data_path: str = "data/raw/transactions.parquet"
    ):
        """
        Initialize the ML pipeline.

        Parameters:
        -----------
        app_name : str
            Name for the Spark application
        data_path : str
            Path to transaction data
        """
        self.app_name = app_name
        self.data_path = data_path
        self.spark = None
        self.feature_engineer = None
        self.model_trainer = None
        self.model_evaluator = None

        logger.info("🎬 ML Pipeline initialized")

    def initialize_spark(self):
        """
        Initialize Spark session.

        Spark is our distributed computing engine.
        It allows us to process large datasets efficiently!
        """
        logger.info("⚡ Initializing Spark...")

        self.spark = SparkSession.builder \
            .appName(self.app_name) \
            .master("local[*]") \
            .config("spark.driver.memory", "4g") \
            .config("spark.sql.shuffle.partitions", "10") \
            .config("spark.default.parallelism", "10") \
            .getOrCreate()

        # Set log level to reduce noise
        self.spark.sparkContext.setLogLevel("WARN")

        logger.info("  ✅ Spark initialized")
        logger.info(f"     Spark version: {self.spark.version}")
        logger.info(f"     Master: {self.spark.sparkContext.master}")

    def load_data(self):
        """
        Load transaction data from disk.

        Returns:
        --------
        DataFrame with transaction data
        """
        logger.info(f"\n📂 Loading data from: {self.data_path}")

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"Data not found at {self.data_path}\n"
                "Please run: python src/data/generate_data.py"
            )

        df = self.spark.read.parquet(self.data_path)
        count = df.count()

        logger.info(f"  ✅ Loaded {count:,} transactions")
        logger.info(f"  Columns: {', '.join(df.columns)}")

        # Show class distribution
        logger.info("\n  📊 Class distribution:")
        dist = df.groupBy("category").count().orderBy("category").collect()
        for row in dist:
            logger.info(f"     {row['category']}: {row['count']:,}")

        return df

    def run_pipeline(self):
        """
        Execute the complete ML pipeline.

        This is the main method that runs everything!
        """
        try:
            logger.info("\n" + "="*70)
            logger.info("🚀 STARTING COMPLETE ML PIPELINE")
            logger.info("="*70)
            logger.info(f"Pipeline: {self.app_name}")
            logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("="*70 + "\n")

            start_time = datetime.now()

            # ================================================================
            # STEP 1: INITIALIZE SPARK
            # ================================================================
            self.initialize_spark()

            # ================================================================
            # STEP 2: LOAD DATA
            # ================================================================
            df = self.load_data()

            # ================================================================
            # STEP 3: FEATURE ENGINEERING
            # ================================================================
            logger.info("\n" + "="*70)
            logger.info("🔧 STEP 3: FEATURE ENGINEERING")
            logger.info("="*70)

            self.feature_engineer = FeatureEngineer(self.spark)

            # Apply feature engineering
            df_engineered = self.feature_engineer.engineer_features(df)

            # Build and fit pipeline
            logger.info("\n🏗️  Building feature transformation pipeline...")
            feature_pipeline = self.feature_engineer.build_feature_pipeline(
                label_col='category',
                text_col='description'
            )

            logger.info("🔄 Fitting pipeline on data...")
            pipeline_model = feature_pipeline.fit(df_engineered)

            logger.info("🔄 Transforming data...")
            df_transformed = pipeline_model.transform(df_engineered)

            # Select only necessary columns for modeling
            df_final = df_transformed.select('features', 'label', 'category')
            df_final = df_final.cache()

            logger.info(f"✅ Feature engineering complete!")
            logger.info(f"   Final feature vector created")

            # Save pipeline for future use
            pipeline_path = "data/models/feature_pipeline"
            os.makedirs(os.path.dirname(pipeline_path), exist_ok=True)
            pipeline_model.write().overwrite().save(pipeline_path)
            logger.info(f"💾 Saved feature pipeline: {pipeline_path}")

            # ================================================================
            # STEP 4: SPLIT DATA
            # ================================================================
            logger.info("\n" + "="*70)
            logger.info("✂️  STEP 4: SPLITTING DATA")
            logger.info("="*70)

            self.model_trainer = ModelTrainer(self.spark)
            train_df, val_df, test_df = self.model_trainer.split_data(
                df_final,
                train_ratio=0.7,
                validation_ratio=0.15,
                test_ratio=0.15
            )

            # ================================================================
            # STEP 5: TRAIN MODELS
            # ================================================================
            logger.info("\n" + "="*70)
            logger.info("🤖 STEP 5: TRAINING MODELS")
            logger.info("="*70)

            models = self.model_trainer.train_all_models(train_df)

            logger.info(f"\n✅ Trained {len(models)} models successfully!")

            # Save all models
            logger.info("\n💾 Saving models...")
            for model_name, model in models.items():
                try:
                    self.model_trainer.save_model(model, model_name)
                except Exception as e:
                    logger.warning(f"⚠️  Could not save {model_name}: {e}")

            # ================================================================
            # STEP 6: EVALUATE MODELS
            # ================================================================
            logger.info("\n" + "="*70)
            logger.info("📊 STEP 6: EVALUATING MODELS")
            logger.info("="*70)

            self.model_evaluator = ModelEvaluator(self.spark)
            evaluation_results = {}

            for model_name, model in models.items():
                try:
                    result = self.model_evaluator.evaluate_model(
                        model,
                        test_df,
                        model_name
                    )
                    evaluation_results[model_name] = result
                except Exception as e:
                    logger.error(f"❌ Failed to evaluate {model_name}: {e}")

            # ================================================================
            # STEP 7: COMPARE MODELS
            # ================================================================
            logger.info("\n" + "="*70)
            logger.info("🏆 STEP 7: COMPARING MODELS")
            logger.info("="*70)

            comparison_df = self.model_evaluator.compare_models(evaluation_results)

            # ================================================================
            # STEP 8: SAVE RESULTS
            # ================================================================
            logger.info("\n" + "="*70)
            logger.info("💾 STEP 8: SAVING RESULTS")
            logger.info("="*70)

            self.model_evaluator.save_evaluation_results(
                evaluation_results,
                comparison_df
            )

            self.model_evaluator.generate_evaluation_report(evaluation_results)

            # ================================================================
            # STEP 9: PIPELINE COMPLETE
            # ================================================================
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            logger.info("\n" + "="*70)
            logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("="*70)
            logger.info(f"Total execution time: {duration:.2f} seconds ({duration/60:.1f} minutes)")
            logger.info(f"Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("="*70)

            # Print summary
            logger.info("\n📊 FINAL SUMMARY:")
            logger.info("="*70)
            logger.info(f"✅ Data loaded: {df.count():,} transactions")
            logger.info(f"✅ Features engineered: {len(df_engineered.columns)} features")
            logger.info(f"✅ Models trained: {len(models)}")
            logger.info(f"✅ Models evaluated: {len(evaluation_results)}")
            logger.info("\n🏆 TOP 3 MODELS (by F1-Score):")
            top_3 = comparison_df.head(3)
            for idx, row in top_3.iterrows():
                logger.info(f"  {row['Rank']}. {row['Model']}")
                logger.info(f"     Accuracy: {row['Accuracy']:.4f}, F1-Score: {row['F1-Score']:.4f}")

            logger.info("\n📂 OUTPUTS:")
            logger.info("  ├─ Models: data/models/")
            logger.info("  ├─ Reports: data/reports/")
            logger.info("  └─ Pipeline: data/models/feature_pipeline/")

            logger.info("\n💡 NEXT STEPS:")
            logger.info("  1. Review: data/reports/evaluation_report.md")
            logger.info("  2. Compare: data/reports/model_comparison.csv")
            logger.info("  3. Deploy: Best model to production")

            logger.info("\n" + "="*70)
            logger.info("🎓 CONGRATULATIONS! You've built a complete ML system!")
            logger.info("="*70 + "\n")

            return evaluation_results, comparison_df

        except Exception as e:
            logger.error(f"\n❌ Pipeline failed with error: {e}")
            import traceback
            traceback.print_exc()
            raise

        finally:
            # Clean up
            if self.spark:
                logger.info("\n🧹 Cleaning up...")
                self.spark.stop()
                logger.info("  ✅ Spark session stopped")


def main():
    """
    Main entry point for the pipeline.

    Run this to execute the complete ML workflow!
    """
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║     🏦  BANKING AML FRAUD DETECTION PIPELINE  🏦            ║
    ║                                                              ║
    ║              Complete MLOps Workflow                        ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # Configuration
    DATA_PATH = "data/raw/transactions.parquet"

    # Check if data exists
    if not os.path.exists(DATA_PATH):
        logger.error(f"❌ Data not found at {DATA_PATH}")
        logger.info("\n💡 Please generate data first:")
        logger.info("   python src/data/generate_data.py\n")
        sys.exit(1)

    # Create and run pipeline
    pipeline = MLPipeline(
        app_name="Banking_AML_Detection_Pipeline",
        data_path=DATA_PATH
    )

    try:
        results, comparison = pipeline.run_pipeline()
        logger.info("\n✅ Pipeline execution successful!")
        return 0

    except Exception as e:
        logger.error(f"\n❌ Pipeline execution failed: {e}")
        return 1


if __name__ == "__main__":
    """
    Execute the pipeline when running this file directly.

    Usage:
        python src/main_pipeline.py
    """
    exit_code = main()
    sys.exit(exit_code)
