"""
============================================================================
MODEL INFERENCE - DEPLOY AND USE TRAINED MODELS
============================================================================

Welcome to the deployment module! 🚀

After training models, we need to USE them! This module shows you how to:
1. Load saved models
2. Make predictions on new data
3. Serve models for real-time inference
4. Batch predictions for large datasets

DEPLOYMENT SCENARIOS:
---------------------
1. **Batch Prediction**: Process many transactions at once
   - Use case: Daily fraud review
   - Example: Analyze all yesterday's transactions

2. **Real-time Prediction**: Instant fraud detection
   - Use case: Transaction approval/denial
   - Example: Flag transaction before it completes

3. **API Serving**: REST API for applications
   - Use case: Integration with other systems
   - Example: Banking app calls our fraud API

Let's learn how to deploy! 🚀
============================================================================
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.ml import PipelineModel
import logging
import os
from typing import Dict, Any, List
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelInference:
    """
    Handles model deployment and inference.

    Think of this as your model deployment assistant!
    """

    def __init__(
        self,
        spark: SparkSession,
        model_path: str = None,
        pipeline_path: str = "data/models/feature_pipeline"
    ):
        """
        Initialize inference engine.

        Parameters:
        -----------
        spark : SparkSession
            Spark session
        model_path : str
            Path to trained model
        pipeline_path : str
            Path to feature engineering pipeline
        """
        self.spark = spark
        self.model_path = model_path
        self.pipeline_path = pipeline_path
        self.model = None
        self.pipeline = None

        logger.info("🚀 Model Inference Engine initialized")

    def load_model(self, model_path: str = None):
        """
        Load a trained model from disk.

        WHY LOAD MODELS?
        ----------------
        - Don't retrain every time!
        - Use same model for consistent predictions
        - Deploy pre-trained models to production

        Parameters:
        -----------
        model_path : str
            Path to saved model
        """
        if model_path:
            self.model_path = model_path

        if not self.model_path:
            raise ValueError("No model path specified!")

        logger.info(f"📂 Loading model from: {self.model_path}")

        try:
            # Different loading methods for different model types
            if os.path.isdir(self.model_path):
                from pyspark.ml import PipelineModel
                self.model = PipelineModel.load(self.model_path)
            else:
                raise ValueError(f"Invalid model path: {self.model_path}")

            logger.info("✅ Model loaded successfully!")

        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise

    def load_pipeline(self):
        """
        Load the feature engineering pipeline.

        WHY PIPELINE?
        -------------
        We need to transform new data the SAME WAY as training data!
        - Same feature engineering
        - Same scaling
        - Same encoding

        The pipeline ensures consistency!
        """
        logger.info(f"📂 Loading feature pipeline from: {self.pipeline_path}")

        try:
            self.pipeline = PipelineModel.load(self.pipeline_path)
            logger.info("✅ Feature pipeline loaded successfully!")

        except Exception as e:
            logger.error(f"❌ Failed to load pipeline: {e}")
            raise

    def preprocess_data(self, df: DataFrame) -> DataFrame:
        """
        Apply feature engineering to new data.

        IMPORTANT: Use the SAME transformations as training!

        Parameters:
        -----------
        df : DataFrame
            Raw transaction data

        Returns:
        --------
        DataFrame with features ready for prediction
        """
        logger.info("🔧 Preprocessing data...")

        if not self.pipeline:
            logger.info("  Loading feature pipeline...")
            self.load_pipeline()

        # Apply pipeline
        df_transformed = self.pipeline.transform(df)

        logger.info("✅ Data preprocessed!")
        return df_transformed

    def predict(self, df: DataFrame) -> DataFrame:
        """
        Make predictions on new data.

        This is the main inference method!

        Parameters:
        -----------
        df : DataFrame
            Data with 'features' column

        Returns:
        --------
        DataFrame with predictions
        """
        if not self.model:
            raise ValueError("Model not loaded! Call load_model() first.")

        logger.info("🔮 Making predictions...")

        # Make predictions
        predictions = self.model.transform(df)

        logger.info("✅ Predictions complete!")
        return predictions

    def predict_from_raw(self, df: DataFrame) -> DataFrame:
        """
        End-to-end prediction from raw data.

        This does EVERYTHING:
        1. Feature engineering
        2. Prediction
        3. Post-processing

        Parameters:
        -----------
        df : DataFrame
            Raw transaction data

        Returns:
        --------
        DataFrame with predictions and fraud probabilities
        """
        logger.info("🚀 Running end-to-end prediction...")

        # Step 1: Preprocess
        df_processed = self.preprocess_data(df)

        # Step 2: Predict
        predictions = self.predict(df_processed)

        # Step 3: Add interpretable labels
        # Convert numeric predictions back to category names
        category_mapping = {
            0: 'LEGITIMATE',
            1: 'STRUCTURING',
            2: 'LAYERING',
            3: 'SHELL_COMPANY',
            4: 'ROUND_TRIPPING'
        }

        # Add readable prediction labels
        from pyspark.sql.functions import udf
        from pyspark.sql.types import StringType

        label_to_category = udf(
            lambda x: category_mapping.get(int(x), 'UNKNOWN'),
            StringType()
        )

        predictions = predictions.withColumn(
            'predicted_category',
            label_to_category('prediction')
        )

        # Add fraud flag (1 if any type of fraud, 0 if legitimate)
        predictions = predictions.withColumn(
            'is_fraud',
            (predictions['prediction'] != 0).cast('int')
        )

        logger.info("✅ End-to-end prediction complete!")
        return predictions

    def batch_predict(
        self,
        input_path: str,
        output_path: str,
        batch_size: int = 10000
    ):
        """
        Batch prediction for large datasets.

        WHEN TO USE:
        ------------
        - Processing historical data
        - Daily fraud review
        - Offline analysis

        Example: Analyze all of yesterday's transactions

        Parameters:
        -----------
        input_path : str
            Path to input data
        output_path : str
            Where to save predictions
        batch_size : int
            Process in chunks (memory efficiency)
        """
        logger.info(f"📦 Starting batch prediction")
        logger.info(f"  Input: {input_path}")
        logger.info(f"  Output: {output_path}")

        # Load data
        df = self.spark.read.parquet(input_path)
        total_count = df.count()
        logger.info(f"  Total transactions: {total_count:,}")

        # Make predictions
        predictions = self.predict_from_raw(df)

        # Save results
        predictions.write.mode('overwrite').parquet(output_path)

        logger.info(f"✅ Batch prediction complete!")
        logger.info(f"  Results saved to: {output_path}")

        # Summary statistics
        fraud_count = predictions.filter('is_fraud = 1').count()
        fraud_rate = (fraud_count / total_count) * 100

        logger.info(f"\n📊 Prediction Summary:")
        logger.info(f"  Total transactions: {total_count:,}")
        logger.info(f"  Flagged as fraud: {fraud_count:,} ({fraud_rate:.2f}%)")

        # Breakdown by category
        category_dist = predictions.groupBy('predicted_category').count().collect()
        logger.info(f"\n  Breakdown by category:")
        for row in category_dist:
            count = row['count']
            pct = (count / total_count) * 100
            logger.info(f"    {row['predicted_category']}: {count:,} ({pct:.2f}%)")

    def explain_prediction(
        self,
        transaction_id: str,
        predictions: DataFrame
    ) -> Dict[str, Any]:
        """
        Explain why a transaction was flagged as fraud.

        EXPLAINABILITY IS CRITICAL!
        ---------------------------
        - Regulators require explanations
        - Customers deserve to know why
        - Helps improve the model

        Parameters:
        -----------
        transaction_id : str
            ID of transaction to explain
        predictions : DataFrame
            Predictions with features

        Returns:
        --------
        Dictionary with explanation
        """
        logger.info(f"🔍 Explaining prediction for: {transaction_id}")

        # Get the transaction
        txn = predictions.filter(
            predictions['transaction_id'] == transaction_id
        ).first()

        if not txn:
            logger.error(f"Transaction {transaction_id} not found!")
            return {}

        # Build explanation
        explanation = {
            'transaction_id': transaction_id,
            'predicted_category': txn['predicted_category'],
            'is_fraud': bool(txn['is_fraud']),
            'confidence': float(txn['prediction']),
            'key_factors': []
        }

        # Identify key factors (simplified)
        # In production, you'd use SHAP or LIME for better explanations
        factors = []

        if txn['is_large_amount']:
            factors.append("Large transaction amount")

        if txn['is_near_threshold']:
            factors.append("Amount near reporting threshold")

        if txn['is_international']:
            factors.append("International transaction")

        if txn['is_night']:
            factors.append("Transaction at unusual hour")

        if txn['is_round_amount']:
            factors.append("Suspiciously round amount")

        explanation['key_factors'] = factors

        logger.info(f"  Prediction: {explanation['predicted_category']}")
        logger.info(f"  Key factors: {', '.join(factors)}")

        return explanation


def demo_inference():
    """
    Demonstration of model inference.

    Shows how to load and use trained models!
    """
    logger.info("🎓 MODEL INFERENCE DEMONSTRATION")
    logger.info("="*70)

    # Initialize Spark
    spark = SparkSession.builder \
        .appName("ModelInferenceDemo") \
        .master("local[*]") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    try:
        # Initialize inference engine
        inference = ModelInference(spark)

        # Load the best model (you'd select this based on evaluation)
        model_path = "data/models/Random_Forest_Tuned"

        if os.path.exists(model_path):
            inference.load_model(model_path)
            logger.info("✅ Model loaded!")

            # Load some test data
            data_path = "data/raw/transactions.parquet"
            if os.path.exists(data_path):
                df = spark.read.parquet(data_path)

                # Take a small sample
                sample = df.limit(100)

                # Make predictions
                predictions = inference.predict_from_raw(sample)

                # Show results
                logger.info("\n📊 Sample Predictions:")
                predictions.select(
                    'transaction_id',
                    'amount',
                    'predicted_category',
                    'is_fraud'
                ).show(10, truncate=False)

                logger.info("✅ Demo complete!")

        else:
            logger.warning(f"Model not found at {model_path}")
            logger.info("Please train models first: python src/main_pipeline.py")

    finally:
        spark.stop()


if __name__ == "__main__":
    """
    Run inference demo.

    Usage: python src/deployment/model_inference.py
    """
    demo_inference()
