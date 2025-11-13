"""
=============================================================================
MODEL DEPLOYMENT AND SERVING
=============================================================================
This module handles deploying trained models for real-time or batch prediction.

Think of this as putting your trained model "on duty" to make predictions
on new transactions!

DEPLOYMENT MODES:
1. BATCH: Process many transactions at once (e.g., nightly fraud checks)
2. REAL-TIME: Process transactions as they happen (e.g., at ATM or online)
3. REST API: Serve predictions via HTTP endpoints

For AML fraud detection, we typically use:
- REAL-TIME for transaction approval (instant decision needed)
- BATCH for retrospective analysis (find patterns after the fact)
"""

from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession, DataFrame
from pathlib import Path
import logging
import time
from typing import Dict, Any
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudDetectionService:
    """
    Service for deploying and serving fraud detection models.

    This is like a "cashier" that uses the trained model to make decisions!
    """

    def __init__(self, model_path: str, feature_pipeline_path: str):
        """
        Initialize fraud detection service.

        Parameters:
        -----------
        model_path : str
            Path to trained model
        feature_pipeline_path : str
            Path to feature engineering pipeline
        """
        logger.info("🚀 Initializing Fraud Detection Service...")

        # Initialize Spark
        self.spark = SparkSession.builder \
            .appName("FraudDetectionService") \
            .master("local[*]") \
            .getOrCreate()

        # Load models
        logger.info("📦 Loading models...")

        try:
            self.feature_pipeline = PipelineModel.load(feature_pipeline_path)
            logger.info(f"   ✓ Feature pipeline loaded from {feature_pipeline_path}")
        except Exception as e:
            logger.error(f"   ❌ Error loading feature pipeline: {str(e)}")
            self.feature_pipeline = None

        try:
            self.model = PipelineModel.load(model_path)
            logger.info(f"   ✓ Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"   ❌ Error loading model: {str(e)}")
            self.model = None

        # Statistics
        self.predictions_made = 0
        self.fraud_detected = 0
        self.start_time = time.time()

        logger.info("✅ Service initialized and ready!")

    def predict_batch(self, transactions: DataFrame) -> DataFrame:
        """
        Make predictions on a batch of transactions.

        Use this for processing many transactions at once!

        Parameters:
        -----------
        transactions : DataFrame
            DataFrame with transaction data

        Returns:
        --------
        DataFrame : Predictions with fraud probability and decision
        """
        logger.info(f"🔮 Making batch predictions for {transactions.count()} transactions...")

        start_time = time.time()

        try:
            # Apply feature engineering
            if self.feature_pipeline:
                transactions_features = self.feature_pipeline.transform(transactions)
            else:
                transactions_features = transactions

            # Make predictions
            predictions = self.model.transform(transactions_features)

            # Extract results
            results = predictions.select(
                "transaction_id",
                "is_fraud",
                "prediction",
                "probability"
            )

            # Update statistics
            fraud_count = results.filter(results.prediction == 1).count()
            self.predictions_made += results.count()
            self.fraud_detected += fraud_count

            elapsed_time = time.time() - start_time
            throughput = results.count() / elapsed_time

            logger.info(f"   ✅ Predictions complete!")
            logger.info(f"   ⏱️  Time: {elapsed_time:.2f} seconds")
            logger.info(f"   📊 Throughput: {throughput:.0f} transactions/second")
            logger.info(f"   🚨 Fraud detected: {fraud_count} ({fraud_count/results.count()*100:.2f}%)")

            return results

        except Exception as e:
            logger.error(f"   ❌ Error making predictions: {str(e)}")
            return None

    def predict_single(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction on a single transaction.

        Use this for real-time fraud detection!

        Parameters:
        -----------
        transaction : Dict
            Dictionary with transaction data

        Returns:
        --------
        Dict : Prediction result with fraud probability
        """
        logger.info(f"🔮 Making real-time prediction for transaction {transaction.get('transaction_id', 'unknown')}")

        try:
            # Convert to DataFrame
            transaction_df = self.spark.createDataFrame([transaction])

            # Apply feature engineering
            if self.feature_pipeline:
                transaction_features = self.feature_pipeline.transform(transaction_df)
            else:
                transaction_features = transaction_df

            # Make prediction
            prediction = self.model.transform(transaction_features)

            # Extract result
            result_row = prediction.select(
                "transaction_id",
                "prediction",
                "probability"
            ).first()

            # Parse result
            result = {
                'transaction_id': result_row.transaction_id,
                'is_fraud': bool(result_row.prediction),
                'fraud_probability': float(result_row.probability[1]),
                'legitimate_probability': float(result_row.probability[0]),
                'decision': 'DECLINE' if result_row.prediction == 1 else 'APPROVE',
                'confidence': max(float(result_row.probability[0]), float(result_row.probability[1]))
            }

            # Update statistics
            self.predictions_made += 1
            if result['is_fraud']:
                self.fraud_detected += 1

            # Log result
            if result['is_fraud']:
                logger.warning(f"   🚨 FRAUD DETECTED! Probability: {result['fraud_probability']:.2%}")
            else:
                logger.info(f"   ✅ Transaction approved (Confidence: {result['confidence']:.2%})")

            return result

        except Exception as e:
            logger.error(f"   ❌ Error making prediction: {str(e)}")
            return {
                'transaction_id': transaction.get('transaction_id', 'unknown'),
                'error': str(e),
                'decision': 'ERROR'
            }

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get service statistics.

        Returns:
        --------
        Dict : Service statistics
        """
        uptime = time.time() - self.start_time

        stats = {
            'uptime_seconds': uptime,
            'uptime_hours': uptime / 3600,
            'predictions_made': self.predictions_made,
            'fraud_detected': self.fraud_detected,
            'fraud_rate': self.fraud_detected / self.predictions_made if self.predictions_made > 0 else 0,
            'throughput': self.predictions_made / uptime if uptime > 0 else 0
        }

        logger.info("\n" + "="*60)
        logger.info("📊 SERVICE STATISTICS")
        logger.info("="*60)
        logger.info(f"   Uptime: {stats['uptime_hours']:.2f} hours")
        logger.info(f"   Predictions made: {stats['predictions_made']:,}")
        logger.info(f"   Fraud detected: {stats['fraud_detected']:,}")
        logger.info(f"   Fraud rate: {stats['fraud_rate']:.2%}")
        logger.info(f"   Throughput: {stats['throughput']:.2f} predictions/second")
        logger.info("="*60 + "\n")

        return stats

    def stop(self):
        """Stop the service and clean up."""
        logger.info("🛑 Stopping Fraud Detection Service...")
        self.get_statistics()
        self.spark.stop()
        logger.info("   ✓ Service stopped")


def demo_real_time_prediction():
    """
    Demonstrate real-time fraud detection.
    """
    print("\n" + "="*80)
    print("🎬 DEMO: REAL-TIME FRAUD DETECTION")
    print("="*80 + "\n")

    # Example transactions (would come from payment system in production)
    test_transactions = [
        {
            'transaction_id': 'TXN_LEGIT_001',
            'transaction_amount': 45.50,
            'account_age_days': 1200,
            'num_transactions_24h': 2,
            'avg_transaction_amount': 50.00,
            'transaction_hour': 14,
            'transaction_day_of_week': 2,
            'customer_age': 35,
            'credit_score': 750,
            'num_failed_logins': 0,
            'source_country': 'USA',
            'destination_country': 'USA',
            'account_type': 'checking',
            'transaction_description': 'Purchase at grocery store',
            'merchant_category': 'Grocery',
            'customer_notes': 'Regular monthly shopping'
        },
        {
            'transaction_id': 'TXN_FRAUD_001',
            'transaction_amount': 9500.00,
            'account_age_days': 15,
            'num_transactions_24h': 8,
            'avg_transaction_amount': 50.00,
            'transaction_hour': 3,
            'transaction_day_of_week': 6,
            'customer_age': 25,
            'credit_score': 620,
            'num_failed_logins': 5,
            'source_country': 'USA',
            'destination_country': 'Nigeria',
            'account_type': 'checking',
            'transaction_description': 'Wire transfer to overseas account - urgent',
            'merchant_category': 'Money Transfer',
            'customer_notes': 'Disputed transaction'
        }
    ]

    # Note: This demo would work if models are available
    logger.info("💡 Demo: In production, this would:")
    logger.info("   1. Load trained models")
    logger.info("   2. Receive transactions from payment system")
    logger.info("   3. Make instant fraud predictions")
    logger.info("   4. Return APPROVE or DECLINE decision")
    logger.info("   5. Log all decisions for audit trail")

    for txn in test_transactions:
        logger.info(f"\n📝 Transaction: {txn['transaction_id']}")
        logger.info(f"   Amount: ${txn['transaction_amount']:.2f}")
        logger.info(f"   Description: {txn['transaction_description']}")


def demo_batch_prediction():
    """
    Demonstrate batch fraud detection.
    """
    print("\n" + "="*80)
    print("🎬 DEMO: BATCH FRAUD DETECTION")
    print("="*80 + "\n")

    logger.info("💡 Demo: In production, batch mode:")
    logger.info("   1. Loads all transactions from past 24 hours")
    logger.info("   2. Processes thousands/millions of transactions")
    logger.info("   3. Generates fraud reports")
    logger.info("   4. Alerts investigators to suspicious patterns")
    logger.info("   5. Updates risk scores for accounts")


def main():
    """
    Test deployment module.
    """
    print("\n" + "="*80)
    print("🚀 MODEL DEPLOYMENT MODULE")
    print("="*80 + "\n")

    demo_real_time_prediction()
    demo_batch_prediction()

    logger.info("\n✅ Deployment module ready!")


if __name__ == "__main__":
    main()
