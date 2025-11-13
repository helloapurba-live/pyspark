"""
============================================================================
MODEL EVALUATION MODULE
============================================================================

Welcome to the evaluation module! This is where we find out how good our models really are! 📊

WHAT IS MODEL EVALUATION?
--------------------------
Training a model is easy. But is it actually good? That's what we find out here!

Think of it like this:
- Training = Studying for an exam
- Evaluation = Taking the exam
- Metrics = Your grades

WHY MULTIPLE METRICS?
---------------------
Accuracy alone can be misleading! Especially for fraud detection.

Example:
- If only 1% of transactions are fraud
- A model that always says "NOT FRAUD" gets 99% accuracy!
- But catches 0% of fraud (useless!)

That's why we use multiple metrics to get the full picture.

THE METRICS WE'LL USE:
----------------------

1. **ACCURACY**: Overall correctness
   - How often is the model right?
   - Can be misleading with imbalanced classes

2. **PRECISION**: When model says "fraud," how often is it right?
   - High precision = Few false alarms
   - Important: Don't want to annoy customers with false flags

3. **RECALL**: Of all actual frauds, how many did we catch?
   - High recall = Catch most fraud
   - CRITICAL for AML! Missing fraud is expensive/illegal

4. **F1-SCORE**: Balance of precision and recall
   - Harmonic mean of precision and recall
   - Good overall metric

5. **AUC-ROC**: Area Under ROC Curve
   - How well does model distinguish classes?
   - 1.0 = Perfect, 0.5 = Random guessing

6. **CONFUSION MATRIX**: Detailed breakdown
   - Shows exactly what model got right/wrong
   - Essential for understanding errors

Let's evaluate! 🎯
============================================================================
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator
)
from pyspark.sql import functions as F
from typing import Dict, List, Any, Tuple
import logging
import json
import os
from datetime import datetime
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluates ML models with comprehensive metrics.

    This class is your model report card generator!
    """

    def __init__(self, spark: SparkSession):
        """
        Initialize the evaluator.

        Parameters:
        -----------
        spark : SparkSession
            PySpark session
        """
        self.spark = spark
        self.evaluation_results = {}
        logger.info("📊 Model Evaluator initialized")

    def calculate_confusion_matrix(
        self,
        predictions: DataFrame,
        num_classes: int
    ) -> pd.DataFrame:
        """
        Calculate confusion matrix.

        WHAT IS A CONFUSION MATRIX?
        ---------------------------
        A table showing predictions vs actual values.

        Example for binary classification:
                      Predicted
                   Fraud  Not Fraud
        Actual Fraud      85        15     (Caught 85, missed 15)
               Not Fraud  10       890     (10 false alarms, 890 correct)

        For multiclass, it's the same but bigger!

        Reading the matrix:
        - Diagonal = Correct predictions
        - Off-diagonal = Errors (confusions)

        Parameters:
        -----------
        predictions : DataFrame
            Predictions with 'label' and 'prediction' columns
        num_classes : int
            Number of classes

        Returns:
        --------
        Pandas DataFrame with confusion matrix
        """
        logger.info("🔍 Calculating confusion matrix...")

        # Calculate confusion matrix using PySpark
        confusion = predictions.groupBy('label', 'prediction').count()
        confusion = confusion.orderBy('label', 'prediction').collect()

        # Convert to matrix format
        matrix = [[0] * num_classes for _ in range(num_classes)]
        for row in confusion:
            actual = int(row['label'])
            pred = int(row['prediction'])
            count = row['count']
            if actual < num_classes and pred < num_classes:
                matrix[actual][pred] = count

        # Create DataFrame
        class_names = [f"Class_{i}" for i in range(num_classes)]
        cm_df = pd.DataFrame(matrix, index=class_names, columns=class_names)

        logger.info("✅ Confusion matrix calculated")
        return cm_df

    def calculate_class_metrics(
        self,
        predictions: DataFrame,
        num_classes: int
    ) -> Dict[int, Dict[str, float]]:
        """
        Calculate per-class metrics (precision, recall, F1).

        WHY PER-CLASS?
        --------------
        Overall accuracy can hide problems!
        Maybe model is great at detecting LEGITIMATE transactions
        but terrible at STRUCTURING fraud.

        Per-class metrics reveal these issues.

        Parameters:
        -----------
        predictions : DataFrame
            Predictions
        num_classes : int
            Number of classes

        Returns:
        --------
        Dictionary of {class_id: {precision, recall, f1}}
        """
        logger.info("📈 Calculating per-class metrics...")

        class_metrics = {}

        for class_id in range(num_classes):
            # For each class, calculate metrics
            # True Positives: Predicted AND actual are this class
            tp = predictions.filter(
                (F.col('prediction') == class_id) & (F.col('label') == class_id)
            ).count()

            # False Positives: Predicted this class, but actual is different
            fp = predictions.filter(
                (F.col('prediction') == class_id) & (F.col('label') != class_id)
            ).count()

            # False Negatives: Actual is this class, but predicted different
            fn = predictions.filter(
                (F.col('prediction') != class_id) & (F.col('label') == class_id)
            ).count()

            # Calculate metrics
            # ----------------
            # Precision = TP / (TP + FP)
            # "Of all predictions for this class, how many were right?"
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

            # Recall = TP / (TP + FN)
            # "Of all actual instances of this class, how many did we catch?"
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            # F1 = Harmonic mean of precision and recall
            # Balances precision and recall
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            class_metrics[class_id] = {
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1_score': round(f1, 4),
                'support': tp + fn  # Total actual instances of this class
            }

            logger.info(f"  Class {class_id}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

        return class_metrics

    def evaluate_model(
        self,
        model: Any,
        test_df: DataFrame,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a single model.

        This is the main evaluation method - calculates all metrics!

        Parameters:
        -----------
        model : Any
            Trained model
        test_df : DataFrame
            Test data (unseen during training!)
        model_name : str
            Model name for tracking

        Returns:
        --------
        Dictionary with all evaluation metrics
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"📊 Evaluating: {model_name}")
        logger.info(f"{'='*70}")

        # Make predictions
        logger.info("🔮 Making predictions...")
        predictions = model.transform(test_df)
        predictions = predictions.cache()  # Cache for multiple metrics

        test_count = test_df.count()
        logger.info(f"  Test samples: {test_count:,}")

        # Get number of classes
        num_classes = int(test_df.select('label').distinct().count())
        logger.info(f"  Number of classes: {num_classes}")

        # Initialize results dictionary
        results = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'test_samples': test_count,
            'num_classes': num_classes
        }

        # ================================================================
        # METRIC 1: ACCURACY
        # ================================================================
        # Overall correctness: (Correct predictions) / (Total predictions)
        accuracy_evaluator = MulticlassClassificationEvaluator(
            labelCol='label',
            predictionCol='prediction',
            metricName='accuracy'
        )
        accuracy = accuracy_evaluator.evaluate(predictions)
        results['accuracy'] = round(accuracy, 4)
        logger.info(f"\n✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

        # ================================================================
        # METRIC 2: WEIGHTED PRECISION
        # ================================================================
        # Average precision across all classes (weighted by class size)
        precision_evaluator = MulticlassClassificationEvaluator(
            labelCol='label',
            predictionCol='prediction',
            metricName='weightedPrecision'
        )
        precision = precision_evaluator.evaluate(predictions)
        results['weighted_precision'] = round(precision, 4)
        logger.info(f"✅ Weighted Precision: {precision:.4f}")

        # ================================================================
        # METRIC 3: WEIGHTED RECALL
        # ================================================================
        # Average recall across all classes (weighted by class size)
        recall_evaluator = MulticlassClassificationEvaluator(
            labelCol='label',
            predictionCol='prediction',
            metricName='weightedRecall'
        )
        recall = recall_evaluator.evaluate(predictions)
        results['weighted_recall'] = round(recall, 4)
        logger.info(f"✅ Weighted Recall: {recall:.4f}")

        # ================================================================
        # METRIC 4: F1 SCORE
        # ================================================================
        # Harmonic mean of precision and recall
        f1_evaluator = MulticlassClassificationEvaluator(
            labelCol='label',
            predictionCol='prediction',
            metricName='f1'
        )
        f1 = f1_evaluator.evaluate(predictions)
        results['f1_score'] = round(f1, 4)
        logger.info(f"✅ F1 Score: {f1:.4f}")

        # ================================================================
        # METRIC 5: CONFUSION MATRIX
        # ================================================================
        confusion_matrix = self.calculate_confusion_matrix(predictions, num_classes)
        results['confusion_matrix'] = confusion_matrix.to_dict()
        logger.info(f"\n📋 Confusion Matrix:")
        logger.info(f"\n{confusion_matrix.to_string()}")

        # ================================================================
        # METRIC 6: PER-CLASS METRICS
        # ================================================================
        class_metrics = self.calculate_class_metrics(predictions, num_classes)
        results['class_metrics'] = class_metrics

        logger.info(f"\n📊 Per-Class Performance:")
        for class_id, metrics in class_metrics.items():
            logger.info(f"  Class {class_id}:")
            logger.info(f"    ├─ Precision: {metrics['precision']:.4f}")
            logger.info(f"    ├─ Recall: {metrics['recall']:.4f}")
            logger.info(f"    ├─ F1-Score: {metrics['f1_score']:.4f}")
            logger.info(f"    └─ Support: {metrics['support']}")

        # ================================================================
        # SUMMARY
        # ================================================================
        logger.info(f"\n{'='*70}")
        logger.info(f"📊 SUMMARY FOR {model_name}")
        logger.info(f"{'='*70}")
        logger.info(f"Accuracy:    {accuracy:.4f} ⭐")
        logger.info(f"Precision:   {precision:.4f}")
        logger.info(f"Recall:      {recall:.4f}")
        logger.info(f"F1-Score:    {f1:.4f}")
        logger.info(f"{'='*70}\n")

        # Store results
        self.evaluation_results[model_name] = results

        # Clean up
        predictions.unpersist()

        return results

    def compare_models(
        self,
        results: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Compare multiple models side by side.

        This creates a leaderboard of model performance!

        Parameters:
        -----------
        results : Dict
            Dictionary of {model_name: evaluation_results}

        Returns:
        --------
        DataFrame with comparison table
        """
        logger.info("\n" + "="*70)
        logger.info("🏆 MODEL COMPARISON LEADERBOARD")
        logger.info("="*70)

        # Extract key metrics for each model
        comparison_data = []
        for model_name, result in results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': result.get('accuracy', 0),
                'Precision': result.get('weighted_precision', 0),
                'Recall': result.get('weighted_recall', 0),
                'F1-Score': result.get('f1_score', 0)
            })

        # Create DataFrame
        comparison_df = pd.DataFrame(comparison_data)

        # Sort by F1-Score (good overall metric)
        comparison_df = comparison_df.sort_values('F1-Score', ascending=False)

        # Add ranking
        comparison_df.insert(0, 'Rank', range(1, len(comparison_df) + 1))

        logger.info("\n" + comparison_df.to_string(index=False))
        logger.info("\n" + "="*70)

        # Identify best model for each metric
        logger.info("\n🥇 BEST MODELS BY METRIC:")
        logger.info("="*70)
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
            best_model = comparison_df.loc[comparison_df[metric].idxmax(), 'Model']
            best_score = comparison_df[metric].max()
            logger.info(f"  {metric:12s}: {best_model:30s} ({best_score:.4f})")
        logger.info("="*70)

        return comparison_df

    def save_evaluation_results(
        self,
        results: Dict[str, Dict[str, Any]],
        comparison_df: pd.DataFrame,
        output_dir: str = "data/reports"
    ):
        """
        Save evaluation results to disk.

        Creates both JSON (detailed) and CSV (summary) reports.

        Parameters:
        -----------
        results : Dict
            Detailed results for each model
        comparison_df : DataFrame
            Comparison table
        output_dir : str
            Where to save reports
        """
        logger.info(f"\n💾 Saving evaluation results to {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)

        # Save detailed results as JSON
        json_path = os.path.join(output_dir, "evaluation_results.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"  ✅ Saved detailed results: {json_path}")

        # Save comparison table as CSV
        csv_path = os.path.join(output_dir, "model_comparison.csv")
        comparison_df.to_csv(csv_path, index=False)
        logger.info(f"  ✅ Saved comparison table: {csv_path}")

        # Save timestamp
        metadata = {
            'evaluation_date': datetime.now().isoformat(),
            'num_models_evaluated': len(results),
            'output_directory': output_dir
        }
        metadata_path = os.path.join(output_dir, "evaluation_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"  ✅ Saved metadata: {metadata_path}")

        logger.info("\n✅ All evaluation results saved!")

    def generate_evaluation_report(
        self,
        results: Dict[str, Dict[str, Any]],
        output_dir: str = "data/reports"
    ):
        """
        Generate a comprehensive evaluation report.

        Creates a human-readable markdown report.

        Parameters:
        -----------
        results : Dict
            Evaluation results
        output_dir : str
            Where to save report
        """
        logger.info("📝 Generating evaluation report...")
        os.makedirs(output_dir, exist_ok=True)

        report_path = os.path.join(output_dir, "evaluation_report.md")

        with open(report_path, 'w') as f:
            f.write("# Banking AML Fraud Detection - Model Evaluation Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")

            f.write("## Executive Summary\n\n")
            f.write(f"- **Models Evaluated:** {len(results)}\n")
            f.write(f"- **Test Samples:** {list(results.values())[0]['test_samples']:,}\n")
            f.write(f"- **Number of Classes:** {list(results.values())[0]['num_classes']}\n\n")

            f.write("---\n\n")
            f.write("## Model Performance Comparison\n\n")

            # Create comparison table
            comparison_data = []
            for model_name, result in results.items():
                comparison_data.append({
                    'Model': model_name,
                    'Accuracy': f"{result.get('accuracy', 0):.4f}",
                    'Precision': f"{result.get('weighted_precision', 0):.4f}",
                    'Recall': f"{result.get('weighted_recall', 0):.4f}",
                    'F1-Score': f"{result.get('f1_score', 0):.4f}"
                })

            comparison_df = pd.DataFrame(comparison_data)
            f.write(comparison_df.to_markdown(index=False))
            f.write("\n\n")

            f.write("---\n\n")
            f.write("## Detailed Results by Model\n\n")

            for model_name, result in results.items():
                f.write(f"### {model_name}\n\n")
                f.write(f"- **Accuracy:** {result['accuracy']:.4f}\n")
                f.write(f"- **Weighted Precision:** {result['weighted_precision']:.4f}\n")
                f.write(f"- **Weighted Recall:** {result['weighted_recall']:.4f}\n")
                f.write(f"- **F1-Score:** {result['f1_score']:.4f}\n\n")

                if 'class_metrics' in result:
                    f.write("#### Per-Class Metrics\n\n")
                    for class_id, metrics in result['class_metrics'].items():
                        f.write(f"**Class {class_id}:**\n")
                        f.write(f"- Precision: {metrics['precision']:.4f}\n")
                        f.write(f"- Recall: {metrics['recall']:.4f}\n")
                        f.write(f"- F1-Score: {metrics['f1_score']:.4f}\n")
                        f.write(f"- Support: {metrics['support']}\n\n")

                f.write("---\n\n")

        logger.info(f"  ✅ Report saved: {report_path}")


if __name__ == "__main__":
    """
    Demo of evaluation module.
    """
    logger.info("📊 Model Evaluation Demo")
