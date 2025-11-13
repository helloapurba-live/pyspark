"""
=============================================================================
MODEL EVALUATION AND COMPARISON
=============================================================================
This module evaluates and compares ML models for fraud detection.

Think of this as a "judge" that scores each model's performance!

For fraud detection, we care about:
1. ACCURACY: How often is the model correct overall?
2. PRECISION: When it says "fraud", is it usually right? (avoid false alarms)
3. RECALL: Does it catch most of the actual fraud? (don't miss fraudsters!)
4. F1-SCORE: Balance between precision and recall
5. AUC-ROC: Overall ability to distinguish fraud from legitimate
6. AUC-PR: Precision-Recall curve (better for imbalanced data like fraud)

In banking, missing fraud is expensive, but false alarms annoy customers!
We need the right balance.
"""

from pyspark.sql import DataFrame
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)
from pyspark.sql.functions import col, when
from typing import Dict, List, Tuple, Any
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    average_precision_score
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluates and compares ML models.

    This class is like a "test grader" - it runs models through tests
    and gives them scores!
    """

    def __init__(self):
        """Initialize the evaluator."""
        self.evaluation_results = {}

        logger.info("📊 Model Evaluator initialized")

    def evaluate_model(self, predictions: DataFrame, model_name: str) -> Dict[str, float]:
        """
        Evaluate a single model's predictions.

        Parameters:
        -----------
        predictions : DataFrame
            DataFrame with 'prediction' and 'is_fraud' columns
        model_name : str
            Name of the model being evaluated

        Returns:
        --------
        Dict[str, float] : Dictionary of metric name -> value
        """
        logger.info(f"📊 Evaluating {model_name}...")

        metrics = {}

        try:
            # ================================================================
            # BINARY CLASSIFICATION METRICS
            # ================================================================

            # AUC-ROC (Area Under ROC Curve)
            # Measures ability to distinguish classes
            # 0.5 = random guessing, 1.0 = perfect classification
            evaluator_auc = BinaryClassificationEvaluator(
                labelCol="is_fraud",
                rawPredictionCol="rawPrediction",
                metricName="areaUnderROC"
            )
            auc_roc = evaluator_auc.evaluate(predictions)
            metrics['auc_roc'] = auc_roc

            # AUC-PR (Area Under Precision-Recall Curve)
            # Better metric for imbalanced data (like fraud detection!)
            evaluator_pr = BinaryClassificationEvaluator(
                labelCol="is_fraud",
                rawPredictionCol="rawPrediction",
                metricName="areaUnderPR"
            )
            auc_pr = evaluator_pr.evaluate(predictions)
            metrics['auc_pr'] = auc_pr

            # ================================================================
            # MULTICLASS METRICS (treating as binary but using multiclass evaluator)
            # ================================================================

            # Accuracy: (True Positives + True Negatives) / Total
            # Simple but can be misleading with imbalanced data!
            evaluator_acc = MulticlassClassificationEvaluator(
                labelCol="is_fraud",
                predictionCol="prediction",
                metricName="accuracy"
            )
            accuracy = evaluator_acc.evaluate(predictions)
            metrics['accuracy'] = accuracy

            # Weighted Precision: Average precision weighted by class support
            evaluator_prec = MulticlassClassificationEvaluator(
                labelCol="is_fraud",
                predictionCol="prediction",
                metricName="weightedPrecision"
            )
            precision = evaluator_prec.evaluate(predictions)
            metrics['precision'] = precision

            # Weighted Recall: Average recall weighted by class support
            evaluator_rec = MulticlassClassificationEvaluator(
                labelCol="is_fraud",
                predictionCol="prediction",
                metricName="weightedRecall"
            )
            recall = evaluator_rec.evaluate(predictions)
            metrics['recall'] = recall

            # F1 Score: Harmonic mean of precision and recall
            # Best single metric for fraud detection!
            evaluator_f1 = MulticlassClassificationEvaluator(
                labelCol="is_fraud",
                predictionCol="prediction",
                metricName="f1"
            )
            f1 = evaluator_f1.evaluate(predictions)
            metrics['f1'] = f1

            # ================================================================
            # CALCULATE CONFUSION MATRIX METRICS
            # ================================================================
            # This requires converting to pandas for sklearn metrics

            # Collect predictions (careful with large datasets!)
            pdf = predictions.select("is_fraud", "prediction").toPandas()
            y_true = pdf["is_fraud"].values
            y_pred = pdf["prediction"].values

            # Confusion Matrix
            # [True Neg  | False Pos]
            # [False Neg | True Pos ]
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

            metrics['true_positives'] = int(tp)
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)

            # ================================================================
            # BUSINESS METRICS (Important for AML!)
            # ================================================================

            # Fraud Detection Rate: % of actual fraud caught
            fraud_detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics['fraud_detection_rate'] = fraud_detection_rate

            # False Positive Rate: % of legitimate flagged as fraud
            false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
            metrics['false_positive_rate'] = false_positive_rate

            # False Negative Rate: % of fraud missed
            false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
            metrics['false_negative_rate'] = false_negative_rate

            # Specificity: True Negative Rate
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['specificity'] = specificity

            # ================================================================
            # PRINT RESULTS
            # ================================================================

            logger.info(f"\n{'='*70}")
            logger.info(f"📊 EVALUATION RESULTS FOR {model_name}")
            logger.info(f"{'='*70}")

            logger.info(f"\n🎯 Classification Metrics:")
            logger.info(f"   Accuracy:  {metrics['accuracy']:.4f}")
            logger.info(f"   Precision: {metrics['precision']:.4f}")
            logger.info(f"   Recall:    {metrics['recall']:.4f}")
            logger.info(f"   F1-Score:  {metrics['f1']:.4f}")

            logger.info(f"\n📈 AUC Metrics:")
            logger.info(f"   AUC-ROC:   {metrics['auc_roc']:.4f}")
            logger.info(f"   AUC-PR:    {metrics['auc_pr']:.4f}")

            logger.info(f"\n🎭 Confusion Matrix:")
            logger.info(f"   True Positives:  {metrics['true_positives']:,}")
            logger.info(f"   True Negatives:  {metrics['true_negatives']:,}")
            logger.info(f"   False Positives: {metrics['false_positives']:,}")
            logger.info(f"   False Negatives: {metrics['false_negatives']:,}")

            logger.info(f"\n💼 Business Metrics:")
            logger.info(f"   Fraud Detection Rate: {metrics['fraud_detection_rate']:.2%}")
            logger.info(f"   False Positive Rate:  {metrics['false_positive_rate']:.2%}")
            logger.info(f"   False Negative Rate:  {metrics['false_negative_rate']:.2%}")
            logger.info(f"   Specificity:          {metrics['specificity']:.2%}")

            logger.info(f"{'='*70}\n")

            # Store results
            self.evaluation_results[model_name] = metrics

        except Exception as e:
            logger.error(f"❌ Error evaluating {model_name}: {str(e)}")
            return {}

        return metrics

    def compare_models(self, results: Dict[str, Dict[str, float]]) -> DataFrame:
        """
        Compare multiple models.

        This creates a "leaderboard" of models!

        Parameters:
        -----------
        results : Dict[str, Dict[str, float]]
            Dictionary of model_name -> metrics

        Returns:
        --------
        DataFrame : Comparison table
        """
        logger.info("\n" + "="*80)
        logger.info("🏆 MODEL COMPARISON LEADERBOARD")
        logger.info("="*80 + "\n")

        import pandas as pd

        # Convert to DataFrame
        comparison_df = pd.DataFrame(results).T

        # Sort by F1 score (best metric for fraud detection)
        comparison_df = comparison_df.sort_values('f1', ascending=False)

        # Round for readability
        comparison_df = comparison_df.round(4)

        # Print comparison
        logger.info(comparison_df.to_string())

        # Print winner
        best_model = comparison_df.index[0]
        best_f1 = comparison_df.loc[best_model, 'f1']

        logger.info(f"\n🥇 WINNER: {best_model}")
        logger.info(f"   F1-Score: {best_f1:.4f}")

        logger.info("\n" + "="*80 + "\n")

        return comparison_df

    def plot_confusion_matrix(self, predictions: DataFrame, model_name: str,
                             save_path: str = None):
        """
        Plot confusion matrix heatmap.

        Confusion matrix shows:
        - How many fraud cases we caught (True Positives)
        - How many legitimate we correctly identified (True Negatives)
        - How many false alarms we had (False Positives)
        - How many fraud we missed (False Negatives)

        Parameters:
        -----------
        predictions : DataFrame
            Predictions DataFrame
        model_name : str
            Name of model
        save_path : str, optional
            Path to save plot
        """
        logger.info(f"📊 Plotting confusion matrix for {model_name}...")

        # Get predictions as numpy arrays
        pdf = predictions.select("is_fraud", "prediction").toPandas()
        y_true = pdf["is_fraud"].values
        y_pred = pdf["prediction"].values

        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Create plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)

        plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)

        # Add labels
        plt.xticks([0.5, 1.5], ['Legitimate', 'Fraud'])
        plt.yticks([0.5, 1.5], ['Legitimate', 'Fraud'])

        # Add text annotations
        tn, fp, fn, tp = cm.ravel()
        plt.text(0.5, -0.15, f'TN: {tn:,}', ha='center', transform=plt.gca().transAxes)
        plt.text(0.5, -0.2, '(Correctly identified legitimate)', ha='center',
                transform=plt.gca().transAxes, fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"   💾 Saved to {save_path}")

        return plt.gcf()

    def plot_roc_curve(self, predictions: DataFrame, model_name: str,
                      save_path: str = None):
        """
        Plot ROC (Receiver Operating Characteristic) curve.

        ROC curve shows trade-off between:
        - True Positive Rate (catching fraud)
        - False Positive Rate (false alarms)

        Parameters:
        -----------
        predictions : DataFrame
            Predictions DataFrame
        model_name : str
            Name of model
        save_path : str, optional
            Path to save plot
        """
        logger.info(f"📈 Plotting ROC curve for {model_name}...")

        # Get predictions
        pdf = predictions.select("is_fraud", "probability").toPandas()
        y_true = pdf["is_fraud"].values

        # Extract probability of fraud class
        y_score = np.array([p[1] for p in pdf["probability"].values])

        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        # Create plot
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - {model_name}', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"   💾 Saved to {save_path}")

        return plt.gcf()

    def plot_precision_recall_curve(self, predictions: DataFrame, model_name: str,
                                   save_path: str = None):
        """
        Plot Precision-Recall curve.

        Better than ROC for imbalanced datasets like fraud!
        Shows trade-off between precision and recall.

        Parameters:
        -----------
        predictions : DataFrame
            Predictions DataFrame
        model_name : str
            Name of model
        save_path : str, optional
            Path to save plot
        """
        logger.info(f"📊 Plotting Precision-Recall curve for {model_name}...")

        # Get predictions
        pdf = predictions.select("is_fraud", "probability").toPandas()
        y_true = pdf["is_fraud"].values
        y_score = np.array([p[1] for p in pdf["probability"].values])

        # Calculate PR curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        avg_precision = average_precision_score(y_true, y_score)

        # Create plot
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='darkgreen', lw=2,
                label=f'PR curve (AP = {avg_precision:.3f})')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curve - {model_name}', fontsize=16, fontweight='bold')
        plt.legend(loc="lower left", fontsize=12)
        plt.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"   💾 Saved to {save_path}")

        return plt.gcf()

    def plot_feature_importance(self, model: Any, feature_names: List[str],
                               model_name: str, top_n: int = 20,
                               save_path: str = None):
        """
        Plot feature importance.

        Shows which features are most useful for detecting fraud!

        Parameters:
        -----------
        model : Any
            Trained model with feature importances
        feature_names : List[str]
            Names of features
        model_name : str
            Name of model
        top_n : int
            Number of top features to show
        save_path : str, optional
            Path to save plot
        """
        logger.info(f"📊 Plotting feature importance for {model_name}...")

        try:
            # Get feature importances (method varies by model type)
            if hasattr(model, 'featureImportances'):
                importances = model.featureImportances.toArray()
            elif hasattr(model, 'coefficients'):
                importances = np.abs(model.coefficients.toArray())
            else:
                logger.warning(f"   ⚠️  Model {model_name} doesn't support feature importance")
                return None

            # Create DataFrame
            import pandas as pd
            importance_df = pd.DataFrame({
                'feature': feature_names[:len(importances)],
                'importance': importances
            })

            # Sort and get top N
            importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)

            # Create plot
            plt.figure(figsize=(12, 8))
            plt.barh(range(len(importance_df)), importance_df['importance'])
            plt.yticks(range(len(importance_df)), importance_df['feature'])
            plt.xlabel('Importance', fontsize=12)
            plt.title(f'Top {top_n} Feature Importances - {model_name}',
                     fontsize=16, fontweight='bold')
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"   💾 Saved to {save_path}")

            return plt.gcf()

        except Exception as e:
            logger.error(f"   ❌ Error plotting feature importance: {str(e)}")
            return None


def main():
    """
    Test the model evaluation module.
    """
    print("\n" + "="*80)
    print("📊 TESTING MODEL EVALUATION")
    print("="*80 + "\n")

    # This would normally use real predictions
    logger.info("✅ Model evaluation module ready!")


if __name__ == "__main__":
    main()
