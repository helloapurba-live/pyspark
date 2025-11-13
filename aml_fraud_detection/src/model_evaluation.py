"""
==============================================================================
MODEL EVALUATION AND COMPARISON
==============================================================================
This module evaluates and compares different clustering algorithms.

Why Evaluate?
We need to know which algorithm works best! We compare them using:
1. Clustering quality metrics (how good are the clusters?)
2. Fraud detection metrics (how well do they find fraud?)
3. Business metrics (cost, time, interpretability)

Author: Your Friendly AI Teacher
Date: 2025-11-13
==============================================================================
"""

from pyspark.sql.functions import *
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql import SparkSession
import json
import time
from datetime import datetime


class ModelEvaluator:
    """
    This class evaluates clustering models for fraud detection.

    Think of this as a "report card" for our algorithms!
    We grade them on:
    - Clustering quality
    - Fraud detection accuracy
    - Speed and efficiency
    - Business value
    """

    def __init__(self, spark, config):
        """
        Initialize the evaluator.

        Args:
            spark: PySpark session
            config: Configuration dictionary
        """
        self.spark = spark
        self.config = config
        self.evaluation_results = {}

        print("📊 Model Evaluator initialized!")

    def evaluate_all_models(self, clustering_results):
        """
        Evaluate all clustering models and compare them.

        This is like running a competition - all algorithms compete,
        and we crown the winners!

        Args:
            clustering_results: Dictionary of results from clustering algorithms

        Returns:
            Dictionary of evaluation metrics for each algorithm
        """
        print("\n" + "="*80)
        print("📊 MODEL EVALUATION AND COMPARISON")
        print("="*80)

        all_metrics = {}

        for algo_name, result in clustering_results.items():
            print(f"\n📈 Evaluating {algo_name.upper()}...")
            print("-" * 40)

            predictions_df = result['predictions']

            # Evaluate clustering quality
            clustering_metrics = self._evaluate_clustering_quality(
                predictions_df, algo_name
            )

            # Evaluate fraud detection performance
            fraud_metrics = self._evaluate_fraud_detection(
                predictions_df, algo_name
            )

            # Calculate business metrics
            business_metrics = self._calculate_business_metrics(
                predictions_df, algo_name
            )

            # Combine all metrics
            all_metrics[algo_name] = {
                'clustering_quality': clustering_metrics,
                'fraud_detection': fraud_metrics,
                'business_metrics': business_metrics,
                'algorithm': result['algorithm']
            }

            print(f"✅ {algo_name} evaluation complete!")

        # Compare all models
        print("\n" + "="*80)
        print("🏆 MODEL COMPARISON")
        print("="*80)
        self._compare_models(all_metrics)

        # Save evaluation results
        self.evaluation_results = all_metrics

        return all_metrics

    def _evaluate_clustering_quality(self, df, algo_name):
        """
        Evaluate clustering quality using standard metrics.

        Metrics explained (in plain English!):

        1. Silhouette Score (-1 to 1):
           - Measures how well-separated clusters are
           - Higher is better
           - > 0.5 = Good separation
           - < 0.3 = Weak separation

        2. Within-Cluster Sum of Squares (WCSS):
           - Average distance of points to their cluster center
           - Lower is better
           - Measures cluster "tightness"

        3. Between-Cluster Variation:
           - How different are clusters from each other?
           - Higher is better
           - Measures cluster "distinctness"
        """
        metrics = {}

        # Find the cluster column for this algorithm
        cluster_col = self._find_cluster_column(df, algo_name)

        if cluster_col is None:
            print(f"   ⚠️  No cluster column found for {algo_name}")
            return metrics

        try:
            # Silhouette Score
            print("   📐 Calculating Silhouette Score...")
            evaluator = ClusteringEvaluator(
                featuresCol='features',
                predictionCol=cluster_col,
                metricName='silhouette'
            )
            silhouette = evaluator.evaluate(df)
            metrics['silhouette_score'] = silhouette
            print(f"      Silhouette Score: {silhouette:.4f}")

            # Cluster size distribution
            print("   📊 Analyzing cluster distribution...")
            cluster_dist = df.groupBy(cluster_col).count().collect()
            cluster_sizes = [row['count'] for row in cluster_dist]

            metrics['num_clusters'] = len(cluster_sizes)
            metrics['min_cluster_size'] = min(cluster_sizes)
            metrics['max_cluster_size'] = max(cluster_sizes)
            metrics['avg_cluster_size'] = sum(cluster_sizes) / len(cluster_sizes)

            # Balance score (how evenly distributed are clusters?)
            total_points = sum(cluster_sizes)
            ideal_size = total_points / len(cluster_sizes)
            balance_score = 1 - (sum([abs(s - ideal_size) for s in cluster_sizes]) /
                                (2 * total_points))
            metrics['balance_score'] = balance_score

            print(f"      Number of clusters: {metrics['num_clusters']}")
            print(f"      Balance score: {balance_score:.4f}")

        except Exception as e:
            print(f"   ⚠️  Error calculating clustering metrics: {str(e)}")

        return metrics

    def _evaluate_fraud_detection(self, df, algo_name):
        """
        Evaluate fraud detection performance.

        Metrics explained (fraud detection perspective):

        1. Precision (How accurate are our alerts?):
           - Of all transactions we flagged, how many are actually fraud?
           - Precision = True Positives / (True Positives + False Positives)
           - High precision = Few false alarms

        2. Recall (How many frauds do we catch?):
           - Of all fraud transactions, how many did we catch?
           - Recall = True Positives / (True Positives + False Negatives)
           - High recall = Catching most frauds

        3. F1-Score (Balance of Precision and Recall):
           - Harmonic mean of precision and recall
           - F1 = 2 * (Precision * Recall) / (Precision + Recall)
           - Higher is better

        4. False Positive Rate (How many false alarms?):
           - Of all normal transactions, how many did we wrongly flag?
           - Lower is better (fewer false alarms)

        Think of it like airport security:
        - High precision = Few innocent people stopped
        - High recall = Catching most dangerous items
        - Low FPR = Not annoying too many passengers
        """
        metrics = {}

        # Find anomaly score column
        score_col = self._find_anomaly_score_column(df, algo_name)

        if score_col is None or 'is_suspicious' not in df.columns:
            print(f"   ⚠️  Cannot evaluate fraud detection for {algo_name}")
            return metrics

        try:
            # Set threshold at 85th percentile (flag top 15% as fraud)
            threshold = df.approxQuantile(score_col, [0.85], 0.01)[0]

            # Create predictions
            df_with_pred = df.withColumn(
                'predicted_fraud',
                when(col(score_col) > threshold, 1).otherwise(0)
            )

            # Calculate confusion matrix
            tp = df_with_pred.filter(
                (col('predicted_fraud') == 1) & (col('is_suspicious') == 1)
            ).count()

            fp = df_with_pred.filter(
                (col('predicted_fraud') == 1) & (col('is_suspicious') == 0)
            ).count()

            tn = df_with_pred.filter(
                (col('predicted_fraud') == 0) & (col('is_suspicious') == 0)
            ).count()

            fn = df_with_pred.filter(
                (col('predicted_fraud') == 0) & (col('is_suspicious') == 1)
            ).count()

            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn)

            metrics['precision'] = precision
            metrics['recall'] = recall
            metrics['f1_score'] = f1_score
            metrics['false_positive_rate'] = fpr
            metrics['accuracy'] = accuracy
            metrics['true_positives'] = tp
            metrics['false_positives'] = fp
            metrics['true_negatives'] = tn
            metrics['false_negatives'] = fn

            print(f"   🎯 Fraud Detection Performance:")
            print(f"      Precision: {precision:.4f} (How accurate are alerts)")
            print(f"      Recall: {recall:.4f} (How many frauds caught)")
            print(f"      F1-Score: {f1_score:.4f} (Overall balance)")
            print(f"      False Positive Rate: {fpr:.4f} (False alarm rate)")
            print(f"      Accuracy: {accuracy:.4f}")

        except Exception as e:
            print(f"   ⚠️  Error calculating fraud metrics: {str(e)}")

        return metrics

    def _calculate_business_metrics(self, df, algo_name):
        """
        Calculate business-relevant metrics.

        Business metrics explained:

        1. Cost of False Positives:
           - Each false alarm costs money (investigation time)
           - Assume $50 per false positive

        2. Cost of False Negatives:
           - Missing fraud costs money (actual fraud losses)
           - Assume $1000 per missed fraud

        3. Total Business Cost:
           - Sum of false positive and false negative costs

        4. Alert Rate:
           - What % of transactions trigger alerts?
           - Too high = Too many alerts (alert fatigue)
           - Too low = Missing fraud

        5. Detection Speed:
           - How fast can we process transactions?
           - Measured in transactions per second

        Think of this as the "bottom line" - what does it mean for the business?
        """
        metrics = {}

        # Find score column
        score_col = self._find_anomaly_score_column(df, algo_name)

        if score_col is None:
            return metrics

        try:
            # Threshold for alerts
            threshold = df.approxQuantile(score_col, [0.85], 0.01)[0]

            # Count alerts
            num_alerts = df.filter(col(score_col) > threshold).count()
            total_transactions = df.count()
            alert_rate = num_alerts / total_transactions

            metrics['alert_rate'] = alert_rate
            metrics['num_alerts'] = num_alerts

            # Calculate costs (if we have fraud labels)
            if 'is_suspicious' in df.columns:
                df_with_pred = df.withColumn(
                    'predicted_fraud',
                    when(col(score_col) > threshold, 1).otherwise(0)
                )

                fp = df_with_pred.filter(
                    (col('predicted_fraud') == 1) & (col('is_suspicious') == 0)
                ).count()

                fn = df_with_pred.filter(
                    (col('predicted_fraud') == 0) & (col('is_suspicious') == 1)
                ).count()

                # Business costs (example values)
                cost_per_fp = 50  # $50 to investigate false positive
                cost_per_fn = 1000  # $1000 average fraud loss

                total_cost = (fp * cost_per_fp) + (fn * cost_per_fn)

                metrics['false_positive_cost'] = fp * cost_per_fp
                metrics['false_negative_cost'] = fn * cost_per_fn
                metrics['total_cost'] = total_cost
                metrics['cost_per_transaction'] = total_cost / total_transactions

                print(f"   💰 Business Metrics:")
                print(f"      Alert rate: {alert_rate:.2%}")
                print(f"      Total cost: ${total_cost:,.2f}")
                print(f"      Cost per transaction: ${metrics['cost_per_transaction']:.2f}")

            # Interpretability score (subjective, based on algorithm type)
            interpretability_scores = {
                'kmeans': 0.8,
                'bisecting_kmeans': 0.7,
                'gmm': 0.6,
                'pca_clustering': 0.5,
                'zscore_anomaly': 1.0,  # Very interpretable!
                'distance_outlier': 0.7,
                'density_anomaly': 0.6,
                'ensemble': 0.5,
                'risk_score_clustering': 1.0,  # Very interpretable!
                'behavioral': 0.9,
                'graph_clustering': 0.7,
                'temporal': 0.8
            }

            metrics['interpretability'] = interpretability_scores.get(algo_name, 0.5)

        except Exception as e:
            print(f"   ⚠️  Error calculating business metrics: {str(e)}")

        return metrics

    def _compare_models(self, all_metrics):
        """
        Compare all models and rank them.

        This creates a "leaderboard" showing which algorithms perform best!
        """
        print("\n📋 MODEL LEADERBOARD")
        print("-" * 80)

        # Create comparison table
        comparison = []

        for algo_name, metrics in all_metrics.items():
            clustering = metrics.get('clustering_quality', {})
            fraud = metrics.get('fraud_detection', {})
            business = metrics.get('business_metrics', {})

            row = {
                'algorithm': algo_name,
                'silhouette': clustering.get('silhouette_score', 0),
                'f1_score': fraud.get('f1_score', 0),
                'precision': fraud.get('precision', 0),
                'recall': fraud.get('recall', 0),
                'alert_rate': business.get('alert_rate', 0),
                'interpretability': business.get('interpretability', 0)
            }

            comparison.append(row)

        # Sort by F1-score (best fraud detection performance)
        comparison.sort(key=lambda x: x['f1_score'], reverse=True)

        # Print table
        print(f"{'Rank':<6}{'Algorithm':<25}{'F1':<8}{'Precision':<12}{'Recall':<10}{'Silhouette':<12}")
        print("-" * 80)

        for i, row in enumerate(comparison, 1):
            print(f"{i:<6}{row['algorithm']:<25}{row['f1_score']:<8.3f}"
                  f"{row['precision']:<12.3f}{row['recall']:<10.3f}{row['silhouette']:<12.3f}")

        # Identify best models for different purposes
        print("\n🏆 BEST MODELS BY CATEGORY:")
        print("-" * 80)

        best_f1 = max(comparison, key=lambda x: x['f1_score'])
        print(f"🥇 Best Overall Fraud Detection: {best_f1['algorithm']} (F1: {best_f1['f1_score']:.3f})")

        best_precision = max(comparison, key=lambda x: x['precision'])
        print(f"🎯 Best Precision (Fewest False Alarms): {best_precision['algorithm']} ({best_precision['precision']:.3f})")

        best_recall = max(comparison, key=lambda x: x['recall'])
        print(f"🔍 Best Recall (Catches Most Fraud): {best_recall['algorithm']} ({best_recall['recall']:.3f})")

        best_silhouette = max(comparison, key=lambda x: x['silhouette'])
        print(f"📊 Best Clustering Quality: {best_silhouette['algorithm']} ({best_silhouette['silhouette']:.3f})")

        best_interpretable = max(comparison, key=lambda x: x['interpretability'])
        print(f"📖 Most Interpretable: {best_interpretable['algorithm']} ({best_interpretable['interpretability']:.2f})")

        # Calculate overall score (weighted combination)
        print("\n🎖️  OVERALL WINNER (Weighted Score):")
        print("-" * 80)

        for row in comparison:
            # Weighted score: 40% F1, 20% precision, 20% recall, 20% interpretability
            row['overall_score'] = (
                row['f1_score'] * 0.4 +
                row['precision'] * 0.2 +
                row['recall'] * 0.2 +
                row['interpretability'] * 0.2
            )

        winner = max(comparison, key=lambda x: x['overall_score'])
        print(f"🏆 WINNER: {winner['algorithm']}")
        print(f"   Overall Score: {winner['overall_score']:.3f}")
        print(f"   F1-Score: {winner['f1_score']:.3f}")
        print(f"   Precision: {winner['precision']:.3f}")
        print(f"   Recall: {winner['recall']:.3f}")

        return comparison

    def _find_cluster_column(self, df, algo_name):
        """Find the cluster prediction column for an algorithm."""
        possible_columns = [
            f'{algo_name}_cluster',
            'prediction',
            'cluster'
        ]

        for col in possible_columns:
            if col in df.columns:
                return col

        return None

    def _find_anomaly_score_column(self, df, algo_name):
        """Find the anomaly score column for an algorithm."""
        possible_columns = [
            f'{algo_name}_anomaly_score',
            'anomaly_score',
            'score'
        ]

        for col in possible_columns:
            if col in df.columns:
                return col

        return None

    def save_evaluation_results(self, output_path):
        """
        Save evaluation results to file.
        """
        print(f"\n💾 Saving evaluation results to: {output_path}")

        # Convert to JSON-serializable format
        results_json = json.dumps(self.evaluation_results, indent=2, default=str)

        # Write to file (in a real system, this would be HDFS or S3)
        with open(output_path, 'w') as f:
            f.write(results_json)

        print("   ✓ Results saved!")

    def generate_evaluation_report(self, output_path):
        """
        Generate a comprehensive evaluation report.
        """
        print(f"\n📄 Generating evaluation report...")

        report = []
        report.append("="*80)
        report.append("AML FRAUD DETECTION - MODEL EVALUATION REPORT")
        report.append("="*80)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("\n\n")

        # Summary for each model
        for algo_name, metrics in self.evaluation_results.items():
            report.append(f"\n{algo_name.upper()}")
            report.append("-" * 80)

            # Clustering quality
            clustering = metrics.get('clustering_quality', {})
            report.append(f"\nClustering Quality:")
            for key, value in clustering.items():
                report.append(f"  {key}: {value}")

            # Fraud detection
            fraud = metrics.get('fraud_detection', {})
            report.append(f"\nFraud Detection Performance:")
            for key, value in fraud.items():
                report.append(f"  {key}: {value}")

            # Business metrics
            business = metrics.get('business_metrics', {})
            report.append(f"\nBusiness Metrics:")
            for key, value in business.items():
                report.append(f"  {key}: {value}")

            report.append("\n")

        # Write report
        with open(output_path, 'w') as f:
            f.write('\n'.join(report))

        print(f"   ✓ Report saved to: {output_path}")


def main():
    """
    Main function to run evaluation standalone.
    """
    print("\n" + "="*80)
    print("📊 MODEL EVALUATION")
    print("="*80)

    print("\n⚠️  This module requires clustering results!")
    print("   Please run the full pipeline first.")


if __name__ == "__main__":
    main()
