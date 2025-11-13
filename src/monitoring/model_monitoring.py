"""
=============================================================================
MODEL MONITORING AND DRIFT DETECTION
=============================================================================
This module monitors deployed models for:
1. DATA DRIFT: Is incoming data different from training data?
2. CONCEPT DRIFT: Are patterns changing over time?
3. PERFORMANCE DEGRADATION: Is accuracy dropping?
4. SYSTEM HEALTH: Is the service working properly?

Think of this as a "health monitor" for your ML system!

WHY MONITORING MATTERS:
- Fraud patterns change over time (fraudsters adapt!)
- Data distributions shift (seasonality, new products, etc.)
- Models can degrade without you noticing
- Early detection prevents costly mistakes

Example: A model trained on 2020 data might not work well on 2024 data
because fraud tactics have evolved!
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelMonitor:
    """
    Monitors model performance and detects drift.

    This is like a "security guard" watching your model!
    """

    def __init__(self, alert_threshold: float = 0.05):
        """
        Initialize model monitor.

        Parameters:
        -----------
        alert_threshold : float
            Threshold for raising alerts (default: 0.05 for p-value tests)
        """
        self.alert_threshold = alert_threshold
        self.baseline_data = None
        self.drift_history = []
        self.performance_history = []

        logger.info("🔍 Model Monitor initialized")
        logger.info(f"   Alert threshold: {alert_threshold}")

    def set_baseline(self, training_data: pd.DataFrame):
        """
        Set baseline data distribution from training data.

        This is what we compare new data against!

        Parameters:
        -----------
        training_data : pd.DataFrame
            Training data to use as baseline
        """
        logger.info("📊 Setting baseline data distribution...")

        self.baseline_data = training_data
        self.baseline_stats = {
            'mean': training_data.mean(),
            'std': training_data.std(),
            'min': training_data.min(),
            'max': training_data.max(),
            'quantiles': training_data.quantile([0.25, 0.5, 0.75])
        }

        logger.info(f"   ✓ Baseline set with {len(training_data)} samples")

    def detect_data_drift_ks_test(self, new_data: pd.DataFrame,
                                   column: str) -> Tuple[bool, float, str]:
        """
        Detect data drift using Kolmogorov-Smirnov test.

        The KS test checks if two distributions are different.

        How it works:
        - Compares baseline data distribution to new data
        - Returns p-value: low p-value = distributions are different
        - If p-value < threshold → DRIFT DETECTED!

        Parameters:
        -----------
        new_data : pd.DataFrame
            New data to check
        column : str
            Column to check for drift

        Returns:
        --------
        Tuple[bool, float, str] : (drift_detected, p_value, message)
        """
        if self.baseline_data is None:
            logger.warning("⚠️  No baseline data set!")
            return False, 0.0, "No baseline"

        # Get baseline and new distributions
        baseline_values = self.baseline_data[column].dropna()
        new_values = new_data[column].dropna()

        # Perform KS test
        statistic, p_value = stats.ks_2samp(baseline_values, new_values)

        # Check for drift
        drift_detected = p_value < self.alert_threshold

        if drift_detected:
            message = f"⚠️  DRIFT DETECTED in {column}! (p-value: {p_value:.4f})"
            logger.warning(message)
        else:
            message = f"✓ No drift in {column} (p-value: {p_value:.4f})"
            logger.info(message)

        # Log drift event
        self.drift_history.append({
            'timestamp': datetime.now(),
            'column': column,
            'drift_detected': drift_detected,
            'p_value': p_value,
            'statistic': statistic
        })

        return drift_detected, p_value, message

    def detect_data_drift_all_features(self, new_data: pd.DataFrame) -> Dict:
        """
        Check all numerical features for drift.

        Parameters:
        -----------
        new_data : pd.DataFrame
            New data to check

        Returns:
        --------
        Dict : Drift detection results for all features
        """
        logger.info("🔍 Checking all features for data drift...")

        results = {}
        drift_count = 0

        # Check each numerical column
        for column in new_data.select_dtypes(include=[np.number]).columns:
            if column in self.baseline_data.columns:
                drift_detected, p_value, message = self.detect_data_drift_ks_test(
                    new_data, column
                )

                results[column] = {
                    'drift_detected': drift_detected,
                    'p_value': p_value
                }

                if drift_detected:
                    drift_count += 1

        logger.info(f"\n📊 Drift Detection Summary:")
        logger.info(f"   Features checked: {len(results)}")
        logger.info(f"   Drift detected: {drift_count}")
        logger.info(f"   Drift rate: {drift_count/len(results)*100:.1f}%")

        if drift_count > 0:
            logger.warning(f"\n⚠️  WARNING: Drift detected in {drift_count} features!")
            logger.warning("   Consider retraining the model with recent data.")

        return results

    def monitor_performance(self, true_labels: np.ndarray,
                          predictions: np.ndarray,
                          timestamp: datetime = None) -> Dict:
        """
        Monitor model performance over time.

        Tracks if model accuracy is dropping!

        Parameters:
        -----------
        true_labels : np.ndarray
            True fraud labels
        predictions : np.ndarray
            Model predictions
        timestamp : datetime, optional
            Timestamp for this batch

        Returns:
        --------
        Dict : Performance metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        if timestamp is None:
            timestamp = datetime.now()

        logger.info(f"📊 Monitoring performance at {timestamp}...")

        # Calculate metrics
        metrics = {
            'timestamp': timestamp,
            'accuracy': accuracy_score(true_labels, predictions),
            'precision': precision_score(true_labels, predictions, zero_division=0),
            'recall': recall_score(true_labels, predictions, zero_division=0),
            'f1': f1_score(true_labels, predictions, zero_division=0),
            'samples': len(true_labels)
        }

        # Add to history
        self.performance_history.append(metrics)

        # Check for performance degradation
        if len(self.performance_history) > 1:
            previous_f1 = self.performance_history[-2]['f1']
            current_f1 = metrics['f1']
            f1_drop = previous_f1 - current_f1

            if f1_drop > 0.1:  # 10% drop
                logger.warning(f"⚠️  PERFORMANCE DEGRADATION DETECTED!")
                logger.warning(f"   F1-Score dropped from {previous_f1:.4f} to {current_f1:.4f}")
                logger.warning(f"   Drop: {f1_drop:.4f} ({f1_drop/previous_f1*100:.1f}%)")

        # Print metrics
        logger.info(f"   Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"   Precision: {metrics['precision']:.4f}")
        logger.info(f"   Recall:    {metrics['recall']:.4f}")
        logger.info(f"   F1-Score:  {metrics['f1']:.4f}")

        return metrics

    def plot_performance_over_time(self, save_path: str = None):
        """
        Plot performance metrics over time.

        Visualizes if model is degrading!

        Parameters:
        -----------
        save_path : str, optional
            Path to save plot
        """
        if not self.performance_history:
            logger.warning("⚠️  No performance history to plot")
            return

        logger.info("📈 Plotting performance over time...")

        # Convert to DataFrame
        df = pd.DataFrame(self.performance_history)

        # Create plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot each metric
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        titles = ['Accuracy Over Time', 'Precision Over Time',
                 'Recall Over Time', 'F1-Score Over Time']

        for ax, metric, title in zip(axes.flat, metrics, titles):
            ax.plot(df['timestamp'], df[metric], marker='o', linewidth=2)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel(metric.capitalize(), fontsize=12)
            ax.grid(alpha=0.3)
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"   💾 Saved to {save_path}")

        return fig

    def plot_drift_heatmap(self, drift_results: Dict, save_path: str = None):
        """
        Plot heatmap of drift detection results.

        Visualizes which features are drifting!

        Parameters:
        -----------
        drift_results : Dict
            Results from detect_data_drift_all_features
        save_path : str, optional
            Path to save plot
        """
        logger.info("📊 Creating drift detection heatmap...")

        # Prepare data
        features = list(drift_results.keys())
        drift_status = [1 if drift_results[f]['drift_detected'] else 0 for f in features]
        p_values = [drift_results[f]['p_value'] for f in features]

        # Create DataFrame
        df = pd.DataFrame({
            'Feature': features,
            'Drift': drift_status,
            'P-Value': p_values
        })

        # Create plot
        fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.3)))

        # Create heatmap
        data_matrix = df[['Drift']].T
        sns.heatmap(data_matrix, annot=True, fmt='d', cmap='RdYlGn_r',
                   cbar=False, ax=ax, xticklabels=df['Feature'])

        ax.set_title('Data Drift Detection Heatmap', fontsize=16, fontweight='bold')
        ax.set_ylabel('')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"   💾 Saved to {save_path}")

        return fig

    def generate_monitoring_report(self) -> str:
        """
        Generate comprehensive monitoring report.

        Returns:
        --------
        str : Monitoring report
        """
        report = []
        report.append("="*80)
        report.append("📊 MODEL MONITORING REPORT")
        report.append("="*80)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Drift summary
        report.append("\n" + "-"*80)
        report.append("🔍 DRIFT DETECTION SUMMARY")
        report.append("-"*80)

        if self.drift_history:
            recent_drifts = [d for d in self.drift_history if d['drift_detected']]
            report.append(f"Total drift checks: {len(self.drift_history)}")
            report.append(f"Drift events detected: {len(recent_drifts)}")

            if recent_drifts:
                report.append("\nRecent drift events:")
                for drift in recent_drifts[-5:]:  # Last 5
                    report.append(f"  - {drift['column']}: p-value={drift['p_value']:.4f} ({drift['timestamp']})")
        else:
            report.append("No drift checks performed yet")

        # Performance summary
        report.append("\n" + "-"*80)
        report.append("📈 PERFORMANCE SUMMARY")
        report.append("-"*80)

        if self.performance_history:
            latest = self.performance_history[-1]
            report.append(f"Latest performance (as of {latest['timestamp']}):")
            report.append(f"  Accuracy:  {latest['accuracy']:.4f}")
            report.append(f"  Precision: {latest['precision']:.4f}")
            report.append(f"  Recall:    {latest['recall']:.4f}")
            report.append(f"  F1-Score:  {latest['f1']:.4f}")

            if len(self.performance_history) > 1:
                first = self.performance_history[0]
                report.append(f"\nPerformance change since {first['timestamp']}:")
                report.append(f"  Accuracy:  {latest['accuracy'] - first['accuracy']:+.4f}")
                report.append(f"  F1-Score:  {latest['f1'] - first['f1']:+.4f}")
        else:
            report.append("No performance monitoring data yet")

        report.append("\n" + "="*80)

        report_text = "\n".join(report)
        logger.info(report_text)

        return report_text


def demo_monitoring():
    """
    Demonstrate model monitoring.
    """
    print("\n" + "="*80)
    print("🔍 DEMO: MODEL MONITORING")
    print("="*80 + "\n")

    logger.info("💡 Model monitoring tracks:")
    logger.info("   1. DATA DRIFT: Are feature distributions changing?")
    logger.info("   2. PERFORMANCE: Is model accuracy dropping?")
    logger.info("   3. ALERTS: Notify when issues detected")
    logger.info("\n💡 Example scenario:")
    logger.info("   - Model trained on 2020 fraud data")
    logger.info("   - Now it's 2024, fraud tactics have evolved")
    logger.info("   - Monitor detects drift in transaction patterns")
    logger.info("   - Alert sent: Time to retrain model!")


def main():
    """
    Test monitoring module.
    """
    print("\n" + "="*80)
    print("🔍 MODEL MONITORING MODULE")
    print("="*80 + "\n")

    demo_monitoring()

    logger.info("\n✅ Monitoring module ready!")


if __name__ == "__main__":
    main()
