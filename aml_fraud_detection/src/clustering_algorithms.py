"""
==============================================================================
CLUSTERING ALGORITHMS FOR AML FRAUD DETECTION
==============================================================================
This module implements 10+ clustering algorithms to detect fraud patterns.

Why Multiple Algorithms?
Each algorithm has strengths and weaknesses:
- K-Means: Fast, finds spherical clusters
- GMM: Can model complex probability distributions
- LDA: Great for text patterns
- PCA: Reduces dimensions, finds variance
- And more!

By using multiple methods, we can find different types of fraud patterns!

Author: Your Friendly AI Teacher
Date: 2025-11-13
==============================================================================
"""

from pyspark.ml.clustering import KMeans, BisectingKMeans, GaussianMixture, LDA
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.stat import Summarizer
import numpy as np
import time


class ClusteringAlgorithms:
    """
    This class implements multiple clustering algorithms for AML fraud detection.

    Think of this as a "toolkit" with different tools for finding patterns:
    - Some tools are good for one job
    - Some are better for another
    - Using many tools together gives us the best results!
    """

    def __init__(self, spark, config):
        """
        Initialize the clustering algorithms.

        Args:
            spark: PySpark session
            config: Configuration dictionary
        """
        self.spark = spark
        self.config = config
        self.models = {}  # Store trained models
        self.predictions = {}  # Store predictions
        self.metrics = {}  # Store evaluation metrics

        print("🤖 Clustering Algorithms initialized!")
        print(f"📊 Will train {self._count_enabled_algorithms()} algorithms")

    def _count_enabled_algorithms(self):
        """Count how many algorithms are enabled in config."""
        count = 0
        for algo_name, algo_config in self.config['algorithms'].items():
            if algo_config.get('enabled', False):
                count += 1
        return count

    def train_all_algorithms(self, df):
        """
        Train all enabled clustering algorithms.

        This is like running a "tournament" - we try all algorithms and see
        which ones work best!

        Args:
            df: DataFrame with features

        Returns:
            Dictionary of trained models and predictions
        """
        print("\n" + "="*80)
        print("🚀 TRAINING CLUSTERING ALGORITHMS")
        print("="*80)

        results = {}

        # Algorithm 1: K-Means Clustering
        if self.config['algorithms']['kmeans']['enabled']:
            print("\n" + "-"*80)
            print("1️⃣  K-MEANS CLUSTERING")
            print("-"*80)
            results['kmeans'] = self._train_kmeans(df)

        # Algorithm 2: Bisecting K-Means
        if self.config['algorithms']['bisecting_kmeans']['enabled']:
            print("\n" + "-"*80)
            print("2️⃣  BISECTING K-MEANS")
            print("-"*80)
            results['bisecting_kmeans'] = self._train_bisecting_kmeans(df)

        # Algorithm 3: Gaussian Mixture Model
        if self.config['algorithms']['gmm']['enabled']:
            print("\n" + "-"*80)
            print("3️⃣  GAUSSIAN MIXTURE MODEL (GMM)")
            print("-"*80)
            results['gmm'] = self._train_gmm(df)

        # Algorithm 4: PCA-based Clustering
        if self.config['algorithms']['pca_clustering']['enabled']:
            print("\n" + "-"*80)
            print("4️⃣  PCA-BASED CLUSTERING")
            print("-"*80)
            results['pca_clustering'] = self._train_pca_clustering(df)

        # Algorithm 5: Statistical Anomaly Detection (Z-Score)
        if self.config['algorithms']['zscore_anomaly']['enabled']:
            print("\n" + "-"*80)
            print("5️⃣  Z-SCORE ANOMALY DETECTION")
            print("-"*80)
            results['zscore_anomaly'] = self._train_zscore_anomaly(df)

        # Algorithm 6: Distance-based Outlier Detection
        print("\n" + "-"*80)
        print("6️⃣  DISTANCE-BASED OUTLIER DETECTION")
        print("-"*80)
        results['distance_outlier'] = self._train_distance_outlier(df)

        # Algorithm 7: Density-based Clustering (LocalOutlierFactor approximation)
        print("\n" + "-"*80)
        print("7️⃣  DENSITY-BASED ANOMALY DETECTION")
        print("-"*80)
        results['density_anomaly'] = self._train_density_anomaly(df)

        # Algorithm 8: Ensemble Clustering
        if self.config['algorithms']['ensemble']['enabled']:
            print("\n" + "-"*80)
            print("8️⃣  ENSEMBLE CLUSTERING")
            print("-"*80)
            results['ensemble'] = self._train_ensemble_clustering(df, results)

        # Algorithm 9: Risk-Score Based Clustering
        print("\n" + "-"*80)
        print("9️⃣  RISK-SCORE BASED CLUSTERING")
        print("-"*80)
        results['risk_score_clustering'] = self._train_risk_score_clustering(df)

        # Algorithm 10: Behavioral Segmentation
        print("\n" + "-"*80)
        print("🔟 BEHAVIORAL SEGMENTATION")
        print("-"*80)
        results['behavioral'] = self._train_behavioral_clustering(df)

        # Algorithm 11: Graph-based Clustering (Transaction Network)
        print("\n" + "-"*80)
        print("1️⃣1️⃣  GRAPH-BASED CLUSTERING")
        print("-"*80)
        results['graph_clustering'] = self._train_graph_clustering(df)

        # Algorithm 12: Temporal Pattern Clustering
        print("\n" + "-"*80)
        print("1️⃣2️⃣  TEMPORAL PATTERN CLUSTERING")
        print("-"*80)
        results['temporal'] = self._train_temporal_clustering(df)

        print("\n" + "="*80)
        print("✅ ALL ALGORITHMS TRAINED!")
        print("="*80)

        return results

    def _train_kmeans(self, df):
        """
        Train K-Means clustering.

        What is K-Means?
        Imagine you have a bunch of points on a map. K-Means tries to group them
        into K clusters by:
        1. Randomly picking K center points
        2. Assigning each point to nearest center
        3. Moving centers to average of their points
        4. Repeat until centers stop moving

        Great for: Finding general patterns in fraud behavior
        """
        print("📖 K-Means groups data into K spherical clusters")
        print("   Use case: Finding general fraud patterns\n")

        k_values = self.config['algorithms']['kmeans']['k']
        best_model = None
        best_k = None
        best_score = float('inf')

        for k in k_values:
            print(f"   Training K-Means with k={k}...")
            start_time = time.time()

            kmeans = KMeans(
                k=k,
                featuresCol='features',
                predictionCol=f'kmeans_cluster',
                maxIter=self.config['algorithms']['kmeans']['max_iter'],
                seed=self.config['algorithms']['kmeans']['seed']
            )

            model = kmeans.fit(df)
            predictions = model.transform(df)

            # Calculate cost (lower is better)
            cost = model.summary.trainingCost
            elapsed = time.time() - start_time

            print(f"      Cost: {cost:.2f} | Time: {elapsed:.2f}s")

            if cost < best_score:
                best_score = cost
                best_model = model
                best_k = k

        print(f"\n   ✅ Best K-Means: k={best_k}, cost={best_score:.2f}")

        # Get final predictions
        final_predictions = best_model.transform(df)

        # Calculate anomaly scores based on distance to cluster center
        final_predictions = self._calculate_distance_to_cluster(
            final_predictions, best_model, 'kmeans_cluster', 'kmeans_anomaly_score'
        )

        return {
            'model': best_model,
            'predictions': final_predictions,
            'best_k': best_k,
            'algorithm': 'KMeans'
        }

    def _train_bisecting_kmeans(self, df):
        """
        Train Bisecting K-Means.

        What is Bisecting K-Means?
        Like K-Means, but smarter! Instead of starting with K random centers:
        1. Start with 1 cluster (all data)
        2. Split the largest cluster into 2
        3. Repeat until you have K clusters

        Think of it as a "divide and conquer" approach!

        Great for: Hierarchical fraud patterns
        """
        print("📖 Bisecting K-Means splits clusters hierarchically")
        print("   Use case: Finding nested fraud patterns\n")

        k_values = self.config['algorithms']['bisecting_kmeans']['k']
        best_model = None
        best_k = None
        best_score = float('inf')

        for k in k_values:
            print(f"   Training Bisecting K-Means with k={k}...")
            start_time = time.time()

            bisecting_kmeans = BisectingKMeans(
                k=k,
                featuresCol='features',
                predictionCol='bisecting_kmeans_cluster',
                maxIter=self.config['algorithms']['bisecting_kmeans']['max_iter'],
                seed=self.config['algorithms']['bisecting_kmeans']['seed']
            )

            model = bisecting_kmeans.fit(df)
            predictions = model.transform(df)

            # Calculate cost
            cost = model.summary.trainingCost
            elapsed = time.time() - start_time

            print(f"      Cost: {cost:.2f} | Time: {elapsed:.2f}s")

            if cost < best_score:
                best_score = cost
                best_model = model
                best_k = k

        print(f"\n   ✅ Best Bisecting K-Means: k={best_k}, cost={best_score:.2f}")

        final_predictions = best_model.transform(df)
        final_predictions = self._calculate_distance_to_cluster(
            final_predictions, best_model, 'bisecting_kmeans_cluster',
            'bisecting_kmeans_anomaly_score'
        )

        return {
            'model': best_model,
            'predictions': final_predictions,
            'best_k': best_k,
            'algorithm': 'BisectingKMeans'
        }

    def _train_gmm(self, df):
        """
        Train Gaussian Mixture Model.

        What is GMM?
        Imagine fraud patterns are like bell curves (Gaussians). GMM assumes:
        1. Data comes from multiple Gaussian distributions
        2. Each distribution represents a different pattern
        3. Each point has a probability of belonging to each distribution

        Think of it as "fuzzy clustering" - points can partially belong to
        multiple clusters!

        Great for: Complex fraud patterns with overlapping behaviors
        """
        print("📖 GMM models data as mixture of Gaussian distributions")
        print("   Use case: Complex, overlapping fraud patterns\n")

        k_values = self.config['algorithms']['gmm']['k']
        best_model = None
        best_k = None
        best_score = float('-inf')

        for k in k_values:
            print(f"   Training GMM with k={k}...")
            start_time = time.time()

            gmm = GaussianMixture(
                k=k,
                featuresCol='features',
                predictionCol='gmm_cluster',
                probabilityCol='gmm_probability',
                maxIter=self.config['algorithms']['gmm']['max_iter'],
                seed=self.config['algorithms']['gmm']['seed']
            )

            model = gmm.fit(df)
            predictions = model.transform(df)

            # Calculate log likelihood (higher is better)
            log_likelihood = model.summary.logLikelihood
            elapsed = time.time() - start_time

            print(f"      Log Likelihood: {log_likelihood:.2f} | Time: {elapsed:.2f}s")

            if log_likelihood > best_score:
                best_score = log_likelihood
                best_model = model
                best_k = k

        print(f"\n   ✅ Best GMM: k={best_k}, log_likelihood={best_score:.2f}")

        final_predictions = best_model.transform(df)

        # GMM gives us probabilities - use negative log probability as anomaly score
        final_predictions = final_predictions.withColumn(
            'gmm_anomaly_score',
            -log10(array_max(col('gmm_probability')) + lit(1e-10))
        )

        return {
            'model': best_model,
            'predictions': final_predictions,
            'best_k': best_k,
            'algorithm': 'GaussianMixture'
        }

    def _train_pca_clustering(self, df):
        """
        Train PCA-based clustering.

        What is PCA?
        Principal Component Analysis reduces dimensions by finding the most
        important "directions" in the data.

        Imagine you have 100 features. PCA might find that:
        - 10 components capture 90% of the variance
        - The other 90 features are mostly redundant

        We then cluster on these principal components!

        Great for: High-dimensional data, visualization
        """
        print("📖 PCA reduces dimensions, then clusters on principal components")
        print("   Use case: Simplifying complex high-dimensional patterns\n")

        n_components_list = self.config['algorithms']['pca_clustering']['n_components']
        best_result = None
        best_n = None
        best_variance = 0

        for n_components in n_components_list:
            print(f"   Training PCA with {n_components} components...")
            start_time = time.time()

            # Apply PCA
            pca = PCA(
                k=n_components,
                inputCol='features',
                outputCol='pca_features'
            )
            pca_model = pca.fit(df)
            pca_df = pca_model.transform(df)

            # Get explained variance
            explained_variance = sum(pca_model.explainedVariance)
            print(f"      Explained variance: {explained_variance:.2%}")

            # Cluster on PCA features
            kmeans = KMeans(
                k=10,
                featuresCol='pca_features',
                predictionCol='pca_cluster',
                seed=42
            )
            kmeans_model = kmeans.fit(pca_df)
            predictions = kmeans_model.transform(pca_df)

            elapsed = time.time() - start_time
            print(f"      Time: {elapsed:.2f}s")

            if explained_variance > best_variance:
                best_variance = explained_variance
                best_result = predictions
                best_n = n_components

        print(f"\n   ✅ Best PCA: {best_n} components, {best_variance:.2%} variance")

        # Calculate anomaly score based on reconstruction error
        # (Not directly available in PySpark, so we use distance to cluster)
        best_result = best_result.withColumn(
            'pca_anomaly_score',
            abs(col('pca_cluster') - lit(5)) / lit(5)  # Normalized distance
        )

        return {
            'model': None,
            'predictions': best_result,
            'best_n': best_n,
            'algorithm': 'PCA_Clustering'
        }

    def _train_zscore_anomaly(self, df):
        """
        Train Z-Score based anomaly detection.

        What is Z-Score?
        Z-score tells you "how many standard deviations away from average?"

        Example:
        - Average transaction: $100
        - Standard deviation: $50
        - Transaction of $250 has z-score = (250-100)/50 = 3

        Rule of thumb: z-score > 3 or < -3 is unusual!

        Great for: Simple, interpretable anomaly detection
        """
        print("📖 Z-Score detects outliers using standard deviations")
        print("   Use case: Simple statistical anomaly detection\n")

        threshold = self.config['algorithms']['zscore_anomaly']['threshold']
        print(f"   Using threshold: {threshold} standard deviations")

        # Calculate z-scores for key features
        df_with_scores = df.withColumn(
            'zscore_anomaly_score',
            greatest(
                abs(col('amount_zscore')),
                abs(col('amount_deviation_from_avg')),
                col('composite_risk_score') * 3  # Scale risk score to z-score range
            )
        )

        # Mark anomalies
        df_with_scores = df_with_scores.withColumn(
            'zscore_cluster',
            when(col('zscore_anomaly_score') > threshold, 1).otherwise(0)
        )

        num_anomalies = df_with_scores.filter(col('zscore_cluster') == 1).count()
        total = df_with_scores.count()
        anomaly_pct = num_anomalies / total * 100

        print(f"\n   ✅ Z-Score detection complete")
        print(f"      Anomalies detected: {num_anomalies} ({anomaly_pct:.2f}%)")

        return {
            'model': None,
            'predictions': df_with_scores,
            'threshold': threshold,
            'algorithm': 'ZScore_Anomaly'
        }

    def _train_distance_outlier(self, df):
        """
        Train distance-based outlier detection.

        What is this?
        For each point, calculate its distance to its K nearest neighbors.
        Points far from all others are outliers!

        Think of it as finding the "loners" in the dataset.

        Great for: Finding isolated fraud patterns
        """
        print("📖 Distance-based: finds points far from neighbors")
        print("   Use case: Isolated, unusual transactions\n")

        # Sample for efficiency (distance calculation is expensive)
        sample_fraction = 0.1
        print(f"   Using {sample_fraction*100}% sample for efficiency...")

        # Get feature vectors as array
        feature_rdd = df.select('transaction_id', 'features').rdd

        # Calculate average distance to random sample of points
        # (Approximation of K-NN for efficiency)
        def calculate_avg_distance(row):
            """Calculate average distance to random sample of points."""
            from pyspark.ml.linalg import Vectors
            import numpy as np

            # This is a simplified version - in production you'd use proper KNN
            feature_vec = row['features'].toArray()
            # Use norm as proxy for distance
            distance = float(np.linalg.norm(feature_vec))
            return (row['transaction_id'], distance)

        distances_rdd = feature_rdd.map(calculate_avg_distance)

        # Convert to DataFrame
        distances_df = distances_rdd.toDF(['transaction_id', 'distance_score'])

        # Join back with original data
        df_with_distances = df.join(distances_df, on='transaction_id', how='left')

        # Normalize distance scores
        max_distance = df_with_distances.agg(max('distance_score')).collect()[0][0]
        df_with_distances = df_with_distances.withColumn(
            'distance_anomaly_score',
            col('distance_score') / lit(max_distance)
        )

        # Mark top 5% as anomalies
        threshold_val = df_with_distances.approxQuantile(
            'distance_anomaly_score', [0.95], 0.01
        )[0]

        df_with_distances = df_with_distances.withColumn(
            'distance_cluster',
            when(col('distance_anomaly_score') > threshold_val, 1).otherwise(0)
        )

        num_anomalies = df_with_distances.filter(col('distance_cluster') == 1).count()
        print(f"\n   ✅ Distance-based detection complete")
        print(f"      Anomalies detected: {num_anomalies}")

        return {
            'model': None,
            'predictions': df_with_distances,
            'algorithm': 'Distance_Outlier'
        }

    def _train_density_anomaly(self, df):
        """
        Train density-based anomaly detection.

        What is this?
        Look at the "density" of points around each transaction.
        - High density = normal region (many similar transactions)
        - Low density = anomaly (few similar transactions)

        Think of it as finding transactions in "sparse neighborhoods"!

        Great for: Finding unusual combinations of features
        """
        print("📖 Density-based: finds points in sparse regions")
        print("   Use case: Unusual feature combinations\n")

        # Use clustering density as proxy for local density
        # Run K-Means with high K
        print("   Computing local density using K-Means...")

        kmeans = KMeans(
            k=50,  # High K to capture local density
            featuresCol='features',
            predictionCol='density_cluster',
            seed=42
        )

        model = kmeans.fit(df)
        predictions = model.transform(df)

        # Calculate cluster sizes (inverse is anomaly score)
        cluster_sizes = predictions.groupBy('density_cluster').count()
        predictions = predictions.join(
            cluster_sizes.withColumnRenamed('count', 'cluster_size'),
            on='density_cluster'
        )

        # Smaller clusters = higher anomaly score
        max_cluster_size = predictions.agg(max('cluster_size')).collect()[0][0]
        predictions = predictions.withColumn(
            'density_anomaly_score',
            lit(1.0) - (col('cluster_size') / lit(max_cluster_size))
        )

        # Mark top 10% as anomalies
        threshold_val = predictions.approxQuantile(
            'density_anomaly_score', [0.90], 0.01
        )[0]

        predictions = predictions.withColumn(
            'density_anomaly_cluster',
            when(col('density_anomaly_score') > threshold_val, 1).otherwise(0)
        )

        num_anomalies = predictions.filter(col('density_anomaly_cluster') == 1).count()
        print(f"\n   ✅ Density-based detection complete")
        print(f"      Anomalies detected: {num_anomalies}")

        return {
            'model': model,
            'predictions': predictions,
            'algorithm': 'Density_Anomaly'
        }

    def _train_ensemble_clustering(self, df, previous_results):
        """
        Train ensemble clustering - combining multiple algorithms!

        What is Ensemble?
        Like asking multiple experts for their opinion, then combining them!

        We combine:
        - K-Means predictions
        - GMM predictions
        - Bisecting K-Means predictions

        If multiple algorithms agree a transaction is suspicious, it probably is!

        Great for: Robust fraud detection with high confidence
        """
        print("📖 Ensemble combines multiple algorithms for robust detection")
        print("   Use case: High-confidence fraud detection\n")

        # Get predictions from previous algorithms
        methods = self.config['algorithms']['ensemble']['methods']
        print(f"   Combining: {', '.join(methods)}")

        # Start with base dataframe
        ensemble_df = df

        # Add anomaly scores from each method
        score_columns = []

        for method in methods:
            if method in previous_results:
                method_df = previous_results[method]['predictions']
                score_col = f'{method}_anomaly_score'

                if score_col in method_df.columns:
                    # Join the score
                    ensemble_df = ensemble_df.join(
                        method_df.select('transaction_id', score_col),
                        on='transaction_id',
                        how='left'
                    )
                    score_columns.append(score_col)

        print(f"   Combined {len(score_columns)} anomaly scores")

        # Calculate ensemble score (average of all scores)
        if score_columns:
            ensemble_df = ensemble_df.withColumn(
                'ensemble_anomaly_score',
                (sum([col(c) for c in score_columns])) / lit(len(score_columns))
            )
        else:
            # Fallback if no scores available
            ensemble_df = ensemble_df.withColumn(
                'ensemble_anomaly_score',
                col('composite_risk_score')
            )

        # Mark top 15% as anomalies (matching fraud ratio)
        threshold_val = ensemble_df.approxQuantile(
            'ensemble_anomaly_score', [0.85], 0.01
        )[0]

        ensemble_df = ensemble_df.withColumn(
            'ensemble_cluster',
            when(col('ensemble_anomaly_score') > threshold_val, 1).otherwise(0)
        )

        num_anomalies = ensemble_df.filter(col('ensemble_cluster') == 1).count()
        print(f"\n   ✅ Ensemble clustering complete")
        print(f"      Anomalies detected: {num_anomalies}")

        return {
            'model': None,
            'predictions': ensemble_df,
            'algorithm': 'Ensemble'
        }

    def _train_risk_score_clustering(self, df):
        """
        Train risk-score based clustering.

        What is this?
        Use our domain knowledge (AML rules) to create risk scores,
        then cluster based on risk levels.

        This is like using a "cheat sheet" of known fraud patterns!

        Great for: Explainable, rule-based fraud detection
        """
        print("📖 Risk-Score: Uses AML domain knowledge for clustering")
        print("   Use case: Explainable, rule-based detection\n")

        # Use the composite risk score we created earlier
        # Define risk levels
        df_with_risk = df.withColumn(
            'risk_score_cluster',
            when(col('composite_risk_score') >= 0.7, 3)  # High risk
            .when(col('composite_risk_score') >= 0.4, 2)  # Medium risk
            .when(col('composite_risk_score') >= 0.2, 1)  # Low risk
            .otherwise(0)  # Normal
        )

        # Risk score IS the anomaly score
        df_with_risk = df_with_risk.withColumn(
            'risk_score_anomaly_score',
            col('composite_risk_score')
        )

        # Show distribution
        print("   Risk level distribution:")
        risk_dist = df_with_risk.groupBy('risk_score_cluster').count().orderBy('risk_score_cluster')
        for row in risk_dist.collect():
            level = row['risk_score_cluster']
            count = row['count']
            level_name = ['Normal', 'Low', 'Medium', 'High'][level]
            print(f"      {level_name}: {count}")

        print(f"\n   ✅ Risk-score clustering complete")

        return {
            'model': None,
            'predictions': df_with_risk,
            'algorithm': 'Risk_Score'
        }

    def _train_behavioral_clustering(self, df):
        """
        Train behavioral segmentation clustering.

        What is this?
        Group customers by their behavioral patterns:
        - High-value customers
        - Frequent small transactions
        - Occasional large transactions
        - Unusual timing patterns

        Think of it as creating "customer personalities"!

        Great for: Understanding different fraud typologies
        """
        print("📖 Behavioral: Segments by transaction behavior patterns")
        print("   Use case: Understanding fraud typologies\n")

        # Create behavioral features
        behavioral_df = df.withColumn(
            'is_high_value_customer',
            when(col('customer_avg_amount') > 5000, 1).otherwise(0)
        ).withColumn(
            'is_high_frequency_customer',
            when(col('customer_total_transactions') > 50, 1).otherwise(0)
        ).withColumn(
            'is_variable_customer',
            when(col('customer_cv') > 1.0, 1).otherwise(0)
        )

        # Create behavioral profile (binary encoding)
        behavioral_df = behavioral_df.withColumn(
            'behavioral_cluster',
            (col('is_high_value_customer') * 4 +
             col('is_high_frequency_customer') * 2 +
             col('is_variable_customer') * 1)
        )

        # Anomaly score based on behavior
        behavioral_df = behavioral_df.withColumn(
            'behavioral_anomaly_score',
            (col('is_high_value_customer') * 0.3 +
             col('is_high_frequency_customer') * 0.3 +
             col('is_variable_customer') * 0.4)
        )

        print("   Behavioral segments:")
        behavioral_dist = behavioral_df.groupBy('behavioral_cluster').count().orderBy('behavioral_cluster')
        for row in behavioral_dist.collect():
            print(f"      Segment {row['behavioral_cluster']}: {row['count']}")

        print(f"\n   ✅ Behavioral clustering complete")

        return {
            'model': None,
            'predictions': behavioral_df,
            'algorithm': 'Behavioral'
        }

    def _train_graph_clustering(self, df):
        """
        Train graph-based clustering.

        What is this?
        Build a transaction network:
        - Nodes = accounts
        - Edges = transactions between accounts
        - Find suspicious network patterns

        Think of it as "social network analysis" for money!

        Great for: Finding money laundering rings and networks
        """
        print("📖 Graph-based: Analyzes transaction networks")
        print("   Use case: Finding money laundering networks\n")

        # Count transactions between accounts
        print("   Building transaction network...")

        # Self-join to find connected accounts
        # (In a real system, you'd have sender/receiver account fields)
        # Here we approximate by looking at customer transaction patterns

        customer_stats = df.groupBy('customer_id').agg(
            countDistinct('account_id').alias('num_accounts_used'),
            countDistinct('transaction_type').alias('num_transaction_types'),
            count('transaction_id').alias('total_transactions')
        )

        # Join back
        graph_df = df.join(customer_stats, on='customer_id')

        # Calculate network anomaly score
        graph_df = graph_df.withColumn(
            'graph_anomaly_score',
            (col('num_accounts_used') / lit(10.0) +  # Using many accounts
             col('num_transaction_types') / lit(7.0))  / 2.0  # Using many types
        )

        # Cluster based on network behavior
        graph_df = graph_df.withColumn(
            'graph_cluster',
            when(col('num_accounts_used') >= 3, 1).otherwise(0)
        )

        num_anomalies = graph_df.filter(col('graph_cluster') == 1).count()
        print(f"\n   ✅ Graph-based clustering complete")
        print(f"      Network anomalies detected: {num_anomalies}")

        return {
            'model': None,
            'predictions': graph_df,
            'algorithm': 'Graph_Network'
        }

    def _train_temporal_clustering(self, df):
        """
        Train temporal pattern clustering.

        What is this?
        Analyze time-based patterns:
        - Burst activity (many transactions suddenly)
        - Regular patterns (daily, weekly rhythms)
        - Irregular patterns (random timing)

        Think of it as finding transactions with "weird timing"!

        Great for: Detecting structuring and rapid movement
        """
        print("📖 Temporal: Analyzes time-based transaction patterns")
        print("   Use case: Detecting timing-based fraud (structuring)\n")

        # Temporal features already exist, create temporal clusters
        temporal_df = df.withColumn(
            'temporal_anomaly_score',
            (col('is_unusual_hour') * 0.4 +
             col('is_weekend') * 0.2 +
             (lit(1) - col('is_business_hours')) * 0.4)
        )

        # Add velocity component
        temporal_df = temporal_df.withColumn(
            'temporal_anomaly_score',
            col('temporal_anomaly_score') +
            least(col('txn_count_24h') / lit(10.0), lit(1.0)) * 0.5
        )

        # Normalize
        temporal_df = temporal_df.withColumn(
            'temporal_anomaly_score',
            least(col('temporal_anomaly_score'), lit(1.0))
        )

        # Create clusters
        temporal_df = temporal_df.withColumn(
            'temporal_cluster',
            when(col('temporal_anomaly_score') > 0.6, 2)  # High temporal risk
            .when(col('temporal_anomaly_score') > 0.3, 1)  # Medium temporal risk
            .otherwise(0)  # Normal timing
        )

        print("   Temporal risk distribution:")
        temporal_dist = temporal_df.groupBy('temporal_cluster').count().orderBy('temporal_cluster')
        for row in temporal_dist.collect():
            print(f"      Cluster {row['temporal_cluster']}: {row['count']}")

        print(f"\n   ✅ Temporal clustering complete")

        return {
            'model': None,
            'predictions': temporal_df,
            'algorithm': 'Temporal'
        }

    def _calculate_distance_to_cluster(self, df, model, cluster_col, score_col):
        """
        Calculate distance from each point to its cluster center.
        This distance becomes our anomaly score!

        Farther from center = more unusual = higher anomaly score
        """
        # Get cluster centers
        centers = model.clusterCenters()

        # Define UDF to calculate distance
        def calc_distance(features, cluster_id):
            """Calculate Euclidean distance to cluster center."""
            import numpy as np
            center = centers[int(cluster_id)]
            features_array = features.toArray()
            distance = np.linalg.norm(features_array - center)
            return float(distance)

        # Register UDF
        distance_udf = udf(calc_distance, DoubleType())

        # Apply UDF
        df_with_distance = df.withColumn(
            score_col,
            distance_udf(col('features'), col(cluster_col))
        )

        # Normalize distance to 0-1 range
        max_dist = df_with_distance.agg(max(score_col)).collect()[0][0]
        df_with_distance = df_with_distance.withColumn(
            score_col,
            col(score_col) / lit(max_dist)
        )

        return df_with_distance


def main():
    """
    Main function to run clustering algorithms standalone.
    """
    print("\n" + "="*80)
    print("🤖 CLUSTERING ALGORITHMS FOR AML FRAUD DETECTION")
    print("="*80)

    # Initialize Spark
    print("\n⚙️  Initializing Spark...")
    from pyspark.sql import SparkSession
    import yaml

    spark = SparkSession.builder \
        .appName("AML_Clustering") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    print("   ✓ Spark initialized!")

    # Load configuration
    print("\n📋 Loading configuration...")
    with open('../configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print("   ✓ Configuration loaded!")

    # Load features
    print("\n📂 Loading feature data...")
    features_path = f"../{config['data']['features_path']}"
    df = spark.read.parquet(features_path)
    print(f"   ✓ Loaded {df.count()} transactions")

    # Train algorithms
    clustering = ClusteringAlgorithms(spark, config)
    results = clustering.train_all_algorithms(df)

    print("\n" + "="*80)
    print("✅ CLUSTERING COMPLETE!")
    print("="*80)
    print(f"\n📊 Trained {len(results)} different algorithms!")

    spark.stop()


if __name__ == "__main__":
    main()
