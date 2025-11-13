"""
==============================================================================
FEATURE ENGINEERING PIPELINE
==============================================================================
This module transforms raw banking data into features that machine learning
algorithms can understand.

Think of this as a "translator" - converting messy real-world data into
clean, numerical features that algorithms can process.

Key Concepts:
1. Tabular Features: Numbers and categories (amount, type, country, etc.)
2. Text Features: Words from descriptions (using TF-IDF)
3. Feature Scaling: Making all numbers comparable (normalization)
4. Feature Engineering: Creating new useful features from existing ones

Author: Your Friendly AI Teacher
Date: 2025-11-13
==============================================================================
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.ml.feature import (
    VectorAssembler, StandardScaler, MinMaxScaler,
    StringIndexer, OneHotEncoder, HashingTF, IDF,
    Tokenizer, RegexTokenizer, StopWordsRemover,
    CountVectorizer, PCA
)
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors, VectorUDT
import yaml


class FeatureEngineer:
    """
    This class handles all feature engineering tasks.

    What it does:
    1. Creates time-based features (hour, day, month)
    2. Creates aggregated features (customer's average transaction amount)
    3. Processes text descriptions using NLP techniques
    4. Encodes categorical variables
    5. Scales numerical features
    6. Combines everything into a single feature vector

    Think of it as a chef preparing ingredients - cleaning, chopping, and
    organizing everything before cooking!
    """

    def __init__(self, spark, config):
        """
        Initialize the feature engineer.

        Args:
            spark: PySpark session
            config: Configuration dictionary
        """
        self.spark = spark
        self.config = config
        print("🔧 Feature Engineer initialized!")

    def engineer_features(self, raw_df):
        """
        Main method to engineer all features.

        This is the "assembly line" that transforms raw data into ML-ready features.

        Args:
            raw_df: Raw transaction DataFrame

        Returns:
            DataFrame with engineered features
        """
        print("\n" + "="*80)
        print("🏭 FEATURE ENGINEERING PIPELINE")
        print("="*80)

        # Step 1: Time-based features
        print("\n⏰ Step 1: Creating time-based features...")
        df = self._create_time_features(raw_df)

        # Step 2: Customer aggregated features
        print("\n👥 Step 2: Creating customer aggregated features...")
        df = self._create_customer_features(df)

        # Step 3: Transaction velocity features
        print("\n🚀 Step 3: Creating transaction velocity features...")
        df = self._create_velocity_features(df)

        # Step 4: Text features from descriptions
        print("\n📝 Step 4: Processing text descriptions...")
        df = self._create_text_features(df)

        # Step 5: Statistical features
        print("\n📊 Step 5: Creating statistical features...")
        df = self._create_statistical_features(df)

        # Step 6: AML-specific risk indicators
        print("\n🚨 Step 6: Creating AML risk indicators...")
        df = self._create_aml_risk_features(df)

        print("\n✅ Feature engineering complete!")
        print(f"   Total features created: {len(df.columns)}")

        return df

    def _create_time_features(self, df):
        """
        Create time-based features.

        Why? Time patterns matter in fraud detection!
        - Fraudsters often operate at unusual hours
        - Day of week affects normal transaction patterns
        - Seasonality can be important

        Features created:
        - Hour of day (0-23)
        - Day of week (1-7)
        - Is weekend? (0 or 1)
        - Is business hours? (0 or 1)
        - Month of year
        - Quarter of year
        """
        df = df \
            .withColumn('hour_sin', sin(col('transaction_hour') * 2 * 3.14159 / 24)) \
            .withColumn('hour_cos', cos(col('transaction_hour') * 2 * 3.14159 / 24)) \
            .withColumn('day_sin', sin(col('transaction_day_of_week') * 2 * 3.14159 / 7)) \
            .withColumn('day_cos', cos(col('transaction_day_of_week') * 2 * 3.14159 / 7)) \
            .withColumn('is_weekend',
                       when(col('transaction_day_of_week').isin([1, 7]), 1).otherwise(0)) \
            .withColumn('is_business_hours',
                       when((col('transaction_hour') >= 9) & (col('transaction_hour') <= 17), 1).otherwise(0)) \
            .withColumn('is_unusual_hour',
                       when((col('transaction_hour') >= 0) & (col('transaction_hour') <= 5), 1).otherwise(0)) \
            .withColumn('transaction_quarter', quarter(col('timestamp_dt')))

        print(f"   ✓ Created time-based features")
        return df

    def _create_customer_features(self, df):
        """
        Create customer-level aggregated features.

        Why? A customer's history tells us what's "normal" for them!
        - Average transaction amount
        - Total number of transactions
        - Standard deviation (variability) in amounts
        - Time since first transaction

        This is like building a "profile" for each customer.
        """
        # Define window for customer-level aggregations
        customer_window = Window.partitionBy('customer_id')

        df = df \
            .withColumn('customer_avg_amount', avg('amount').over(customer_window)) \
            .withColumn('customer_total_transactions', count('transaction_id').over(customer_window)) \
            .withColumn('customer_stddev_amount', stddev('amount').over(customer_window)) \
            .withColumn('customer_max_amount', max('amount').over(customer_window)) \
            .withColumn('customer_min_amount', min('amount').over(customer_window))

        # Fill nulls (for customers with only 1 transaction, stddev is null)
        df = df.fillna({'customer_stddev_amount': 0})

        # Deviation from customer's normal behavior
        df = df.withColumn('amount_deviation_from_avg',
                          (col('amount') - col('customer_avg_amount')) / (col('customer_stddev_amount') + 1))

        print(f"   ✓ Created customer aggregated features")
        return df

    def _create_velocity_features(self, df):
        """
        Create transaction velocity features.

        Why? Rapid transaction patterns are a red flag in AML!
        - Number of transactions in last 24 hours
        - Number of transactions in last 7 days
        - Time since last transaction

        Think of this as measuring the "speed" of money movement.
        """
        # Sort by customer and timestamp
        df = df.withColumn('timestamp_unix', unix_timestamp(col('timestamp_dt')))

        # Define time windows for velocity calculations
        # Last 24 hours
        window_24h = Window.partitionBy('customer_id') \
            .orderBy('timestamp_unix') \
            .rangeBetween(-86400, 0)  # 86400 seconds = 24 hours

        # Last 7 days
        window_7d = Window.partitionBy('customer_id') \
            .orderBy('timestamp_unix') \
            .rangeBetween(-604800, 0)  # 604800 seconds = 7 days

        df = df \
            .withColumn('txn_count_24h', count('transaction_id').over(window_24h)) \
            .withColumn('txn_amount_sum_24h', sum('amount').over(window_24h)) \
            .withColumn('txn_count_7d', count('transaction_id').over(window_7d)) \
            .withColumn('txn_amount_sum_7d', sum('amount').over(window_7d))

        # Time since last transaction for this customer
        customer_time_window = Window.partitionBy('customer_id').orderBy('timestamp_unix')
        df = df.withColumn('time_since_last_txn',
                          col('timestamp_unix') - lag('timestamp_unix', 1).over(customer_time_window))
        df = df.fillna({'time_since_last_txn': 0})

        print(f"   ✓ Created velocity features")
        return df

    def _create_text_features(self, df):
        """
        Create features from text descriptions using NLP.

        Why? Words matter! "Large cash deposit" vs "grocery store purchase"
        contain very different signals.

        Process:
        1. Tokenization: Split text into words
        2. Remove stop words: Remove "the", "and", "is", etc.
        3. TF-IDF: Calculate importance of each word
           - TF (Term Frequency): How often word appears
           - IDF (Inverse Document Frequency): How unique the word is

        Think of this as teaching the computer to "read" transaction descriptions!
        """
        # Tokenize: Split descriptions into words
        tokenizer = RegexTokenizer(
            inputCol="description",
            outputCol="words",
            pattern="\\W"  # Split on non-word characters
        )
        df_words = tokenizer.transform(df)

        # Remove stop words (common words like "the", "and", "is")
        remover = StopWordsRemover(
            inputCol="words",
            outputCol="filtered_words"
        )
        df_filtered = remover.transform(df_words)

        # Create TF-IDF features
        # TF: How many times word appears in this description
        cv = CountVectorizer(
            inputCol="filtered_words",
            outputCol="raw_features",
            vocabSize=self.config['features']['text']['max_features'],
            minDF=self.config['features']['text']['min_df']
        )
        cv_model = cv.fit(df_filtered)
        df_tf = cv_model.transform(df_filtered)

        # IDF: How unique/important is this word across all descriptions
        idf = IDF(
            inputCol="raw_features",
            outputCol="text_features"
        )
        idf_model = idf.fit(df_tf)
        df_tfidf = idf_model.transform(df_tf)

        print(f"   ✓ Created text features (TF-IDF)")
        print(f"   ✓ Vocabulary size: {len(cv_model.vocabulary)}")

        # Store vocabulary for later use
        self.text_vocabulary = cv_model.vocabulary

        return df_tfidf

    def _create_statistical_features(self, df):
        """
        Create statistical features that capture unusual patterns.

        Why? Statistics help us spot outliers!
        - Z-score: How many standard deviations from mean?
        - Percentile rank: Is this in top 1% of amounts?
        - Coefficient of variation: How variable is this customer?

        Think of this as using math to find the "odd ones out"!
        """
        # Calculate global statistics for amount
        amount_stats = df.select(
            mean('amount').alias('global_mean_amount'),
            stddev('amount').alias('global_stddev_amount')
        ).collect()[0]

        global_mean = amount_stats['global_mean_amount']
        global_stddev = amount_stats['global_stddev_amount']

        # Z-score: How unusual is this amount?
        df = df.withColumn('amount_zscore',
                          (col('amount') - lit(global_mean)) / lit(global_stddev))

        # Is this amount an outlier?
        df = df.withColumn('is_amount_outlier',
                          when(abs(col('amount_zscore')) > 3, 1).otherwise(0))

        # Ratio to customer's average
        df = df.withColumn('amount_ratio_to_customer_avg',
                          col('amount') / (col('customer_avg_amount') + 1))

        # Customer's coefficient of variation (variability)
        df = df.withColumn('customer_cv',
                          col('customer_stddev_amount') / (col('customer_avg_amount') + 1))

        print(f"   ✓ Created statistical features")
        return df

    def _create_aml_risk_features(self, df):
        """
        Create AML-specific risk indicator features.

        These are based on real-world AML red flags that compliance officers
        look for:

        1. Structuring indicators
        2. Rapid movement indicators
        3. Round amount indicators
        4. High-risk channel indicators
        5. Cross-border indicators

        Think of these as the "cheat sheet" that fraud investigators use!
        """
        # High-value transaction flag
        high_value_threshold = self.config['aml_rules']['suspicious_patterns']['high_value_threshold']
        df = df.withColumn('is_high_value',
                          when(col('amount') >= high_value_threshold, 1).otherwise(0))

        # Round amount flag (exactly divisible by 1000)
        df = df.withColumn('is_suspiciously_round',
                          when((col('amount') % 1000 == 0) & (col('amount') >= 10000), 1).otherwise(0))

        # Just-below-threshold flag (structuring indicator)
        df = df.withColumn('is_just_below_threshold',
                          when((col('amount') >= 9000) & (col('amount') < 10000), 1).otherwise(0))

        # High-velocity flag
        rapid_txn_threshold = self.config['aml_rules']['suspicious_patterns']['rapid_transactions_count']
        df = df.withColumn('is_high_velocity',
                          when(col('txn_count_24h') >= rapid_txn_threshold, 1).otherwise(0))

        # Unusual timing flag
        df = df.withColumn('is_odd_hours',
                          when(col('is_unusual_hour') == 1, 1).otherwise(0))

        # High-risk transaction type
        df = df.withColumn('is_high_risk_type',
                          when(col('transaction_type').isin(['WIRE_TRANSFER', 'INTERNATIONAL_WIRE', 'CASH_DEPOSIT']), 1)
                          .otherwise(0))

        # Composite risk score (simple weighted sum)
        df = df.withColumn('composite_risk_score',
                          (col('is_high_value') * 2 +
                           col('is_suspiciously_round') * 3 +
                           col('is_just_below_threshold') * 4 +
                           col('is_high_velocity') * 3 +
                           col('is_odd_hours') * 2 +
                           col('is_high_risk_type') * 2) / 16.0)  # Normalize to 0-1

        print(f"   ✓ Created AML risk indicator features")
        return df

    def prepare_for_clustering(self, df):
        """
        Prepare final feature vectors for clustering algorithms.

        This is the "final packaging" step - combining all features into
        a single vector that clustering algorithms can use.

        Steps:
        1. Select numerical features
        2. Encode categorical features
        3. Combine with text features
        4. Scale everything to same range
        5. Create final feature vector
        """
        print("\n" + "="*80)
        print("🎯 PREPARING FEATURES FOR CLUSTERING")
        print("="*80)

        # Step 1: Select numerical features
        print("\n1️⃣  Selecting numerical features...")
        numerical_features = [
            'amount', 'amount_log', 'transaction_hour', 'transaction_day_of_week',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'is_weekend', 'is_business_hours', 'is_unusual_hour',
            'customer_avg_amount', 'customer_total_transactions',
            'customer_stddev_amount', 'customer_max_amount', 'customer_min_amount',
            'amount_deviation_from_avg',
            'txn_count_24h', 'txn_amount_sum_24h', 'txn_count_7d', 'txn_amount_sum_7d',
            'time_since_last_txn',
            'amount_zscore', 'is_amount_outlier', 'amount_ratio_to_customer_avg',
            'customer_cv',
            'is_high_value', 'is_suspiciously_round', 'is_just_below_threshold',
            'is_high_velocity', 'is_odd_hours', 'is_high_risk_type',
            'composite_risk_score', 'account_balance'
        ]
        print(f"   ✓ Selected {len(numerical_features)} numerical features")

        # Step 2: Encode categorical features
        print("\n2️⃣  Encoding categorical features...")
        categorical_features = ['transaction_type', 'channel', 'merchant_category']

        # Use StringIndexer to convert strings to numbers
        indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid="keep")
                   for col in categorical_features]

        # Use OneHotEncoder to create binary vectors
        encoders = [OneHotEncoder(inputCol=f"{col}_index", outputCol=f"{col}_encoded")
                   for col in categorical_features]

        encoded_categorical_features = [f"{col}_encoded" for col in categorical_features]
        print(f"   ✓ Encoded {len(categorical_features)} categorical features")

        # Step 3: Combine all features
        print("\n3️⃣  Combining all features...")
        all_features = numerical_features + encoded_categorical_features + ['text_features']

        # Step 4: Assemble features into a single vector
        print("\n4️⃣  Assembling feature vector...")
        assembler = VectorAssembler(
            inputCols=all_features,
            outputCol="raw_features_vector",
            handleInvalid="skip"
        )

        # Step 5: Scale features (StandardScaler: mean=0, stddev=1)
        print("\n5️⃣  Scaling features...")
        scaler = StandardScaler(
            inputCol="raw_features_vector",
            outputCol="features",
            withStd=True,
            withMean=True
        )

        # Create and fit pipeline
        print("\n6️⃣  Building and fitting pipeline...")
        pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler])

        # Fit and transform
        pipeline_model = pipeline.fit(df)
        final_df = pipeline_model.transform(df)

        print("\n✅ Feature preparation complete!")
        print(f"   Total features in vector: {final_df.select('features').first()[0].size}")

        return final_df, pipeline_model

    def save_features(self, df, path):
        """
        Save engineered features to disk.
        """
        print(f"\n💾 Saving features to: {path}")
        # Select important columns to save
        columns_to_save = [
            'transaction_id', 'customer_id', 'account_id',
            'amount', 'timestamp', 'is_suspicious',
            'features', 'composite_risk_score'
        ]
        df.select(columns_to_save).write.mode('overwrite').parquet(path)
        print("   ✓ Features saved successfully!")


def main():
    """
    Main function to run feature engineering standalone.
    """
    print("\n" + "="*80)
    print("🔧 FEATURE ENGINEERING PIPELINE")
    print("="*80)

    # Initialize Spark
    print("\n⚙️  Initializing Spark...")
    spark = SparkSession.builder \
        .appName("AML_Feature_Engineering") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    print("   ✓ Spark initialized successfully!")

    # Load configuration
    print("\n📋 Loading configuration...")
    with open('../configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print("   ✓ Configuration loaded!")

    # Load raw data
    print("\n📂 Loading raw data...")
    raw_data_path = f"../{config['data']['raw_data_path']}"
    raw_df = spark.read.parquet(raw_data_path)
    print(f"   ✓ Loaded {raw_df.count()} transactions")

    # Engineer features
    engineer = FeatureEngineer(spark, config)
    feature_df = engineer.engineer_features(raw_df)

    # Prepare for clustering
    final_df, pipeline_model = engineer.prepare_for_clustering(feature_df)

    # Show sample
    print("\n📊 Sample of engineered features:")
    final_df.select('transaction_id', 'amount', 'composite_risk_score', 'is_suspicious').show(10)

    # Save
    output_path = f"../{config['data']['features_path']}"
    engineer.save_features(final_df, output_path)

    print("\n" + "="*80)
    print("✅ FEATURE ENGINEERING COMPLETE!")
    print("="*80)

    spark.stop()


if __name__ == "__main__":
    main()
