"""
=============================================================================
FEATURE ENGINEERING PIPELINE
=============================================================================
This module transforms raw banking data into ML-ready features.

Think of this as a "food processor" for data:
- Raw data goes in (messy, mixed formats)
- Clean, numerical features come out (ready for ML models)

We handle:
1. Text vectorization (converting words to numbers)
2. Numerical scaling (making numbers comparable)
3. Categorical encoding (converting categories to numbers)
4. Feature creation (making new useful features from existing ones)
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, when, hour, dayofweek, month, year, avg, count, sum as spark_sum,
    lag, lead, concat_ws, lower, trim, regexp_replace, length, split, size
)
from pyspark.sql.window import Window
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    VectorAssembler, StandardScaler, MinMaxScaler, RobustScaler,
    StringIndexer, OneHotEncoder, Tokenizer, HashingTF, IDF,
    Word2Vec, CountVectorizer, ChiSqSelector, PCA
)
from pyspark.ml.feature import Imputer
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    This class handles all feature engineering for our AML fraud detection.

    It's like a Swiss Army knife for data transformation!
    """

    def __init__(self, spark: SparkSession, config: Dict):
        """
        Initialize the feature engineer.

        Parameters:
        -----------
        spark : SparkSession
            Active Spark session
        config : Dict
            Configuration dictionary with feature engineering settings
        """
        self.spark = spark
        self.config = config
        self.pipeline = None
        self.feature_columns = []

        logger.info("🔧 Feature Engineer initialized")

    def build_pipeline(self, text_columns: List[str], numeric_columns: List[str],
                      categorical_columns: List[str]) -> Pipeline:
        """
        Build the complete feature engineering pipeline.

        This creates a sequence of transformations that happen automatically!
        Like an assembly line in a factory.

        Parameters:
        -----------
        text_columns : List[str]
            Columns with text data (e.g., transaction_description)
        numeric_columns : List[str]
            Columns with numbers (e.g., transaction_amount)
        categorical_columns : List[str]
            Columns with categories (e.g., account_type)

        Returns:
        --------
        Pipeline : PySpark ML Pipeline
        """
        logger.info("🏗️  Building feature engineering pipeline...")

        stages = []

        # =====================================================================
        # STAGE 1: Handle Missing Values
        # =====================================================================
        # Real-world data often has missing values - we need to fill them!
        logger.info("   Stage 1: Handling missing values...")

        imputer = Imputer(
            inputCols=numeric_columns,
            outputCols=[f"{col}_imputed" for col in numeric_columns],
            strategy="mean"  # Fill missing with average value
        )
        stages.append(imputer)

        # Update column names after imputation
        numeric_columns_imputed = [f"{col}_imputed" for col in numeric_columns]

        # =====================================================================
        # STAGE 2: Process Text Features
        # =====================================================================
        # Convert text to numbers using TF-IDF (Term Frequency-Inverse Document Frequency)
        # This captures which words are important and unique
        logger.info("   Stage 2: Processing text features...")

        text_feature_columns = []
        for text_col in text_columns:
            # Step 1: Tokenize (split text into words)
            tokenizer = Tokenizer(
                inputCol=text_col,
                outputCol=f"{text_col}_tokens"
            )
            stages.append(tokenizer)

            # Step 2: HashingTF (convert words to numerical features)
            hashingTF = HashingTF(
                inputCol=f"{text_col}_tokens",
                outputCol=f"{text_col}_tf",
                numFeatures=100  # Create 100 features per text column
            )
            stages.append(hashingTF)

            # Step 3: IDF (weight features by importance)
            idf = IDF(
                inputCol=f"{text_col}_tf",
                outputCol=f"{text_col}_features"
            )
            stages.append(idf)

            text_feature_columns.append(f"{text_col}_features")

        logger.info(f"      ✓ Created text features from {len(text_columns)} columns")

        # =====================================================================
        # STAGE 3: Encode Categorical Features
        # =====================================================================
        # Convert categories (like 'checking', 'savings') to numbers
        logger.info("   Stage 3: Encoding categorical features...")

        categorical_encoded_columns = []
        for cat_col in categorical_columns:
            # Step 1: String Indexer (assign number to each category)
            indexer = StringIndexer(
                inputCol=cat_col,
                outputCol=f"{cat_col}_index",
                handleInvalid="keep"  # Handle new categories gracefully
            )
            stages.append(indexer)

            # Step 2: One-Hot Encoding (create binary columns for each category)
            # Example: account_type='checking' becomes [1, 0, 0]
            #          account_type='savings' becomes [0, 1, 0]
            encoder = OneHotEncoder(
                inputCols=[f"{cat_col}_index"],
                outputCols=[f"{cat_col}_encoded"],
                dropLast=True  # Avoid redundancy (n-1 encoding)
            )
            stages.append(encoder)

            categorical_encoded_columns.append(f"{cat_col}_encoded")

        logger.info(f"      ✓ Encoded {len(categorical_columns)} categorical columns")

        # =====================================================================
        # STAGE 4: Scale Numerical Features
        # =====================================================================
        # Make all numbers comparable (e.g., age vs. transaction amount)
        logger.info("   Stage 4: Scaling numerical features...")

        # First, assemble numeric features into a vector
        assembler_numeric = VectorAssembler(
            inputCols=numeric_columns_imputed,
            outputCol="numeric_features_raw"
        )
        stages.append(assembler_numeric)

        # Then scale them (StandardScaler: mean=0, std=1)
        scaler = StandardScaler(
            inputCol="numeric_features_raw",
            outputCol="numeric_features_scaled",
            withMean=True,
            withStd=True
        )
        stages.append(scaler)

        logger.info(f"      ✓ Scaled {len(numeric_columns_imputed)} numerical columns")

        # =====================================================================
        # STAGE 5: Combine All Features
        # =====================================================================
        # Merge text, categorical, and numerical features into one vector
        logger.info("   Stage 5: Combining all features...")

        all_feature_columns = (
            text_feature_columns +
            categorical_encoded_columns +
            ["numeric_features_scaled"]
        )

        final_assembler = VectorAssembler(
            inputCols=all_feature_columns,
            outputCol="features"
        )
        stages.append(final_assembler)

        logger.info(f"      ✓ Combined {len(all_feature_columns)} feature groups")

        # =====================================================================
        # STAGE 6: Feature Selection (Optional)
        # =====================================================================
        # Select only the most important features using Chi-Squared test
        if self.config.get('feature_selection', {}).get('enabled', False):
            logger.info("   Stage 6: Selecting top features...")

            k_best = self.config['feature_selection'].get('k_best', 50)
            selector = ChiSqSelector(
                numTopFeatures=k_best,
                featuresCol="features",
                outputCol="selected_features",
                labelCol="is_fraud"
            )
            stages.append(selector)

            final_features_col = "selected_features"
            logger.info(f"      ✓ Selecting top {k_best} features")
        else:
            final_features_col = "features"

        # Build the complete pipeline
        pipeline = Pipeline(stages=stages)

        self.pipeline = pipeline
        self.feature_columns = all_feature_columns

        logger.info(f"✅ Feature engineering pipeline built with {len(stages)} stages")

        return pipeline

    def create_advanced_features(self, df: DataFrame) -> DataFrame:
        """
        Create advanced engineered features before the pipeline.

        These are domain-specific features that experts know are useful for fraud detection!

        Parameters:
        -----------
        df : DataFrame
            Input DataFrame

        Returns:
        --------
        DataFrame : DataFrame with additional engineered features
        """
        logger.info("🎨 Creating advanced engineered features...")

        # Create a copy to avoid modifying original
        df_enhanced = df

        # =================================================================
        # TIME-BASED FEATURES
        # =================================================================
        # Fraud patterns often involve unusual timing

        # Is transaction on weekend? (Fraud more common on weekends)
        df_enhanced = df_enhanced.withColumn(
            "is_weekend",
            when(col("transaction_day_of_week").isin([5, 6]), 1).otherwise(0)
        )

        # Is transaction at night? (2 AM - 6 AM = suspicious)
        df_enhanced = df_enhanced.withColumn(
            "is_night",
            when((col("transaction_hour") >= 2) & (col("transaction_hour") <= 6), 1).otherwise(0)
        )

        # =================================================================
        # AMOUNT-BASED FEATURES
        # =================================================================
        # Unusual amounts are red flags

        # Ratio of current transaction to average
        # If someone usually spends $50 but suddenly spends $5000 = suspicious!
        df_enhanced = df_enhanced.withColumn(
            "amount_to_avg_ratio",
            col("transaction_amount") / (col("avg_transaction_amount") + 1)  # +1 to avoid division by zero
        )

        # Is amount a "round number"? (e.g., $1000, $5000)
        # Money launderers often use round numbers
        df_enhanced = df_enhanced.withColumn(
            "is_round_amount",
            when(col("transaction_amount") % 1000 == 0, 1).otherwise(0)
        )

        # =================================================================
        # VELOCITY FEATURES
        # =================================================================
        # How fast is the account transacting?

        # Transaction velocity (transactions per day the account is open)
        df_enhanced = df_enhanced.withColumn(
            "transaction_velocity",
            col("num_transactions_24h") / (col("account_age_days") + 1)
        )

        # =================================================================
        # RISK SCORE FEATURES
        # =================================================================
        # Combine multiple signals into risk scores

        # Account risk score (new account + many transactions = risky)
        df_enhanced = df_enhanced.withColumn(
            "account_risk_score",
            when(col("account_age_days") < 30, 3)  # Very new = high risk
            .when(col("account_age_days") < 180, 2)  # Somewhat new = medium risk
            .otherwise(1)  # Old account = low risk
        )

        # Credit risk score (low credit score = higher risk)
        df_enhanced = df_enhanced.withColumn(
            "credit_risk_score",
            when(col("credit_score") < 600, 3)  # Bad credit = high risk
            .when(col("credit_score") < 700, 2)  # Fair credit = medium risk
            .otherwise(1)  # Good credit = low risk
        )

        # =================================================================
        # TEXT LENGTH FEATURES
        # =================================================================
        # Suspicious transactions often have short or no descriptions

        df_enhanced = df_enhanced.withColumn(
            "description_length",
            length(col("transaction_description"))
        )

        df_enhanced = df_enhanced.withColumn(
            "has_description",
            when(col("description_length") > 0, 1).otherwise(0)
        )

        # =================================================================
        # CROSS-BORDER FEATURES
        # =================================================================
        # International transactions have higher fraud risk

        df_enhanced = df_enhanced.withColumn(
            "is_international",
            when(col("source_country") != col("destination_country"), 1).otherwise(0)
        )

        logger.info(f"   ✓ Created 12 advanced features")
        logger.info(f"   ✓ Total columns now: {len(df_enhanced.columns)}")

        return df_enhanced

    def fit_transform(self, df: DataFrame) -> Tuple[DataFrame, Pipeline]:
        """
        Fit the pipeline on training data and transform it.

        This is like "learning" from the data and then applying the transformation.

        Parameters:
        -----------
        df : DataFrame
            Training data

        Returns:
        --------
        Tuple[DataFrame, Pipeline] : Transformed data and fitted pipeline
        """
        logger.info("🔄 Fitting and transforming training data...")

        # Create advanced features first
        df_enhanced = self.create_advanced_features(df)

        # Define column types
        text_columns = ['transaction_description', 'merchant_category']
        categorical_columns = ['source_country', 'destination_country', 'account_type']

        # Numeric columns (including engineered features)
        numeric_columns = [
            'transaction_amount', 'account_age_days', 'num_transactions_24h',
            'avg_transaction_amount', 'transaction_hour', 'transaction_day_of_week',
            'customer_age', 'credit_score', 'num_failed_logins',
            'is_weekend', 'is_night', 'amount_to_avg_ratio', 'is_round_amount',
            'transaction_velocity', 'account_risk_score', 'credit_risk_score',
            'description_length', 'has_description', 'is_international'
        ]

        # Build pipeline
        pipeline = self.build_pipeline(text_columns, numeric_columns, categorical_columns)

        # Fit the pipeline (learn from data)
        logger.info("📚 Learning from training data (fitting pipeline)...")
        pipeline_model = pipeline.fit(df_enhanced)

        # Transform the data (apply what we learned)
        logger.info("🔄 Applying transformations...")
        df_transformed = pipeline_model.transform(df_enhanced)

        logger.info("✅ Feature engineering complete!")

        return df_transformed, pipeline_model

    def transform(self, df: DataFrame, fitted_pipeline: Pipeline) -> DataFrame:
        """
        Transform new data using already-fitted pipeline.

        Use this for validation and test data!

        Parameters:
        -----------
        df : DataFrame
            New data to transform
        fitted_pipeline : Pipeline
            Already fitted pipeline model

        Returns:
        --------
        DataFrame : Transformed data
        """
        logger.info("🔄 Transforming new data with fitted pipeline...")

        # Create advanced features first (same as training)
        df_enhanced = self.create_advanced_features(df)

        # Apply fitted pipeline
        df_transformed = fitted_pipeline.transform(df_enhanced)

        logger.info("✅ Transformation complete!")

        return df_transformed


def main():
    """
    Test the feature engineering pipeline.
    """
    print("\n" + "="*80)
    print("🔧 TESTING FEATURE ENGINEERING PIPELINE")
    print("="*80 + "\n")

    # Initialize Spark
    spark = SparkSession.builder \
        .appName("FeatureEngineeringTest") \
        .master("local[*]") \
        .getOrCreate()

    # Load sample data
    logger.info("📂 Loading sample data...")
    df = spark.read.csv('./data/raw/banking_transactions.csv', header=True, inferSchema=True)

    logger.info(f"   ✓ Loaded {df.count()} transactions")

    # Initialize feature engineer
    config = {
        'feature_selection': {
            'enabled': True,
            'k_best': 50
        }
    }

    engineer = FeatureEngineer(spark, config)

    # Fit and transform
    df_transformed, pipeline_model = engineer.fit_transform(df)

    # Show results
    logger.info("\n📊 Sample of transformed data:")
    df_transformed.select("transaction_id", "is_fraud", "features").show(5, truncate=False)

    # Save transformed data
    output_path = './data/processed/features_train.parquet'
    logger.info(f"\n💾 Saving transformed data to {output_path}")
    df_transformed.write.mode('overwrite').parquet(output_path)

    logger.info("\n✅ Feature engineering test complete!")

    spark.stop()


if __name__ == "__main__":
    main()
