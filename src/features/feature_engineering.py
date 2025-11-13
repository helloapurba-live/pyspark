"""
============================================================================
FEATURE ENGINEERING PIPELINE
============================================================================

Welcome to Feature Engineering - where raw data becomes ML-ready features!

WHAT IS FEATURE ENGINEERING?
-----------------------------
Think of it like cooking: you don't just throw raw ingredients into a dish.
You chop, season, and combine them strategically. Similarly, we transform
raw transaction data into features that help our models spot fraud patterns.

WHY IS THIS IMPORTANT?
----------------------
The saying goes: "Garbage in, garbage out."
Good features = Good model performance
Bad features = Even the best algorithm will struggle

WHAT WE'LL DO:
--------------
1. **Tabular Features**: Process numerical and categorical data
   - Scale amounts (normalize)
   - Encode categories (convert text to numbers)
   - Create time-based features

2. **Text Features**: Extract information from descriptions
   - TF-IDF (finds important words)
   - Count Vectorizer (word frequencies)
   - N-grams (word combinations)

3. **Feature Combination**: Merge everything together
   - Numerical + Text = Complete feature set
   - Ready for ML algorithms!

Let's build this step by step! 🚀
============================================================================
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    VectorAssembler,
    StandardScaler,
    StringIndexer,
    OneHotEncoder,
    Tokenizer,
    HashingTF,
    IDF,
    StopWordsRemover,
    CountVectorizer,
    SQLTransformer
)
from typing import List, Tuple
import logging

# Set up logging so we can see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    This class handles all feature engineering for our fraud detection system.

    Think of it as a feature factory: raw data goes in, ML-ready features come out!
    """

    def __init__(self, spark: SparkSession):
        """
        Initialize the feature engineer.

        Parameters:
        -----------
        spark : SparkSession
            Our PySpark session (the engine that processes big data)
        """
        self.spark = spark
        logger.info("🔧 Feature Engineer initialized")

    def create_time_features(self, df: DataFrame) -> DataFrame:
        """
        Extract useful information from timestamps.

        WHY TIME MATTERS:
        ----------------
        - Fraudsters often operate at unusual times (late night)
        - Patterns differ by day (weekends vs weekdays)
        - Time series can reveal suspicious bursts of activity

        Parameters:
        -----------
        df : DataFrame
            Input data with 'timestamp' column

        Returns:
        --------
        DataFrame with additional time-based features
        """
        logger.info("⏰ Creating time-based features...")

        # Extract various time components
        df = df.withColumn('hour', F.hour('timestamp'))
        df = df.withColumn('day_of_week', F.dayofweek('timestamp'))
        df = df.withColumn('day_of_month', F.dayofmonth('timestamp'))
        df = df.withColumn('month', F.month('timestamp'))

        # Create categorical time periods
        # -------------------------------
        # Is it night? (10 PM - 6 AM) - Unusual banking hours!
        df = df.withColumn('is_night',
                           F.when((F.col('hour') >= 22) | (F.col('hour') <= 6), 1).otherwise(0))

        # Is it weekend? - Different patterns on weekends
        df = df.withColumn('is_weekend',
                           F.when(F.col('day_of_week').isin([1, 7]), 1).otherwise(0))

        # Business hours? (9 AM - 5 PM on weekdays)
        df = df.withColumn('is_business_hours',
                           F.when((F.col('hour').between(9, 17)) &
                                  (~F.col('day_of_week').isin([1, 7])), 1).otherwise(0))

        logger.info("  ✅ Added: hour, day_of_week, month, is_night, is_weekend, is_business_hours")
        return df

    def create_amount_features(self, df: DataFrame) -> DataFrame:
        """
        Engineer features from transaction amounts.

        AMOUNT PATTERNS THAT MATTER:
        ----------------------------
        - Just below $10,000? (Structuring red flag!)
        - Unusually round number? ($50,000 vs $49,873.42)
        - Very large or very small?
        - Logarithmic scale (helps with skewed distributions)

        Parameters:
        -----------
        df : DataFrame
            Input data with 'amount' column

        Returns:
        --------
        DataFrame with amount-based features
        """
        logger.info("💰 Creating amount-based features...")

        # Flag transactions near reporting threshold
        # ------------------------------------------
        # $10,000 is a key threshold in US banking regulations
        df = df.withColumn('is_near_threshold',
                           F.when((F.col('amount') >= 7000) & (F.col('amount') < 10000), 1).otherwise(0))

        # Flag large transactions
        df = df.withColumn('is_large_amount',
                           F.when(F.col('amount') > 10000, 1).otherwise(0))

        # Flag suspiciously round amounts
        # -------------------------------
        # Real transactions are usually not perfect round numbers
        # $50,000 is more suspicious than $49,847.63
        df = df.withColumn('amount_mod_1000', F.col('amount') % 1000)
        df = df.withColumn('is_round_amount',
                           F.when(F.col('amount_mod_1000') < 10, 1).otherwise(0))

        # Log transformation
        # ------------------
        # Many ML algorithms work better with log-scaled amounts
        # Helps when data ranges from $10 to $1,000,000
        df = df.withColumn('amount_log', F.log1p(F.col('amount')))

        # Amount bins (categorize into ranges)
        # ------------------------------------
        df = df.withColumn('amount_bin',
                           F.when(F.col('amount') < 100, 'SMALL')
                           .when(F.col('amount') < 1000, 'MEDIUM')
                           .when(F.col('amount') < 10000, 'LARGE')
                           .otherwise('VERY_LARGE'))

        logger.info("  ✅ Added: is_near_threshold, is_large_amount, is_round_amount, amount_log, amount_bin")
        return df

    def create_text_features(self, df: DataFrame) -> DataFrame:
        """
        Extract simple text-based features from descriptions.

        TEXT CAN REVEAL PATTERNS:
        -------------------------
        - Length of description (fraudsters might be vague or overly detailed)
        - Number of words
        - Presence of specific keywords

        Note: More advanced NLP (TF-IDF) comes later in the pipeline!

        Parameters:
        -----------
        df : DataFrame
            Input data with 'description' column

        Returns:
        --------
        DataFrame with text-based features
        """
        logger.info("📝 Creating text-based features...")

        # Description length
        df = df.withColumn('description_length', F.length('description'))

        # Word count
        df = df.withColumn('description_words', F.size(F.split('description', ' ')))

        # Flag for short/vague descriptions (might be suspicious)
        df = df.withColumn('is_short_description',
                           F.when(F.col('description_length') < 15, 1).otherwise(0))

        # Keyword flags - common in different fraud types
        # -----------------------------------------------
        df = df.withColumn('has_international_keyword',
                           F.when(F.lower(F.col('description')).like('%international%') |
                                  F.lower(F.col('description')).like('%foreign%') |
                                  F.lower(F.col('description')).like('%offshore%'), 1).otherwise(0))

        df = df.withColumn('has_cash_keyword',
                           F.when(F.lower(F.col('description')).like('%cash%') |
                                  F.lower(F.col('description')).like('%deposit%'), 1).otherwise(0))

        logger.info("  ✅ Added: description_length, description_words, keyword flags")
        return df

    def create_categorical_flags(self, df: DataFrame) -> DataFrame:
        """
        Create binary flags from categorical features.

        MAKING CATEGORIES USEFUL:
        -------------------------
        We turn categorical information into binary yes/no flags
        that are easier for models to understand.

        Parameters:
        -----------
        df : DataFrame
            Input data

        Returns:
        --------
        DataFrame with additional categorical flags
        """
        logger.info("🏷️  Creating categorical flags...")

        # International transaction flag
        df = df.withColumn('is_international',
                           F.when(F.col('location') != 'US', 1).otherwise(0))

        # High-risk merchant categories
        high_risk_categories = ['CASH_DEPOSIT', 'INTERNATIONAL_WIRE', 'INVESTMENT']
        df = df.withColumn('is_high_risk_category',
                           F.when(F.col('merchant_category').isin(high_risk_categories), 1).otherwise(0))

        logger.info("  ✅ Added: is_international, is_high_risk_category")
        return df

    def build_feature_pipeline(
        self,
        label_col: str = 'category',
        text_col: str = 'description'
    ) -> Pipeline:
        """
        Build a complete PySpark ML Pipeline for feature engineering.

        WHAT'S A PIPELINE?
        ------------------
        A pipeline is like an assembly line in a factory:
        Raw data → Step 1 → Step 2 → Step 3 → Final features

        Benefits:
        - Ensures same transformations on training and test data
        - Reproducible (same steps every time)
        - Easy to deploy (save the whole pipeline!)

        This pipeline handles:
        1. Text processing (NLP)
        2. Categorical encoding
        3. Feature assembly
        4. Scaling

        Parameters:
        -----------
        label_col : str
            Name of the target column we're predicting
        text_col : str
            Name of the text column to process

        Returns:
        --------
        Pipeline object ready to fit on data
        """
        logger.info("🏗️  Building feature transformation pipeline...")

        stages = []

        # ====================================================================
        # STAGE 1: TEXT PROCESSING (NLP for description field)
        # ====================================================================
        # This extracts information from transaction descriptions

        # Step 1a: Tokenization - Break text into words
        # ---------------------------------------------
        # "Cash deposit at branch" → ["Cash", "deposit", "at", "branch"]
        tokenizer = Tokenizer(inputCol=text_col, outputCol="words")
        stages.append(tokenizer)
        logger.info("  📍 Added: Tokenizer (breaks text into words)")

        # Step 1b: Remove Stop Words - Remove common words that don't help
        # ----------------------------------------------------------------
        # ["Cash", "deposit", "at", "branch"] → ["Cash", "deposit", "branch"]
        # Removes: "at", "the", "a", "is", etc. (words that appear everywhere)
        stop_words_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
        stages.append(stop_words_remover)
        logger.info("  📍 Added: StopWordsRemover (removes common words)")

        # Step 1c: HashingTF - Convert words to numerical features
        # --------------------------------------------------------
        # This uses the "hashing trick" to convert words to numbers efficiently
        # Each word gets mapped to a number, then we count occurrences
        hashing_tf = HashingTF(inputCol="filtered_words", outputCol="raw_text_features", numFeatures=100)
        stages.append(hashing_tf)
        logger.info("  📍 Added: HashingTF (converts words to numbers)")

        # Step 1d: IDF - Weight words by importance
        # -----------------------------------------
        # TF-IDF = Term Frequency - Inverse Document Frequency
        # Gives higher weight to rare, distinctive words
        # "transaction" appears everywhere → low weight
        # "offshore" appears rarely → high weight (more informative!)
        idf = IDF(inputCol="raw_text_features", outputCol="text_features")
        stages.append(idf)
        logger.info("  📍 Added: IDF (weights important words higher)")

        # ====================================================================
        # STAGE 2: CATEGORICAL ENCODING
        # ====================================================================
        # ML algorithms need numbers, not text categories

        # Encode merchant_category
        # -----------------------
        # Converts: "RETAIL", "FOOD", "INVESTMENT" → 0, 1, 2
        merchant_indexer = StringIndexer(
            inputCol="merchant_category",
            outputCol="merchant_category_index",
            handleInvalid="keep"  # Keep unknown categories
        )
        stages.append(merchant_indexer)
        logger.info("  📍 Added: StringIndexer for merchant_category")

        # One-Hot Encoding merchant category
        # ----------------------------------
        # Better representation for categories!
        # Instead of: RETAIL=0, FOOD=1
        # We create: [is_RETAIL, is_FOOD, is_INVESTMENT, ...]
        # RETAIL → [1, 0, 0, ...]
        # FOOD → [0, 1, 0, ...]
        merchant_encoder = OneHotEncoder(
            inputCol="merchant_category_index",
            outputCol="merchant_category_encoded"
        )
        stages.append(merchant_encoder)
        logger.info("  📍 Added: OneHotEncoder for merchant_category")

        # Encode location
        location_indexer = StringIndexer(
            inputCol="location",
            outputCol="location_index",
            handleInvalid="keep"
        )
        stages.append(location_indexer)

        location_encoder = OneHotEncoder(
            inputCol="location_index",
            outputCol="location_encoded"
        )
        stages.append(location_encoder)
        logger.info("  📍 Added: StringIndexer and OneHotEncoder for location")

        # Encode amount_bin
        amount_bin_indexer = StringIndexer(
            inputCol="amount_bin",
            outputCol="amount_bin_index",
            handleInvalid="keep"
        )
        stages.append(amount_bin_indexer)
        logger.info("  📍 Added: StringIndexer for amount_bin")

        # ====================================================================
        # STAGE 3: LABEL ENCODING (Our target variable)
        # ====================================================================
        # Convert category labels to numbers for classification
        # LEGITIMATE → 0, STRUCTURING → 1, LAYERING → 2, etc.
        label_indexer = StringIndexer(
            inputCol=label_col,
            outputCol="label",
            handleInvalid="keep"
        )
        stages.append(label_indexer)
        logger.info("  📍 Added: Label encoding (target variable)")

        # ====================================================================
        # STAGE 4: ASSEMBLE NUMERICAL FEATURES
        # ====================================================================
        # Combine all numerical features into a single vector
        # (ML algorithms need a single input vector)

        numerical_features = [
            'amount',
            'amount_log',
            'hour',
            'day_of_week',
            'day_of_month',
            'month',
            'is_night',
            'is_weekend',
            'is_business_hours',
            'is_near_threshold',
            'is_large_amount',
            'is_round_amount',
            'is_international',
            'is_high_risk_category',
            'description_length',
            'description_words',
            'is_short_description',
            'has_international_keyword',
            'has_cash_keyword',
            'amount_bin_index'
        ]

        numerical_assembler = VectorAssembler(
            inputCols=numerical_features,
            outputCol="numerical_features_raw"
        )
        stages.append(numerical_assembler)
        logger.info(f"  📍 Added: VectorAssembler for {len(numerical_features)} numerical features")

        # ====================================================================
        # STAGE 5: SCALE NUMERICAL FEATURES
        # ====================================================================
        # Standardize features to have mean=0 and standard deviation=1
        # WHY? Different features have different scales:
        #   - amount: $10 to $1,000,000
        #   - hour: 0 to 23
        # Scaling puts them on the same playing field!

        scaler = StandardScaler(
            inputCol="numerical_features_raw",
            outputCol="numerical_features_scaled",
            withMean=True,
            withStd=True
        )
        stages.append(scaler)
        logger.info("  📍 Added: StandardScaler (normalizes numerical features)")

        # ====================================================================
        # STAGE 6: FINAL FEATURE ASSEMBLY
        # ====================================================================
        # Combine ALL features into final vector:
        # - Scaled numerical features
        # - Text features (TF-IDF)
        # - Encoded categorical features

        final_assembler = VectorAssembler(
            inputCols=[
                "numerical_features_scaled",
                "text_features",
                "merchant_category_encoded",
                "location_encoded"
            ],
            outputCol="features"  # This is the final input for ML models!
        )
        stages.append(final_assembler)
        logger.info("  📍 Added: Final VectorAssembler (combines all features)")

        # ====================================================================
        # CREATE THE PIPELINE
        # ====================================================================
        pipeline = Pipeline(stages=stages)
        logger.info(f"✅ Pipeline created with {len(stages)} stages")
        logger.info("   Pipeline flow:")
        logger.info("   Raw Data → Text Processing → Encoding → Scaling → Final Features")

        return pipeline

    def engineer_features(self, df: DataFrame) -> DataFrame:
        """
        Apply all feature engineering transformations.

        This is the main method you'll call to transform raw data
        into ML-ready features.

        Parameters:
        -----------
        df : DataFrame
            Raw transaction data

        Returns:
        --------
        DataFrame with all engineered features
        """
        logger.info("\n" + "="*70)
        logger.info("🚀 Starting Feature Engineering Process")
        logger.info("="*70)

        # Apply all transformations in sequence
        df = self.create_time_features(df)
        df = self.create_amount_features(df)
        df = self.create_text_features(df)
        df = self.create_categorical_flags(df)

        # Cache the result (PySpark optimization)
        # ---------------------------------------
        # Caching stores the result in memory for faster repeated access
        df = df.cache()

        total_features = len(df.columns)
        logger.info(f"\n✅ Feature engineering complete!")
        logger.info(f"   Total columns: {total_features}")
        logger.info("="*70 + "\n")

        return df


def demonstrate_feature_engineering(spark: SparkSession, data_path: str):
    """
    Demonstration function showing how to use the FeatureEngineer.

    This is educational - shows you the complete workflow!

    Parameters:
    -----------
    spark : SparkSession
        PySpark session
    data_path : str
        Path to transaction data
    """
    logger.info("🎓 FEATURE ENGINEERING DEMONSTRATION")
    logger.info("="*70)

    # Load data
    logger.info(f"📂 Loading data from: {data_path}")
    df = spark.read.parquet(data_path)

    logger.info(f"   Loaded {df.count():,} transactions")
    logger.info(f"   Original columns: {len(df.columns)}")

    # Show sample of raw data
    logger.info("\n📊 Sample of raw data:")
    df.select('transaction_id', 'amount', 'description', 'category').show(5, truncate=False)

    # Initialize feature engineer
    engineer = FeatureEngineer(spark)

    # Apply feature engineering
    df_engineered = engineer.engineer_features(df)

    # Show sample of engineered data
    logger.info("\n🔧 Sample of engineered data:")
    sample_cols = ['amount', 'amount_log', 'is_near_threshold', 'hour',
                   'is_night', 'description_length', 'category']
    df_engineered.select(sample_cols).show(5)

    # Build and show pipeline
    pipeline = engineer.build_feature_pipeline()

    logger.info("\n📊 Feature Engineering Statistics:")
    logger.info(f"   Original columns: {len(df.columns)}")
    logger.info(f"   Engineered columns: {len(df_engineered.columns)}")
    logger.info(f"   New features created: {len(df_engineered.columns) - len(df.columns)}")

    logger.info("\n✅ Demonstration complete!")


if __name__ == "__main__":
    """
    Run this file directly to see feature engineering in action!

    Usage: python src/features/feature_engineering.py
    """
    # Create Spark session
    spark = SparkSession.builder \
        .appName("FeatureEngineeringDemo") \
        .master("local[*]") \
        .getOrCreate()

    # Path to data
    data_path = "data/raw/transactions.parquet"

    # Run demonstration
    try:
        demonstrate_feature_engineering(spark, data_path)
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        logger.info("💡 Make sure to generate data first:")
        logger.info("   python src/data/generate_data.py")
    finally:
        spark.stop()
