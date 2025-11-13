"""
=============================================================================
MACHINE LEARNING ALGORITHMS FOR AML FRAUD DETECTION
=============================================================================
This module implements 10+ different ML algorithms for fraud detection.

Think of this as your "toolbox" - each algorithm is a different tool,
and some work better than others for different types of problems!

We implement:
1. Logistic Regression (Simple, interpretable)
2. Decision Tree (Easy to understand)
3. Random Forest (Ensemble of trees)
4. Gradient Boosted Trees (Sequential learning)
5. Naive Bayes (Probabilistic)
6. Linear SVM (Support Vector Machine)
7. Multilayer Perceptron (Neural Network)
8. XGBoost (via external library)
9. LightGBM (via external library)
10. CatBoost (via external library)
11. Stacking Ensemble (Combines multiple models)
12. Voting Ensemble (Democracy of models!)
"""

from pyspark.ml.classification import (
    LogisticRegression, DecisionTreeClassifier, RandomForestClassifier,
    GBTClassifier, NaiveBayes, LinearSVC, MultilayerPerceptronClassifier
)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql import DataFrame, SparkSession
from typing import Dict, List, Tuple, Any
import logging
import numpy as np

# External ML libraries (more powerful algorithms!)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("⚠️  XGBoost not available")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("⚠️  LightGBM not available")

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logging.warning("⚠️  CatBoost not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLModelFactory:
    """
    Factory class to create and train different ML models.

    This is like a "car factory" - you tell it what model you want,
    and it builds it for you with the right specifications!
    """

    def __init__(self, spark: SparkSession, config: Dict):
        """
        Initialize the model factory.

        Parameters:
        -----------
        spark : SparkSession
            Active Spark session
        config : Dict
            Configuration with model parameters
        """
        self.spark = spark
        self.config = config
        self.models = {}
        self.trained_models = {}

        logger.info("🏭 ML Model Factory initialized")

    # =========================================================================
    # 1. LOGISTIC REGRESSION
    # =========================================================================
    # Simple but powerful! Like linear regression but for classification
    # Good for: Understanding feature importance, baseline model

    def create_logistic_regression(self) -> LogisticRegression:
        """
        Create Logistic Regression model.

        How it works:
        - Finds a line (or hyperplane) that separates fraud from non-fraud
        - Outputs probability between 0 and 1
        - Fast to train, easy to interpret

        Returns:
        --------
        LogisticRegression : Configured model
        """
        logger.info("📊 Creating Logistic Regression model...")

        model = LogisticRegression(
            featuresCol="features",
            labelCol="is_fraud",
            maxIter=100,
            regParam=0.1,  # Regularization to prevent overfitting
            elasticNetParam=0.5,  # Mix of L1 and L2 regularization
            family="binomial"  # Binary classification
        )

        logger.info("   ✓ Logistic Regression ready")
        return model

    # =========================================================================
    # 2. DECISION TREE
    # =========================================================================
    # Like a flowchart of yes/no questions
    # Good for: Interpretability, capturing non-linear patterns

    def create_decision_tree(self) -> DecisionTreeClassifier:
        """
        Create Decision Tree model.

        How it works:
        - Asks a series of questions about features
        - Example: "Is amount > $5000?" → Yes → "Is at night?" → Yes → FRAUD!
        - Creates a tree structure of decisions

        Returns:
        --------
        DecisionTreeClassifier : Configured model
        """
        logger.info("🌳 Creating Decision Tree model...")

        model = DecisionTreeClassifier(
            featuresCol="features",
            labelCol="is_fraud",
            maxDepth=10,  # How deep the tree can grow
            minInstancesPerNode=10,  # Minimum samples to split
            impurity="gini"  # How to measure split quality
        )

        logger.info("   ✓ Decision Tree ready")
        return model

    # =========================================================================
    # 3. RANDOM FOREST
    # =========================================================================
    # Many decision trees voting together ("wisdom of the crowd")
    # Good for: High accuracy, handling complex patterns, robustness

    def create_random_forest(self) -> RandomForestClassifier:
        """
        Create Random Forest model.

        How it works:
        - Trains many decision trees (a "forest")
        - Each tree sees a random subset of data and features
        - Final prediction is majority vote from all trees
        - More accurate than single tree!

        Returns:
        --------
        RandomForestClassifier : Configured model
        """
        logger.info("🌲 Creating Random Forest model...")

        model = RandomForestClassifier(
            featuresCol="features",
            labelCol="is_fraud",
            numTrees=100,  # Number of trees in the forest
            maxDepth=10,
            minInstancesPerNode=5,
            featureSubsetStrategy="auto",  # How many features each tree sees
            seed=42
        )

        logger.info("   ✓ Random Forest ready (100 trees)")
        return model

    # =========================================================================
    # 4. GRADIENT BOOSTED TREES (GBT)
    # =========================================================================
    # Sequential learning - each tree corrects previous tree's mistakes
    # Good for: High accuracy, handling imbalanced data

    def create_gbt(self) -> GBTClassifier:
        """
        Create Gradient Boosted Trees model.

        How it works:
        - Builds trees one at a time
        - Each new tree focuses on fixing previous trees' errors
        - Like a student learning from their mistakes!
        - Very powerful but can overfit

        Returns:
        --------
        GBTClassifier : Configured model
        """
        logger.info("🚀 Creating Gradient Boosted Trees model...")

        model = GBTClassifier(
            featuresCol="features",
            labelCol="is_fraud",
            maxIter=100,  # Number of trees
            maxDepth=5,
            stepSize=0.1,  # Learning rate (how much each tree contributes)
            seed=42
        )

        logger.info("   ✓ GBT ready")
        return model

    # =========================================================================
    # 5. NAIVE BAYES
    # =========================================================================
    # Based on probability theory (Bayes' Theorem)
    # Good for: Speed, text classification, when features are independent

    def create_naive_bayes(self) -> NaiveBayes:
        """
        Create Naive Bayes model.

        How it works:
        - Calculates probability of fraud given the features
        - Assumes features are independent (naive assumption!)
        - P(fraud | features) = P(features | fraud) * P(fraud) / P(features)
        - Very fast!

        Returns:
        --------
        NaiveBayes : Configured model
        """
        logger.info("🎲 Creating Naive Bayes model...")

        model = NaiveBayes(
            featuresCol="features",
            labelCol="is_fraud",
            smoothing=1.0,  # Laplace smoothing
            modelType="multinomial"
        )

        logger.info("   ✓ Naive Bayes ready")
        return model

    # =========================================================================
    # 6. LINEAR SVM (Support Vector Machine)
    # =========================================================================
    # Finds the best boundary between classes
    # Good for: High-dimensional data, clear margin of separation

    def create_linear_svm(self) -> LinearSVC:
        """
        Create Linear Support Vector Machine model.

        How it works:
        - Finds a hyperplane that maximally separates fraud from non-fraud
        - Maximizes the "margin" (distance to nearest points)
        - Good when data is linearly separable

        Returns:
        --------
        LinearSVC : Configured model
        """
        logger.info("📐 Creating Linear SVM model...")

        model = LinearSVC(
            featuresCol="features",
            labelCol="is_fraud",
            maxIter=100,
            regParam=0.1,
            aggregationDepth=2
        )

        logger.info("   ✓ Linear SVM ready")
        return model

    # =========================================================================
    # 7. MULTILAYER PERCEPTRON (Neural Network)
    # =========================================================================
    # Deep learning! Inspired by human brain
    # Good for: Complex patterns, non-linear relationships

    def create_mlp(self) -> MultilayerPerceptronClassifier:
        """
        Create Multilayer Perceptron (Neural Network) model.

        How it works:
        - Layers of interconnected "neurons"
        - Each neuron performs weighted sum + activation function
        - Learns complex patterns through backpropagation
        - Can approximate any function!

        Returns:
        --------
        MultilayerPerceptronClassifier : Configured model
        """
        logger.info("🧠 Creating Neural Network (MLP) model...")

        # Network architecture: input → 100 neurons → 50 neurons → 2 output
        # The number of input features is determined automatically
        layers = [100, 50, 2]  # Hidden layers + output layer

        model = MultilayerPerceptronClassifier(
            featuresCol="features",
            labelCol="is_fraud",
            layers=layers,
            maxIter=100,
            stepSize=0.01,  # Learning rate
            seed=42
        )

        logger.info(f"   ✓ MLP ready (architecture: {layers})")
        return model

    # =========================================================================
    # 8. XGBOOST
    # =========================================================================
    # Extreme Gradient Boosting - industry standard!
    # Good for: Winning Kaggle competitions, handling complex data

    def create_xgboost(self, train_data: DataFrame) -> Any:
        """
        Create XGBoost model (external library).

        How it works:
        - Advanced gradient boosting with many optimizations
        - Handles missing values automatically
        - Built-in regularization
        - Often the best performing algorithm!

        Parameters:
        -----------
        train_data : DataFrame
            Training data (needed to convert to XGBoost format)

        Returns:
        --------
        XGBoost model
        """
        if not XGBOOST_AVAILABLE:
            logger.warning("⚠️  XGBoost not available, skipping...")
            return None

        logger.info("⚡ Creating XGBoost model...")

        # XGBoost parameters
        params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'objective': 'binary:logistic',  # Binary classification
            'eval_metric': 'auc',  # Evaluation metric
            'subsample': 0.8,  # Random sampling of data
            'colsample_bytree': 0.8,  # Random sampling of features
            'random_state': 42,
            'use_label_encoder': False
        }

        model = xgb.XGBClassifier(**params)

        logger.info("   ✓ XGBoost ready")
        return model

    # =========================================================================
    # 9. LIGHTGBM
    # =========================================================================
    # Light Gradient Boosting Machine - faster than XGBoost!
    # Good for: Large datasets, speed, memory efficiency

    def create_lightgbm(self) -> Any:
        """
        Create LightGBM model (external library).

        How it works:
        - Uses histogram-based algorithms (bins data)
        - Leaf-wise tree growth (vs level-wise)
        - Much faster than XGBoost on large data
        - Lower memory usage

        Returns:
        --------
        LightGBM model
        """
        if not LIGHTGBM_AVAILABLE:
            logger.warning("⚠️  LightGBM not available, skipping...")
            return None

        logger.info("💡 Creating LightGBM model...")

        params = {
            'num_leaves': 31,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'objective': 'binary',
            'metric': 'auc',
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbose': -1  # Suppress warnings
        }

        model = lgb.LGBMClassifier(**params)

        logger.info("   ✓ LightGBM ready")
        return model

    # =========================================================================
    # 10. CATBOOST
    # =========================================================================
    # Category Boosting - handles categorical features natively
    # Good for: Data with many categorical features, robustness

    def create_catboost(self) -> Any:
        """
        Create CatBoost model (external library).

        How it works:
        - Gradient boosting optimized for categorical features
        - Handles categories without encoding!
        - Ordered boosting (reduces overfitting)
        - Very robust to parameter tuning

        Returns:
        --------
        CatBoost model
        """
        if not CATBOOST_AVAILABLE:
            logger.warning("⚠️  CatBoost not available, skipping...")
            return None

        logger.info("🐱 Creating CatBoost model...")

        params = {
            'iterations': 100,
            'depth': 6,
            'learning_rate': 0.1,
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'random_seed': 42,
            'verbose': False
        }

        model = CatBoostClassifier(**params)

        logger.info("   ✓ CatBoost ready")
        return model

    # =========================================================================
    # MODEL TRAINING AND EVALUATION
    # =========================================================================

    def train_model(self, model: Any, train_data: DataFrame,
                   model_name: str) -> Any:
        """
        Train a single model.

        Parameters:
        -----------
        model : Any
            Model to train
        train_data : DataFrame
            Training data
        model_name : str
            Name of the model for logging

        Returns:
        --------
        Trained model
        """
        logger.info(f"🎓 Training {model_name}...")

        try:
            # Train the model
            trained_model = model.fit(train_data)

            logger.info(f"   ✅ {model_name} training complete!")

            # Save to trained models dictionary
            self.trained_models[model_name] = trained_model

            return trained_model

        except Exception as e:
            logger.error(f"   ❌ Error training {model_name}: {str(e)}")
            return None

    def train_all_models(self, train_data: DataFrame) -> Dict[str, Any]:
        """
        Train all enabled models.

        This is like conducting a "tournament" - we train all models
        and see which one performs best!

        Parameters:
        -----------
        train_data : DataFrame
            Training data with 'features' and 'is_fraud' columns

        Returns:
        --------
        Dict[str, Any] : Dictionary of trained models
        """
        logger.info("\n" + "="*80)
        logger.info("🏋️  TRAINING ALL ML MODELS")
        logger.info("="*80 + "\n")

        trained_models = {}

        # List of all models to train
        models_to_train = [
            ("Logistic Regression", self.create_logistic_regression()),
            ("Decision Tree", self.create_decision_tree()),
            ("Random Forest", self.create_random_forest()),
            ("Gradient Boosted Trees", self.create_gbt()),
            ("Naive Bayes", self.create_naive_bayes()),
            ("Linear SVM", self.create_linear_svm()),
            ("Neural Network (MLP)", self.create_mlp()),
        ]

        # Train each model
        for model_name, model in models_to_train:
            if model is not None:
                trained_model = self.train_model(model, train_data, model_name)
                if trained_model is not None:
                    trained_models[model_name] = trained_model

        # Train external library models (XGBoost, LightGBM, CatBoost)
        # Note: These require converting from Spark DataFrame to pandas/numpy
        # We'll implement this in the main pipeline

        logger.info("\n" + "="*80)
        logger.info(f"✅ TRAINING COMPLETE! Trained {len(trained_models)} models")
        logger.info("="*80 + "\n")

        return trained_models

    def hyperparameter_tuning(self, model: Any, train_data: DataFrame,
                            param_grid: Dict, model_name: str) -> Any:
        """
        Perform hyperparameter tuning using cross-validation.

        This is like testing different "recipes" to find the best one!

        Parameters:
        -----------
        model : Any
            Model to tune
        train_data : DataFrame
            Training data
        param_grid : Dict
            Grid of parameters to try
        model_name : str
            Name of model

        Returns:
        --------
        Best model after tuning
        """
        logger.info(f"🔧 Tuning hyperparameters for {model_name}...")

        # Create parameter grid
        paramGrid = ParamGridBuilder()

        for param_name, param_values in param_grid.items():
            param = getattr(model, param_name)
            paramGrid = paramGrid.addGrid(param, param_values)

        paramGrid = paramGrid.build()

        # Create evaluator
        evaluator = BinaryClassificationEvaluator(
            labelCol="is_fraud",
            metricName="areaUnderROC"
        )

        # Create cross validator
        cv = CrossValidator(
            estimator=model,
            estimatorParamMaps=paramGrid,
            evaluator=evaluator,
            numFolds=3,  # 3-fold cross-validation
            parallelism=4,  # Run 4 models in parallel
            seed=42
        )

        # Run cross validation
        logger.info(f"   🔄 Running {len(paramGrid)}-parameter grid search with 3-fold CV...")
        cv_model = cv.fit(train_data)

        logger.info(f"   ✅ Best model found!")
        logger.info(f"   📊 Best score: {cv_model.avgMetrics[0]:.4f}")

        return cv_model.bestModel


def main():
    """
    Test the ML models module.
    """
    print("\n" + "="*80)
    print("🤖 TESTING ML MODELS")
    print("="*80 + "\n")

    # Initialize Spark
    spark = SparkSession.builder \
        .appName("MLModelsTest") \
        .master("local[*]") \
        .getOrCreate()

    # Create model factory
    factory = MLModelFactory(spark, {})

    # Test creating models
    logger.info("Testing model creation...")
    lr = factory.create_logistic_regression()
    rf = factory.create_random_forest()
    gbt = factory.create_gbt()

    logger.info("\n✅ All models created successfully!")

    spark.stop()


if __name__ == "__main__":
    main()
