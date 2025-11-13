"""
============================================================================
ML MODEL TRAINER - 10+ ALGORITHMS FOR AML FRAUD DETECTION
============================================================================

Welcome to the model training module! This is where the AI magic happens! ✨

WHAT ARE WE DOING HERE?
-----------------------
We're training 10+ different machine learning algorithms to detect fraud.
Think of each algorithm as a different "expert" with their own way of
finding patterns in data.

WHY SO MANY ALGORITHMS?
-----------------------
- No single algorithm is always best
- Different algorithms spot different patterns
- We can compare and ensemble them
- Learn which approaches work best for AML detection

THE ALGORITHMS WE'LL USE:
-------------------------
1. Logistic Regression - The simple starter (fast, interpretable)
2. Decision Tree - Tree-based decisions (easy to understand)
3. Random Forest - Multiple trees voting (robust)
4. Gradient Boosted Trees - Trees learning from mistakes (powerful)
5. Naive Bayes - Probability-based (fast, good for text)
6. Linear SVC - Support Vector Classifier (finds decision boundaries)
7. Multilayer Perceptron - Neural network (learns complex patterns)
8. One-vs-Rest - Breaks multiclass into binary problems
9. Random Forest (tuned) - Optimized version
10. GBT (tuned) - Optimized version
11. Ensemble Voting - Combines multiple models

MACHINE LEARNING CRASH COURSE:
------------------------------
Training = Showing the model examples until it learns patterns
Testing = Checking if it learned by testing on new examples
Validation = Making sure it didn't just memorize (overfitting)

Let's train some models! 🚀
============================================================================
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.classification import (
    LogisticRegression,
    DecisionTreeClassifier,
    RandomForestClassifier,
    GBTClassifier,
    NaiveBayes,
    LinearSVC,
    MultilayerPerceptronClassifier,
    OneVsRest
)
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from typing import Dict, List, Tuple, Any
import logging
from datetime import datetime
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Trains and manages multiple ML models for fraud detection.

    This class is your ML training assistant - it handles:
    - Training multiple algorithms
    - Tracking performance
    - Saving models
    - Comparing results
    """

    def __init__(self, spark: SparkSession):
        """
        Initialize the model trainer.

        Parameters:
        -----------
        spark : SparkSession
            PySpark session for distributed computing
        """
        self.spark = spark
        self.models = {}  # Store trained models
        self.results = {}  # Store performance metrics
        logger.info("🤖 Model Trainer initialized")

    def split_data(
        self,
        df: DataFrame,
        train_ratio: float = 0.7,
        validation_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        """
        Split data into training, validation, and test sets.

        WHY THREE SPLITS?
        -----------------
        1. TRAINING SET (70%): Teach the model
        2. VALIDATION SET (15%): Tune hyperparameters
        3. TEST SET (15%): Final honest evaluation

        Think of it like school:
        - Training = Studying from textbook
        - Validation = Practice exams
        - Test = Final exam

        Parameters:
        -----------
        df : DataFrame
            Complete dataset
        train_ratio : float
            Fraction for training (default 70%)
        validation_ratio : float
            Fraction for validation (default 15%)
        test_ratio : float
            Fraction for testing (default 15%)
        seed : int
            Random seed for reproducibility

        Returns:
        --------
        Tuple of (train_df, validation_df, test_df)
        """
        logger.info("✂️  Splitting data into train/validation/test sets...")

        # Validate ratios sum to 1.0
        assert abs(train_ratio + validation_ratio + test_ratio - 1.0) < 0.001, \
            "Ratios must sum to 1.0"

        # First split: separate test set
        train_val, test = df.randomSplit(
            [train_ratio + validation_ratio, test_ratio],
            seed=seed
        )

        # Second split: separate train and validation
        train, validation = train_val.randomSplit(
            [train_ratio / (train_ratio + validation_ratio),
             validation_ratio / (train_ratio + validation_ratio)],
            seed=seed
        )

        # Cache datasets for faster access
        train = train.cache()
        validation = validation.cache()
        test = test.cache()

        logger.info(f"  ✅ Training set: {train.count():,} samples")
        logger.info(f"  ✅ Validation set: {validation.count():,} samples")
        logger.info(f"  ✅ Test set: {test.count():,} samples")

        # Show class distribution in each set
        logger.info("\n  📊 Class distribution:")
        for name, dataset in [("Train", train), ("Validation", validation), ("Test", test)]:
            dist = dataset.groupBy("label").count().orderBy("label").collect()
            logger.info(f"    {name}: {dist}")

        return train, validation, test

    def train_logistic_regression(
        self,
        train_df: DataFrame,
        name: str = "Logistic_Regression"
    ) -> Tuple[LogisticRegression, Any]:
        """
        Train a Logistic Regression model.

        WHAT IS LOGISTIC REGRESSION?
        ----------------------------
        Despite the name, it's used for CLASSIFICATION!
        It finds the probability that a transaction belongs to each class.

        Pros:
        - Fast to train
        - Interpretable (can see feature importance)
        - Works well as a baseline
        - Good for linearly separable data

        Cons:
        - Assumes linear relationships
        - Might miss complex patterns

        When to use: Always start with this as your baseline!

        Parameters:
        -----------
        train_df : DataFrame
            Training data with 'features' and 'label' columns
        name : str
            Model name for tracking

        Returns:
        --------
        Tuple of (model, trained_model)
        """
        logger.info(f"\n🎯 Training {name}...")
        logger.info("  Algorithm: Logistic Regression")
        logger.info("  Type: Linear classifier")
        logger.info("  Strength: Fast and interpretable")

        # Configure the model
        lr = LogisticRegression(
            featuresCol='features',
            labelCol='label',
            predictionCol='prediction',
            maxIter=100,  # Maximum training iterations
            regParam=0.01,  # Regularization (prevents overfitting)
            elasticNetParam=0.0,  # L2 regularization
            family='multinomial'  # For multiclass classification
        )

        # Train the model
        logger.info("  🔄 Training in progress...")
        model = lr.fit(train_df)

        logger.info(f"  ✅ {name} training complete!")
        self.models[name] = model

        return lr, model

    def train_decision_tree(
        self,
        train_df: DataFrame,
        name: str = "Decision_Tree",
        max_depth: int = 10
    ) -> Tuple[DecisionTreeClassifier, Any]:
        """
        Train a Decision Tree model.

        WHAT IS A DECISION TREE?
        ------------------------
        Makes decisions like a flowchart:
        "Is amount > $10,000?"
          ├─ Yes: "Is it international?"
          │    ├─ Yes: HIGH RISK
          │    └─ No: Check next...
          └─ No: Check next...

        Pros:
        - Easy to understand and visualize
        - Handles non-linear patterns
        - No need for feature scaling
        - Can capture interactions between features

        Cons:
        - Can overfit (memorize training data)
        - Unstable (small changes in data → different tree)
        - Might not generalize well

        When to use: When you need interpretable results!

        Parameters:
        -----------
        train_df : DataFrame
            Training data
        name : str
            Model name
        max_depth : int
            Maximum tree depth (prevents overfitting)

        Returns:
        --------
        Tuple of (model, trained_model)
        """
        logger.info(f"\n🌲 Training {name}...")
        logger.info("  Algorithm: Decision Tree")
        logger.info("  Type: Tree-based classifier")
        logger.info("  Strength: Interpretable, handles non-linear patterns")

        dt = DecisionTreeClassifier(
            featuresCol='features',
            labelCol='label',
            predictionCol='prediction',
            maxDepth=max_depth,  # Limit depth to prevent overfitting
            minInstancesPerNode=10,  # Minimum samples in leaf nodes
            impurity='gini'  # Gini impurity for splitting
        )

        logger.info(f"  🔄 Training with max_depth={max_depth}...")
        model = dt.fit(train_df)

        logger.info(f"  ✅ {name} training complete!")
        logger.info(f"     Tree depth: {model.depth}")
        logger.info(f"     Number of nodes: {model.numNodes}")
        self.models[name] = model

        return dt, model

    def train_random_forest(
        self,
        train_df: DataFrame,
        name: str = "Random_Forest",
        num_trees: int = 100,
        max_depth: int = 10
    ) -> Tuple[RandomForestClassifier, Any]:
        """
        Train a Random Forest model.

        WHAT IS RANDOM FOREST?
        ----------------------
        Ensemble of many decision trees voting together!
        "Wisdom of crowds" approach.

        How it works:
        1. Create 100 different decision trees
        2. Each tree trained on random subset of data
        3. Each tree votes on prediction
        4. Final prediction = majority vote

        Why it's awesome:
        - More robust than single decision tree
        - Reduces overfitting
        - Handles complex patterns
        - Often wins Kaggle competitions!

        Pros:
        - High accuracy
        - Handles missing values well
        - Works with both numerical and categorical features
        - Less prone to overfitting than single tree

        Cons:
        - Slower to train than single tree
        - Less interpretable (100 trees!)
        - Can be memory intensive

        When to use: One of your go-to algorithms for most problems!

        Parameters:
        -----------
        train_df : DataFrame
            Training data
        name : str
            Model name
        num_trees : int
            Number of trees in the forest
        max_depth : int
            Maximum depth of each tree

        Returns:
        --------
        Tuple of (model, trained_model)
        """
        logger.info(f"\n🌳 Training {name}...")
        logger.info("  Algorithm: Random Forest")
        logger.info("  Type: Ensemble of decision trees")
        logger.info("  Strength: Robust, handles complex patterns")

        rf = RandomForestClassifier(
            featuresCol='features',
            labelCol='label',
            predictionCol='prediction',
            numTrees=num_trees,  # Number of trees to train
            maxDepth=max_depth,  # Depth of each tree
            minInstancesPerNode=5,
            subsamplingRate=0.8,  # Use 80% of data for each tree (bootstrap)
            featureSubsetStrategy='sqrt'  # Number of features per tree
        )

        logger.info(f"  🔄 Training {num_trees} trees with max_depth={max_depth}...")
        model = rf.fit(train_df)

        logger.info(f"  ✅ {name} training complete!")
        logger.info(f"     Number of trees: {model.getNumTrees}")
        self.models[name] = model

        return rf, model

    def train_gradient_boosted_trees(
        self,
        train_df: DataFrame,
        name: str = "Gradient_Boosted_Trees",
        max_iter: int = 50,
        max_depth: int = 5
    ) -> Tuple[GBTClassifier, Any]:
        """
        Train a Gradient Boosted Trees model.

        WHAT IS GRADIENT BOOSTING?
        --------------------------
        Sequential ensemble: trees learn from previous trees' mistakes!

        How it works:
        1. Train tree #1 → Makes some mistakes
        2. Train tree #2 → Focuses on tree #1's mistakes
        3. Train tree #3 → Focuses on combined mistakes
        4. ... continue until performance plateaus
        5. Final prediction = weighted combination

        Think of it like studying:
        - First pass: Learn basics (get 70% right)
        - Second pass: Focus on what you got wrong (now 85% right)
        - Third pass: Fix remaining mistakes (now 95% right!)

        Pros:
        - Often gives best performance
        - Wins many ML competitions
        - Handles complex patterns
        - Good with imbalanced classes

        Cons:
        - Slower to train (sequential, not parallel)
        - More prone to overfitting than Random Forest
        - Requires careful tuning
        - PySpark GBT only supports binary classification directly

        When to use: When you need top performance and have time to tune!

        Parameters:
        -----------
        train_df : DataFrame
            Training data
        name : str
            Model name
        max_iter : int
            Number of trees (iterations)
        max_depth : int
            Depth of each tree

        Returns:
        --------
        Tuple of (model, trained_model)
        """
        logger.info(f"\n🚀 Training {name}...")
        logger.info("  Algorithm: Gradient Boosted Trees")
        logger.info("  Type: Sequential ensemble")
        logger.info("  Strength: High performance, learns from mistakes")
        logger.info("  Note: Using One-vs-Rest for multiclass")

        # Note: PySpark GBT works best with binary classification
        # For multiclass, we'll train for each class separately (One-vs-Rest)
        gbt = GBTClassifier(
            featuresCol='features',
            labelCol='label',
            predictionCol='prediction',
            maxIter=max_iter,
            maxDepth=max_depth,
            stepSize=0.1  # Learning rate (smaller = more conservative)
        )

        # Wrap in One-vs-Rest for multiclass
        ovr = OneVsRest(
            classifier=gbt,
            featuresCol='features',
            labelCol='label',
            predictionCol='prediction'
        )

        logger.info(f"  🔄 Training {max_iter} iterations with max_depth={max_depth}...")
        model = ovr.fit(train_df)

        logger.info(f"  ✅ {name} training complete!")
        self.models[name] = model

        return ovr, model

    def train_naive_bayes(
        self,
        train_df: DataFrame,
        name: str = "Naive_Bayes"
    ) -> Tuple[NaiveBayes, Any]:
        """
        Train a Naive Bayes model.

        WHAT IS NAIVE BAYES?
        --------------------
        Based on Bayes' Theorem (probability theory from 1700s!)
        Calculates: "Given these features, what's the probability of each class?"

        Why "Naive"?
        - Assumes features are independent (rarely true, but works anyway!)
        - Example: Assumes "amount" and "location" are independent
        - This "naive" assumption makes it fast!

        Pros:
        - Very fast to train and predict
        - Works well with text data (great for our descriptions!)
        - Handles multiclass naturally
        - Requires little training data

        Cons:
        - The independence assumption is often wrong
        - Can be outperformed by more complex models
        - Sensitive to irrelevant features

        When to use: Fast baseline, especially for text-heavy features!

        Parameters:
        -----------
        train_df : DataFrame
            Training data
        name : str
            Model name

        Returns:
        --------
        Tuple of (model, trained_model)
        """
        logger.info(f"\n🎲 Training {name}...")
        logger.info("  Algorithm: Naive Bayes")
        logger.info("  Type: Probabilistic classifier")
        logger.info("  Strength: Fast, works well with text")

        nb = NaiveBayes(
            featuresCol='features',
            labelCol='label',
            predictionCol='prediction',
            smoothing=1.0,  # Laplace smoothing
            modelType='multinomial'  # For multiclass
        )

        logger.info("  🔄 Training...")
        model = nb.fit(train_df)

        logger.info(f"  ✅ {name} training complete!")
        self.models[name] = model

        return nb, model

    def train_linear_svc(
        self,
        train_df: DataFrame,
        name: str = "Linear_SVC"
    ) -> Tuple[Any, Any]:
        """
        Train a Linear Support Vector Classifier.

        WHAT IS SVC?
        ------------
        Finds the best "decision boundary" (hyperplane) to separate classes.

        Imagine plotting points on a graph:
        - Red dots (fraud) on one side
        - Blue dots (legitimate) on the other
        - SVC finds the line that best separates them
        - But also maximizes the "margin" (distance to nearest points)

        Pros:
        - Effective in high-dimensional spaces
        - Memory efficient
        - Works well when classes are separable
        - Robust to outliers

        Cons:
        - PySpark LinearSVC is binary only (need One-vs-Rest)
        - Slower on very large datasets
        - Requires feature scaling (we did this!)

        When to use: When you have high-dimensional data with clear separation!

        Parameters:
        -----------
        train_df : DataFrame
            Training data
        name : str
            Model name

        Returns:
        --------
        Tuple of (model, trained_model)
        """
        logger.info(f"\n📐 Training {name}...")
        logger.info("  Algorithm: Linear Support Vector Classifier")
        logger.info("  Type: Linear classifier")
        logger.info("  Strength: Works well in high dimensions")
        logger.info("  Note: Using One-vs-Rest for multiclass")

        svc = LinearSVC(
            featuresCol='features',
            labelCol='label',
            predictionCol='prediction',
            maxIter=100,
            regParam=0.01  # Regularization
        )

        # Wrap in One-vs-Rest for multiclass
        ovr = OneVsRest(
            classifier=svc,
            featuresCol='features',
            labelCol='label',
            predictionCol='prediction'
        )

        logger.info("  🔄 Training with One-vs-Rest strategy...")
        model = ovr.fit(train_df)

        logger.info(f"  ✅ {name} training complete!")
        self.models[name] = model

        return ovr, model

    def train_neural_network(
        self,
        train_df: DataFrame,
        name: str = "Neural_Network",
        layers: List[int] = None
    ) -> Tuple[MultilayerPerceptronClassifier, Any]:
        """
        Train a Multilayer Perceptron (Neural Network).

        WHAT IS A NEURAL NETWORK?
        -------------------------
        Inspired by the human brain! Made of interconnected "neurons."

        Structure:
        Input Layer → Hidden Layer(s) → Output Layer

        Each connection has a "weight" that's learned during training.
        Like your brain strengthening connections as you learn!

        Pros:
        - Can learn very complex patterns
        - Can approximate any function (universal approximator)
        - Great for non-linear problems
        - Foundation of deep learning

        Cons:
        - Needs more data than traditional ML
        - Slower to train
        - "Black box" - hard to interpret
        - Requires tuning (architecture, learning rate, etc.)

        When to use: When you have lots of data and complex patterns!

        Parameters:
        -----------
        train_df : DataFrame
            Training data
        name : str
            Model name
        layers : List[int]
            Network architecture [input, hidden, ..., output]

        Returns:
        --------
        Tuple of (model, trained_model)
        """
        logger.info(f"\n🧠 Training {name}...")
        logger.info("  Algorithm: Multilayer Perceptron (Neural Network)")
        logger.info("  Type: Deep learning")
        logger.info("  Strength: Learns complex patterns")

        # Get number of features and classes
        if layers is None:
            # Get feature vector size from first row
            sample = train_df.select('features').first()
            num_features = len(sample['features'])

            # Get number of classes
            num_classes = train_df.select('label').distinct().count()

            # Default architecture: input → hidden → hidden → output
            # Hidden layers typically between input and output size
            layers = [
                num_features,  # Input layer
                128,  # First hidden layer
                64,  # Second hidden layer
                num_classes  # Output layer
            ]

        logger.info(f"  Network architecture: {layers}")
        logger.info(f"    ├─ Input layer: {layers[0]} neurons")
        for i, h in enumerate(layers[1:-1], 1):
            logger.info(f"    ├─ Hidden layer {i}: {h} neurons")
        logger.info(f"    └─ Output layer: {layers[-1]} neurons")

        mlp = MultilayerPerceptronClassifier(
            featuresCol='features',
            labelCol='label',
            predictionCol='prediction',
            layers=layers,
            maxIter=100,
            stepSize=0.03,  # Learning rate
            seed=42
        )

        logger.info("  🔄 Training neural network...")
        logger.info("     (This might take a while...)")
        model = mlp.fit(train_df)

        logger.info(f"  ✅ {name} training complete!")
        self.models[name] = model

        return mlp, model

    def train_all_models(
        self,
        train_df: DataFrame
    ) -> Dict[str, Any]:
        """
        Train all models sequentially.

        This is your "train everything" button!
        Trains 10+ different algorithms and compares them.

        Parameters:
        -----------
        train_df : DataFrame
            Training data

        Returns:
        --------
        Dictionary of {model_name: trained_model}
        """
        logger.info("\n" + "="*70)
        logger.info("🚀 TRAINING ALL MODELS")
        logger.info("="*70)
        logger.info("This will train 10+ different algorithms.")
        logger.info("Grab a coffee! This might take a few minutes... ☕")
        logger.info("="*70)

        start_time = datetime.now()

        # Train each model
        try:
            # Model 1: Logistic Regression (baseline)
            self.train_logistic_regression(train_df, "Logistic_Regression_Baseline")

            # Model 2: Decision Tree (default)
            self.train_decision_tree(train_df, "Decision_Tree_Default", max_depth=10)

            # Model 3: Decision Tree (deeper)
            self.train_decision_tree(train_df, "Decision_Tree_Deep", max_depth=20)

            # Model 4: Random Forest (default)
            self.train_random_forest(train_df, "Random_Forest_Default", num_trees=50)

            # Model 5: Random Forest (more trees)
            self.train_random_forest(train_df, "Random_Forest_Large", num_trees=100)

            # Model 6: Random Forest (tuned)
            self.train_random_forest(train_df, "Random_Forest_Tuned",
                                    num_trees=100, max_depth=15)

            # Model 7: Gradient Boosted Trees
            self.train_gradient_boosted_trees(train_df, "GBT_Default",
                                             max_iter=30, max_depth=5)

            # Model 8: GBT (more iterations)
            self.train_gradient_boosted_trees(train_df, "GBT_Large",
                                             max_iter=50, max_depth=5)

            # Model 9: Naive Bayes
            self.train_naive_bayes(train_df, "Naive_Bayes")

            # Model 10: Linear SVC
            self.train_linear_svc(train_df, "Linear_SVC")

            # Model 11: Neural Network (small)
            self.train_neural_network(train_df, "Neural_Network_Small")

            # Model 12: Logistic Regression (tuned)
            lr_tuned = LogisticRegression(
                featuresCol='features',
                labelCol='label',
                maxIter=200,
                regParam=0.001,
                elasticNetParam=0.1,
                family='multinomial'
            )
            model_tuned = lr_tuned.fit(train_df)
            self.models["Logistic_Regression_Tuned"] = model_tuned
            logger.info("✅ Logistic_Regression_Tuned training complete!")

        except Exception as e:
            logger.error(f"❌ Error during training: {e}")
            raise

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        logger.info("\n" + "="*70)
        logger.info("✅ ALL MODELS TRAINED SUCCESSFULLY!")
        logger.info("="*70)
        logger.info(f"Total models trained: {len(self.models)}")
        logger.info(f"Total training time: {duration:.2f} seconds ({duration/60:.1f} minutes)")
        logger.info(f"Average time per model: {duration/len(self.models):.2f} seconds")
        logger.info("="*70)

        return self.models

    def save_model(
        self,
        model: Any,
        name: str,
        output_dir: str = "data/models"
    ):
        """
        Save a trained model to disk.

        WHY SAVE MODELS?
        ----------------
        - Don't retrain every time!
        - Deploy to production
        - Version control your models
        - Share with teammates

        Parameters:
        -----------
        model : Any
            Trained model
        name : str
            Model name
        output_dir : str
            Where to save
        """
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, name)

        try:
            model.write().overwrite().save(model_path)
            logger.info(f"💾 Saved {name} to {model_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save {name}: {e}")


if __name__ == "__main__":
    """
    Run this directly to train models!

    Usage: python src/models/model_trainer.py
    """
    logger.info("🚀 Model Training Demo")
