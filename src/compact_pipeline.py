#!/usr/bin/env python3
"""
Compact AML Fraud Detection Pipeline
=====================================
Minimal, production-ready ML pipeline in ~200 lines

Usage: python src/compact_pipeline.py
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import *
from pyspark.ml.classification import *
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import random
import os

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    'data': {'num_transactions': 50000, 'fraud_ratio': 0.2},
    'models': {'num_trees': 100, 'max_depth': 10, 'max_iter': 50},
    'paths': {
        'data': 'data/raw/transactions.parquet',
        'models': 'data/models',
        'reports': 'data/reports'
    }
}

FRAUD_PATTERNS = {
    'LEGITIMATE': lambda: {
        'amount': random.uniform(5, 500),
        'desc': random.choice(['Grocery', 'Gas', 'Restaurant', 'Shopping']),
        'hour': random.randint(8, 20)
    },
    'STRUCTURING': lambda: {
        'amount': random.uniform(7000, 9900),
        'desc': 'Cash deposit below threshold',
        'hour': random.randint(22, 6) % 24
    },
    'LAYERING': lambda: {
        'amount': random.uniform(10000, 500000),
        'desc': 'International wire transfer',
        'hour': random.randint(0, 23)
    },
    'SHELL_COMPANY': lambda: {
        'amount': round(random.choice([10000, 25000, 50000, 100000]), -3),
        'desc': 'Business consulting fee',
        'hour': random.randint(9, 17)
    },
    'ROUND_TRIPPING': lambda: {
        'amount': random.uniform(20000, 1000000),
        'desc': 'Investment return',
        'hour': random.randint(0, 23)
    }
}

# ============================================================================
# DATA GENERATION
# ============================================================================
def generate_data(num_txns=50000, fraud_ratio=0.2):
    """Generate synthetic banking transactions"""
    categories = list(FRAUD_PATTERNS.keys())
    num_legit = int(num_txns * (1 - fraud_ratio))
    num_fraud = num_txns - num_legit
    fraud_per_type = num_fraud // (len(categories) - 1)

    transactions = []
    for i in range(num_txns):
        # Determine category
        if i < num_legit:
            category = 'LEGITIMATE'
        else:
            category = categories[1 + (i - num_legit) // fraud_per_type]

        # Generate transaction
        pattern = FRAUD_PATTERNS[category]()
        txn = {
            'transaction_id': f'TXN_{i:08d}',
            'amount': pattern['amount'],
            'description': pattern['desc'],
            'hour': pattern['hour'],
            'category': category,
            'timestamp': datetime.now() - timedelta(days=random.randint(0, 365))
        }
        transactions.append(txn)

    return pd.DataFrame(transactions)

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
def build_feature_pipeline():
    """Create PySpark ML pipeline for features"""
    return Pipeline(stages=[
        # Text processing
        Tokenizer(inputCol="description", outputCol="words"),
        StopWordsRemover(inputCol="words", outputCol="filtered_words"),
        HashingTF(inputCol="filtered_words", outputCol="tf", numFeatures=50),
        IDF(inputCol="tf", outputCol="text_features"),

        # Categorical encoding
        StringIndexer(inputCol="category", outputCol="label", handleInvalid="keep"),

        # Feature assembly
        VectorAssembler(
            inputCols=["amount", "hour", "amount_log", "is_night",
                      "is_large", "text_features"],
            outputCol="features_raw"
        ),
        StandardScaler(inputCol="features_raw", outputCol="features",
                      withMean=True, withStd=True)
    ])

def engineer_features(df):
    """Add engineered features"""
    return df \
        .withColumn('amount_log', F.log1p('amount')) \
        .withColumn('is_night', (F.col('hour') < 6) | (F.col('hour') > 22)) \
        .withColumn('is_large', F.col('amount') > 10000)

# ============================================================================
# MODEL TRAINING
# ============================================================================
def train_models(train_df, config):
    """Train all ML models"""
    models = {}

    # Random Forest
    models['RandomForest'] = RandomForestClassifier(
        numTrees=config['num_trees'],
        maxDepth=config['max_depth'],
        featuresCol='features', labelCol='label'
    ).fit(train_df)

    # Gradient Boosting
    gbt = GBTClassifier(maxIter=config['max_iter'], maxDepth=5,
                        featuresCol='features', labelCol='label')
    models['GradientBoosting'] = OneVsRest(classifier=gbt).fit(train_df)

    # Logistic Regression
    models['LogisticRegression'] = LogisticRegression(
        maxIter=100, regParam=0.01, family='multinomial',
        featuresCol='features', labelCol='label'
    ).fit(train_df)

    # Neural Network
    num_features = train_df.select('features').first()[0].size
    num_classes = int(train_df.select('label').distinct().count())
    models['NeuralNetwork'] = MultilayerPerceptronClassifier(
        layers=[num_features, 64, 32, num_classes],
        featuresCol='features', labelCol='label'
    ).fit(train_df)

    return models

# ============================================================================
# EVALUATION
# ============================================================================
def evaluate_models(models, test_df):
    """Evaluate all models and return comparison"""
    evaluator = MulticlassClassificationEvaluator(labelCol='label')
    results = []

    for name, model in models.items():
        preds = model.transform(test_df)
        results.append({
            'Model': name,
            'Accuracy': evaluator.evaluate(preds, {evaluator.metricName: "accuracy"}),
            'F1': evaluator.evaluate(preds, {evaluator.metricName: "f1"}),
            'Precision': evaluator.evaluate(preds, {evaluator.metricName: "weightedPrecision"}),
            'Recall': evaluator.evaluate(preds, {evaluator.metricName: "weightedRecall"})
        })

    return pd.DataFrame(results).sort_values('F1', ascending=False)

# ============================================================================
# MAIN PIPELINE
# ============================================================================
def run_pipeline():
    """Execute complete ML pipeline"""
    print("🚀 Starting AML Fraud Detection Pipeline\n")

    # Initialize Spark
    spark = SparkSession.builder \
        .appName("AML_Compact") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    # Generate data
    print("📊 Generating data...")
    df_pandas = generate_data(
        CONFIG['data']['num_transactions'],
        CONFIG['data']['fraud_ratio']
    )
    df = spark.createDataFrame(df_pandas)
    print(f"   ✓ Generated {len(df_pandas):,} transactions\n")

    # Feature engineering
    print("🔧 Engineering features...")
    df = engineer_features(df)
    pipeline = build_feature_pipeline()
    pipeline_model = pipeline.fit(df)
    df_transformed = pipeline_model.transform(df)
    print("   ✓ Created feature pipeline\n")

    # Split data
    print("✂️  Splitting data...")
    train, test = df_transformed.randomSplit([0.8, 0.2], seed=42)
    train.cache()
    test.cache()
    print(f"   ✓ Train: {train.count():,} | Test: {test.count():,}\n")

    # Train models
    print("🤖 Training models...")
    models = train_models(train, CONFIG['models'])
    print(f"   ✓ Trained {len(models)} models\n")

    # Evaluate
    print("📈 Evaluating models...")
    results = evaluate_models(models, test)
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(results.to_string(index=False))
    print("="*60 + "\n")

    # Save results
    os.makedirs(CONFIG['paths']['reports'], exist_ok=True)
    results.to_csv(f"{CONFIG['paths']['reports']}/comparison.csv", index=False)
    print(f"✅ Results saved to {CONFIG['paths']['reports']}/")

    # Save best model
    best_model_name = results.iloc[0]['Model']
    best_model = models[best_model_name]
    model_path = f"{CONFIG['paths']['models']}/{best_model_name}"
    os.makedirs(CONFIG['paths']['models'], exist_ok=True)
    best_model.write().overwrite().save(model_path)
    print(f"✅ Best model ({best_model_name}) saved to {model_path}/")

    spark.stop()
    print("\n🎉 Pipeline complete!")

if __name__ == "__main__":
    run_pipeline()
