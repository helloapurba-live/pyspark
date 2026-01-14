# 🎯 Banking AML Fraud Detection - Interview & Revision Cheat Sheet

> **Complete technical reference for interviews, code reviews, and quick revision**

---

## 📋 **TABLE OF CONTENTS**

1. [System Design Questions](#system-design-questions)
2. [Machine Learning Concepts](#machine-learning-concepts)
3. [PySpark & Big Data](#pyspark--big-data)
4. [Code Patterns & Snippets](#code-patterns--snippets)
5. [Performance Optimization](#performance-optimization)
6. [MLOps & Production](#mlops--production)
7. [Common Interview Questions](#common-interview-questions)
8. [Debugging Scenarios](#debugging-scenarios)

---

## 🏗️ **SYSTEM DESIGN QUESTIONS**

### **Q: Design a fraud detection system for 100M transactions/day**

**Answer Structure:**

```
1. REQUIREMENTS CLARIFICATION
   ├─ Volume: 100M txns/day = 1,157 txns/sec
   ├─ Latency: < 100ms for real-time scoring
   ├─ Availability: 99.99% uptime
   └─ Accuracy: > 90% recall (don't miss fraud)

2. HIGH-LEVEL ARCHITECTURE
   ┌─────────────┐      ┌──────────────┐      ┌─────────────┐
   │   Kafka     │─────▶│ Spark Stream │─────▶│   Model     │
   │  (Ingestion)│      │  (Processing)│      │  (Scoring)  │
   └─────────────┘      └──────────────┘      └─────────────┘
         │                      │                      │
         ▼                      ▼                      ▼
   ┌─────────────┐      ┌──────────────┐      ┌─────────────┐
   │ Data Lake   │      │Feature Store │      │  Cassandra  │
   │  (Archive)  │      │  (Real-time) │      │  (Results)  │
   └─────────────┘      └──────────────┘      └─────────────┘

3. DATA FLOW
   Transaction → Validation → Feature Extraction → Model Scoring → Decision

4. SCALING STRATEGY
   ├─ Horizontal: Kafka partitions (100), Spark executors (50)
   ├─ Caching: Redis for hot features, model in-memory
   ├─ Batching: Micro-batches (1000 txns every 10s)
   └─ Async: Non-blocking predictions, queue for review

5. FAULT TOLERANCE
   ├─ Kafka: Replication factor = 3
   ├─ Spark: Checkpointing every 5 min
   ├─ Model: A/B deployment, canary rollout
   └─ Fallback: Rule-based system if ML fails

6. MONITORING
   ├─ Metrics: Latency (p50, p99), throughput, error rate
   ├─ Alerts: Drift detection, performance degradation
   └─ Dashboards: Grafana for real-time visibility
```

**Key Points to Mention:**
- **Trade-offs:** Latency vs. accuracy, cost vs. performance
- **CAP theorem:** Consistency vs. Availability (choose AP for fraud detection)
- **Data partitioning:** By customer_id or timestamp
- **Hot path vs. Cold path:** Real-time scoring + batch retraining

---

### **Q: How do you handle class imbalance (0.1% fraud)?**

**Answer:**

```python
# PROBLEM: 99.9% legitimate, 0.1% fraud → Model predicts all "legitimate" = 99.9% accuracy!

# SOLUTION 1: Oversampling minority class (SMOTE)
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy=0.3)  # Increase fraud to 30%
X_resampled, y_resampled = smote.fit_resample(X, y)

# SOLUTION 2: Undersampling majority class
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(sampling_strategy=0.5)
X_resampled, y_resampled = rus.fit_resample(X, y)

# SOLUTION 3: Class weights (penalize fraud misclassification)
RandomForestClassifier(class_weight={0: 1.0, 1: 100.0})

# SOLUTION 4: Ensemble with resampling
for i in range(10):
    # Different random undersamples
    X_sample = undersample(X, ratio=0.3)
    models[i] = train(X_sample)
# Final prediction = majority vote

# SOLUTION 5: Anomaly detection (treat fraud as outliers)
from sklearn.ensemble import IsolationForest
iso = IsolationForest(contamination=0.001)

# EVALUATION: Use F1, Precision-Recall, not Accuracy!
```

**Why This Matters:**
- Accuracy is misleading with imbalanced data
- Recall is critical for fraud (missing fraud is expensive)
- Production systems use ensemble + threshold tuning

---

### **Q: Explain your feature engineering pipeline**

**Answer:**

```python
# PHILOSOPHY: Good features > Complex models

# 1. NUMERICAL FEATURES (Handle wide ranges)
amount_log = log1p(amount)  # $1 to $1M → 0 to 14
amount_zscore = (amount - mean) / std  # Standardize

# 2. THRESHOLD FEATURES (Domain knowledge)
near_10k = (7000 <= amount < 10000)  # Structuring red flag
is_round = (amount % 1000 < 10)      # Shell company tell

# 3. TEMPORAL FEATURES (Time patterns)
hour = extract_hour(timestamp)
is_night = (hour < 6) or (hour > 22)
is_weekend = dayofweek in [6, 7]

# 4. TEXT FEATURES (NLP)
# TF-IDF: "Cash deposit" → [0.2, 0.8, 0.1, ...]
# Why TF-IDF? Weighs rare, discriminative words higher
tfidf = TfidfVectorizer(max_features=100)
text_features = tfidf.fit_transform(descriptions)

# 5. AGGREGATE FEATURES (Customer behavior)
customer_avg_amount = transactions.groupby('customer_id').mean()
txn_velocity = count_last_24h(customer_id)

# 6. INTERACTION FEATURES
amount_per_hour = amount / (hour + 1)  # Unusual for large amounts at night
```

**Key Points:**
- **Domain knowledge** embedded in features (e.g., near_10k)
- **Scaling** for numerical stability
- **One-hot encoding** for categorical (location, merchant)
- **Pipeline pattern** ensures train/test consistency

---

## 🤖 **MACHINE LEARNING CONCEPTS**

### **Q: Why use ensemble methods?**

**Answer:**

```
ENSEMBLE THEORY: Wisdom of crowds

Single Model:
- Random Forest alone: 91% accuracy
- Biases: Overfits to tree-based patterns

Ensemble (RF + GBT + NN):
- Combined: 93% accuracy
- Why? Different models capture different patterns:

  Random Forest:  Good at → Robust averaging, feature interactions
  Gradient Boost: Good at → Sequential error correction
  Neural Network: Good at → Non-linear patterns

ENSEMBLE TYPES:

1. BAGGING (Bootstrap Aggregating)
   - Random Forest = Bagging of decision trees
   - Reduces variance (overfitting)
   - Parallel training

2. BOOSTING (Sequential learning)
   - GBT = Learn from previous mistakes
   - Reduces bias (underfitting)
   - Sequential training

3. STACKING (Meta-learning)
   - Train meta-model on predictions
   - Level 1: Base models (RF, GBT, NN)
   - Level 2: Meta-model (Logistic Reg on L1 predictions)

CODE:
from sklearn.ensemble import VotingClassifier
ensemble = VotingClassifier([
    ('rf', RandomForest()),
    ('gbt', GradientBoosting()),
    ('nn', NeuralNetwork())
], voting='soft')  # Soft = average probabilities
```

---

### **Q: Explain bias-variance tradeoff**

**Visual Explanation:**

```
TARGET (Bull's eye)

HIGH BIAS, LOW VARIANCE         LOW BIAS, HIGH VARIANCE
(Underfitting)                  (Overfitting)

    x                              x  x
      x  x                       x      x
    x    x                         x  x
  (Consistent but wrong)         (Scattered around target)

SWEET SPOT: LOW BIAS, LOW VARIANCE
       x
     x   x
       x
   (Tight cluster on target)

IN PRACTICE:

High Bias → Simple model (Logistic Reg, shallow tree)
  - Learns general patterns only
  - Test error ≈ Train error (both high)
  - Fix: Add features, more complex model

High Variance → Complex model (Deep tree, NN)
  - Memorizes training data
  - Test error >> Train error
  - Fix: Regularization, more data, ensemble

REGULARIZATION:
L1 (Lasso):     Penalty = λ Σ|w|     → Feature selection
L2 (Ridge):     Penalty = λ Σw²      → Weight shrinkage
Elastic Net:    α·L1 + (1-α)·L2      → Best of both
```

---

### **Q: How does Random Forest work?**

**Step-by-Step:**

```
RANDOM FOREST = Ensemble of Decision Trees

ALGORITHM:
1. For i = 1 to 100 (num_trees):
   a. Bootstrap: Sample 80% of data WITH replacement
   b. Build tree:
      - At each split, consider random √n features
      - Choose best split (Gini impurity)
      - Grow until max_depth or min_samples
   c. Save tree_i

2. Prediction:
   - Get prediction from each tree
   - Majority vote (classification) or average (regression)

WHY IT WORKS:
- Bootstrap → Each tree sees different data (diversity)
- Random features → Trees make different mistakes (decorrelation)
- Averaging → Errors cancel out, signal reinforces

HYPERPARAMETERS:
num_trees:         More = better (diminishing returns after 100)
max_depth:         Controls complexity (5-20 typical)
min_samples_leaf:  Minimum samples per leaf (prevents overfitting)
max_features:      √n for classification, n/3 for regression

PROS:
✅ Robust (handles noise, outliers)
✅ Feature importance (built-in)
✅ Parallel training
✅ Works out-of-the-box

CONS:
❌ Large memory footprint
❌ Slow inference (100 trees)
❌ Less interpretable than single tree
```

---

### **Q: Neural Network vs. Traditional ML?**

**Comparison:**

```
┌────────────────────┬─────────────────┬──────────────────┐
│ Aspect             │ Neural Network  │ Random Forest    │
├────────────────────┼─────────────────┼──────────────────┤
│ Data Required      │ 100K+ samples   │ 1K+ samples      │
│ Training Time      │ Hours           │ Minutes          │
│ Inference Speed    │ Fast (GPU)      │ Moderate         │
│ Interpretability   │ Black box       │ Feature import.  │
│ Feature Engineering│ Auto (learns)   │ Manual (better)  │
│ Overfitting Risk   │ High            │ Medium           │
│ Hyperparameters    │ Many (complex)  │ Few (simple)     │
│ Best For           │ Images, text    │ Tabular data     │
└────────────────────┴─────────────────┴──────────────────┘

WHEN TO USE:

Neural Networks:
✅ Large dataset (> 100K samples)
✅ Complex patterns (non-linear interactions)
✅ Images, text, sequences
✅ GPU available

Random Forest:
✅ Small-medium dataset (< 100K)
✅ Tabular data with mixed types
✅ Need interpretability
✅ CPU-only environment

FOR THIS PROJECT:
- Random Forest wins (92.3% vs 90.9%)
- Tabular + text data
- 50K samples (not huge)
- Feature engineering gives RF advantage
```

---

## 💻 **PYSPARK & BIG DATA**

### **Q: Explain PySpark architecture**

**Answer:**

```
SPARK ARCHITECTURE:

┌─────────────────────────────────────────────────────┐
│                    DRIVER                           │
│  ┌─────────────┐                                   │
│  │ SparkContext│  (Orchestrates execution)         │
│  └─────────────┘                                   │
└─────────────────┬───────────────────────────────────┘
                  │
      ┌───────────┼───────────┐
      ▼           ▼           ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│ EXECUTOR │ │ EXECUTOR │ │ EXECUTOR │
│  ┌────┐  │ │  ┌────┐  │ │  ┌────┐  │
│  │Task│  │ │  │Task│  │ │  │Task│  │
│  └────┘  │ │  └────┘  │ │  └────┘  │
│ Cache    │ │ Cache    │ │ Cache    │
└──────────┘ └──────────┘ └──────────┘

KEY CONCEPTS:

1. RDD (Resilient Distributed Dataset)
   - Immutable, partitioned collection
   - Lazy evaluation (transformations)
   - Fault-tolerant (lineage)

2. DataFrame (Structured API)
   - Schema-aware (like SQL table)
   - Optimized query planner
   - Catalyst optimizer

3. TRANSFORMATIONS (Lazy)
   - map, filter, groupBy
   - Creates execution plan
   - No computation until action

4. ACTIONS (Eager)
   - count, collect, save
   - Triggers execution
   - Returns results

EXAMPLE:
df = spark.read.parquet("data.parquet")  # Not executed
df_filtered = df.filter("amount > 1000") # Not executed (lazy)
count = df_filtered.count()              # Executed! (action)
```

---

### **Q: How to optimize PySpark performance?**

**Answer:**

```python
# 1. AVOID COLLECT() (Brings all data to driver)
# BAD:
df_pandas = df.collect()  # OOM if data > driver memory

# GOOD:
df.write.parquet("output")  # Distributed write

# 2. CACHE STRATEGICALLY
# Cache data you'll use multiple times
df_cached = df.filter("amount > 1000").cache()
df_cached.count()  # First computation, stores in memory
df_cached.groupBy(...).count()  # Instant (from cache)

# 3. REPARTITION FOR PARALLELISM
# Too few partitions → Underutilized cluster
# Too many partitions → Overhead
df = df.repartition(200)  # Rule: 2-4x num_cores

# 4. AVOID UDFs (User-Defined Functions)
# BAD (Python UDF, slow):
@udf(returnType=IntegerType())
def is_large(amount):
    return 1 if amount > 10000 else 0

# GOOD (Native Spark function, 100x faster):
df.withColumn('is_large', (F.col('amount') > 10000).cast('int'))

# 5. PREDICATE PUSHDOWN
# Filter early, before expensive operations
df = spark.read.parquet("data")  \
    .filter("date = '2024-03-15'")  # Reads only 1 day, not all data

# 6. BROADCAST SMALL TABLES
# For joins with small lookup tables
from pyspark.sql.functions import broadcast
df.join(broadcast(small_df), "key")  # Sends small_df to all workers

# 7. PERSIST INTERMEDIATE RESULTS
df_intermediate = df.groupBy(...).agg(...)
df_intermediate.persist(StorageLevel.MEMORY_AND_DISK)
# Use df_intermediate multiple times without recomputation

# 8. AVOID SHUFFLES
# Shuffle = Expensive network transfer
# Bad: df.repartition(1000)  # Random shuffle
# Good: df.coalesce(1000)    # Only if reducing partitions

# 9. COLUMNAR STORAGE (Parquet)
# Parquet: Column-oriented, compressed, schema-aware
df.write.parquet("output.parquet")  # 10x smaller than CSV
```

---

### **Q: Explain partitioning in Spark**

**Answer:**

```
PARTITIONING = How data is distributed across cluster

┌─────────────────────────────────────────────────┐
│              DataFrame (1M rows)                │
└─────────────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
  ┌─────────┐  ┌─────────┐  ┌─────────┐
  │Part 1   │  │Part 2   │  │Part 3   │
  │300K rows│  │400K rows│  │300K rows│
  └─────────┘  └─────────┘  └─────────┘
     Node 1      Node 2       Node 3

TYPES:

1. HASH PARTITIONING (repartition)
   - Distributes by hash(key) % num_partitions
   - Random distribution
   - Use: When data has no natural partition key

2. RANGE PARTITIONING (repartitionByRange)
   - Distributes by key ranges
   - Sorted partitions
   - Use: When you need sorted data

3. CUSTOM PARTITIONING
   - Define your own partitioner
   - Example: By customer_id for customer-centric queries

BEST PRACTICES:

# Rule 1: 2-4x num_cores
spark.conf.set("spark.sql.shuffle.partitions", "200")

# Rule 2: Partition size 128MB - 1GB
# Too small → Overhead
# Too large → Memory issues

# Rule 3: Partition by query patterns
df.write.partitionBy("date", "country").parquet("output")
# Enables partition pruning: Only read needed partitions

EXAMPLE:
# Before: 1000 partitions, 10MB each (too many!)
df.coalesce(100)  # Reduce to 100 partitions, 100MB each

# After joins (creates many small partitions):
df_joined.repartition(200)  # Rebalance
```

---

## 💡 **CODE PATTERNS & SNIPPETS**

### **Pattern: Feature Engineering Pipeline**

```python
# REUSABLE PATTERN: PySpark ML Pipeline

from pyspark.ml import Pipeline
from pyspark.ml.feature import *

def build_feature_pipeline(input_cols, label_col):
    """
    Universal feature pipeline
    Handles: Numerical, Categorical, Text
    """
    stages = []

    # 1. Handle missing values
    imputer = Imputer(
        inputCols=numerical_cols,
        outputCols=[f"{c}_imputed" for c in numerical_cols],
        strategy='median'
    )
    stages.append(imputer)

    # 2. Encode categoricals
    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in categorical_cols
    ]
    stages.extend(indexers)

    encoders = [
        OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_vec")
        for c in categorical_cols
    ]
    stages.extend(encoders)

    # 3. Process text
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    tf = HashingTF(inputCol="filtered", outputCol="tf", numFeatures=100)
    idf = IDF(inputCol="tf", outputCol="text_features")
    stages.extend([tokenizer, remover, tf, idf])

    # 4. Assemble all features
    assembler = VectorAssembler(
        inputCols=all_feature_cols,
        outputCol="features_raw"
    )
    stages.append(assembler)

    # 5. Scale
    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features",
        withMean=True,
        withStd=True
    )
    stages.append(scaler)

    # 6. Encode label
    label_indexer = StringIndexer(inputCol=label_col, outputCol="label")
    stages.append(label_indexer)

    return Pipeline(stages=stages)

# USAGE:
pipeline = build_feature_pipeline(input_cols, 'category')
model = pipeline.fit(train_df)
transformed = model.transform(test_df)

# KEY BENEFIT: Same transformations guaranteed for train/test/production
```

---

### **Pattern: Model Training Loop**

```python
# REUSABLE PATTERN: Train multiple models with tracking

def train_and_evaluate_models(train_df, test_df):
    """
    Train all models, track with MLflow, return best
    """
    import mlflow
    from sklearn.metrics import classification_report

    models_config = {
        'RandomForest': {
            'class': RandomForestClassifier,
            'params': {'numTrees': 100, 'maxDepth': 10}
        },
        'GBT': {
            'class': GBTClassifier,
            'params': {'maxIter': 50, 'maxDepth': 5}
        },
        'NeuralNet': {
            'class': MultilayerPerceptronClassifier,
            'params': {'layers': [50, 128, 64, 5], 'maxIter': 100}
        }
    }

    results = []

    for name, config in models_config.items():
        with mlflow.start_run(run_name=name):
            print(f"Training {name}...")

            # 1. Train
            model = config['class'](**config['params']).fit(train_df)

            # 2. Predict
            predictions = model.transform(test_df)

            # 3. Evaluate
            evaluator = MulticlassClassificationEvaluator()
            accuracy = evaluator.evaluate(predictions, {evaluator.metricName: 'accuracy'})
            f1 = evaluator.evaluate(predictions, {evaluator.metricName: 'f1'})

            # 4. Log to MLflow
            mlflow.log_params(config['params'])
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_score", f1)
            mlflow.spark.log_model(model, "model")

            # 5. Store results
            results.append({
                'name': name,
                'model': model,
                'accuracy': accuracy,
                'f1': f1
            })

            print(f"  ✓ {name}: Acc={accuracy:.3f}, F1={f1:.3f}")

    # Return best model by F1
    best = max(results, key=lambda x: x['f1'])
    print(f"\n🏆 Best model: {best['name']} (F1={best['f1']:.3f})")

    return best['model'], results

# USAGE:
best_model, all_results = train_and_evaluate_models(train, test)
```

---

### **Pattern: Data Validation**

```python
# REUSABLE PATTERN: Validate data quality

class DataValidator:
    """Validate data before training/inference"""

    @staticmethod
    def validate_schema(df, expected_schema):
        """Check if DataFrame has expected columns and types"""
        actual_cols = set(df.columns)
        expected_cols = set(expected_schema.keys())

        # Missing columns
        missing = expected_cols - actual_cols
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Type check
        for col, expected_type in expected_schema.items():
            actual_type = dict(df.dtypes)[col]
            if actual_type != expected_type:
                raise TypeError(f"{col}: Expected {expected_type}, got {actual_type}")

        return True

    @staticmethod
    def check_nulls(df, threshold=0.1):
        """Ensure null rate < threshold"""
        from pyspark.sql import functions as F

        null_counts = df.select([
            (F.sum(F.col(c).isNull().cast('int')) / F.count('*')).alias(c)
            for c in df.columns
        ]).first().asDict()

        violations = {k: v for k, v in null_counts.items() if v > threshold}
        if violations:
            raise ValueError(f"High null rate: {violations}")

        return True

    @staticmethod
    def check_ranges(df, ranges):
        """Validate numerical ranges"""
        from pyspark.sql import functions as F

        for col, (min_val, max_val) in ranges.items():
            stats = df.select(F.min(col), F.max(col)).first()
            if stats[0] < min_val or stats[1] > max_val:
                raise ValueError(f"{col} out of range: {stats}")

        return True

# USAGE:
validator = DataValidator()
validator.validate_schema(df, {
    'amount': 'double',
    'description': 'string',
    'timestamp': 'timestamp'
})
validator.check_nulls(df, threshold=0.05)
validator.check_ranges(df, {
    'amount': (0, 10_000_000)
})
```

---

## ⚡ **PERFORMANCE OPTIMIZATION**

### **Q: How to handle large datasets (> 10GB)?**

**Answer:**

```python
# STRATEGY 1: Partition and process in chunks
df = spark.read.parquet("huge_data.parquet")
df = df.repartition(200)  # Distribute across 200 partitions

# Process each partition independently
def process_partition(iterator):
    for batch in iterator:
        # Process batch (1/200th of data)
        yield batch

df.rdd.mapPartitions(process_partition)

# STRATEGY 2: Sampling for prototyping
df_sample = df.sample(0.01)  # 1% sample for fast iteration
# Develop on sample, test on full data

# STRATEGY 3: Incremental processing
for date in dates:
    df_day = spark.read.parquet(f"data/date={date}")
    process(df_day)
    df_day.write.parquet(f"output/date={date}")

# STRATEGY 4: Columnar storage (Parquet)
# Only read needed columns
df = spark.read.parquet("data.parquet").select('amount', 'category')
# Not: df = spark.read.parquet("data.parquet")  # Reads all columns

# STRATEGY 5: Predicate pushdown
df = spark.read.parquet("data.parquet") \
    .filter("date >= '2024-01-01'")  # Filter during read, not after

# STRATEGY 6: Broadcast small tables
# For joins with dimension tables
df_large.join(broadcast(df_small), "key")

# STRATEGY 7: Persist intermediate results
df_expensive = df.groupBy(...).agg(...)  # Expensive operation
df_expensive.persist()  # Cache result
df_expensive.count()  # Materialize
# Now df_expensive is in memory for fast reuse
```

---

### **Q: Memory optimization techniques?**

**Answer:**

```python
# 1. CLEAR CACHE WHEN DONE
df.unpersist()  # Free memory

# 2. USE APPROPRIATE STORAGE LEVEL
from pyspark import StorageLevel

df.persist(StorageLevel.MEMORY_ONLY)           # Fast, but OOM risk
df.persist(StorageLevel.MEMORY_AND_DISK)       # Spill to disk
df.persist(StorageLevel.MEMORY_AND_DISK_SER)   # Serialized (smaller)
df.persist(StorageLevel.DISK_ONLY)             # Slow, but no OOM

# 3. REDUCE SHUFFLE SIZE
# Before shuffle: 10GB
# After: 1GB
df.groupBy('key').agg(F.sum('amount'))  # Combiner reduces data before shuffle

# 4. AVOID WIDE TRANSFORMATIONS
# Narrow transformation (no shuffle): filter, map, select
# Wide transformation (shuffle): groupBy, join, repartition
# Minimize wide transformations

# 5. DATA TYPES
# Use smallest appropriate type
# Bad:  df.withColumn('flag', F.lit(1).cast('long'))   # 8 bytes
# Good: df.withColumn('flag', F.lit(1).cast('byte'))   # 1 byte

# 6. KRYO SERIALIZATION (faster, smaller)
spark.conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

# 7. DYNAMIC ALLOCATION
spark.conf.set("spark.dynamicAllocation.enabled", "true")
# Auto-scale executors based on workload
```

---

## 🚀 **MLOPS & PRODUCTION**

### **Q: How do you monitor ML models in production?**

**Answer:**

```python
# MONITORING FRAMEWORK

class ModelMonitor:
    """Monitor model performance and data drift"""

    def __init__(self, model, reference_data):
        self.model = model
        self.reference_stats = self.compute_stats(reference_data)

    def compute_stats(self, df):
        """Compute statistical summary"""
        return {
            'amount_mean': df.select(F.mean('amount')).first()[0],
            'amount_std': df.select(F.stddev('amount')).first()[0],
            'fraud_rate': df.filter('is_fraud = 1').count() / df.count()
        }

    def check_data_drift(self, production_data):
        """Detect if data distribution has changed"""
        prod_stats = self.compute_stats(production_data)

        drift = {}
        for key in self.reference_stats:
            ref_val = self.reference_stats[key]
            prod_val = prod_stats[key]
            pct_change = abs(prod_val - ref_val) / ref_val

            if pct_change > 0.2:  # 20% threshold
                drift[key] = {
                    'reference': ref_val,
                    'production': prod_val,
                    'change': pct_change
                }

        if drift:
            self.alert(f"⚠️  Data drift detected: {drift}")

        return drift

    def check_model_performance(self, production_data, labels):
        """Monitor prediction accuracy over time"""
        predictions = self.model.transform(production_data)

        # Calculate metrics
        evaluator = MulticlassClassificationEvaluator()
        accuracy = evaluator.evaluate(predictions)

        # Compare to baseline
        if accuracy < self.baseline_accuracy - 0.05:  # 5% drop
            self.alert(f"⚠️  Performance degradation: {accuracy:.3f}")
            self.trigger_retraining()

        # Log to monitoring system
        self.log_metric('accuracy', accuracy, timestamp=datetime.now())

    def alert(self, message):
        """Send alert to monitoring system"""
        # Integrate with: PagerDuty, Slack, Email
        print(f"ALERT: {message}")

    def trigger_retraining(self):
        """Initiate automated retraining"""
        # Trigger Airflow DAG or CI/CD pipeline
        print("🔄 Triggering model retraining...")

# USAGE:
monitor = ModelMonitor(model, reference_data)

# Scheduled job (daily)
def daily_monitoring():
    today_data = load_production_data(date='2024-03-15')
    monitor.check_data_drift(today_data)
    monitor.check_model_performance(today_data, labels)
```

**Metrics to Monitor:**
1. **Model Performance:** Accuracy, Precision, Recall (daily)
2. **Data Drift:** Feature distributions (weekly)
3. **Prediction Drift:** Output distribution (daily)
4. **Latency:** p50, p95, p99 (real-time)
5. **Error Rate:** 4xx, 5xx errors (real-time)
6. **Volume:** Requests per second (real-time)

---

### **Q: Explain your deployment strategy**

**Answer:**

```
DEPLOYMENT PIPELINE:

1. DEVELOPMENT
   ├─ Jupyter notebooks (experimentation)
   ├─ Git version control
   └─ MLflow tracking

2. STAGING
   ├─ Containerize (Docker)
   ├─ Integration tests
   └─ Load testing

3. PRODUCTION
   ├─ Canary deployment (5% traffic)
   ├─ Monitor metrics (30 min)
   ├─ Gradual rollout (5% → 50% → 100%)
   └─ Rollback plan (instant)

DEPLOYMENT PATTERNS:

A. BLUE-GREEN DEPLOYMENT
   ┌──────────┐     ┌──────────┐
   │  Blue    │     │  Green   │
   │ (Old v1) │     │ (New v2) │
   └──────────┘     └──────────┘
         ▲                ▲
         │                │
    90% traffic      10% traffic (canary)

   If v2 OK → Switch to 100% Green
   If v2 fails → Rollback to 100% Blue

B. SHADOW MODE
   Production traffic → Both models
   - Old model: Serves responses
   - New model: Logs predictions (comparison)
   - Validate new model without risk

C. A/B TESTING
   50% users → Model A
   50% users → Model B
   - Measure business metrics (fraud caught, false positives)
   - Choose winner after statistical significance

INFRASTRUCTURE:

# Kubernetes deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fraud-detection-api
spec:
  replicas: 10  # Horizontal scaling
  template:
    spec:
      containers:
      - name: api
        image: fraud-detector:v1.2.3
        resources:
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:  # Health check
          httpGet:
            path: /health
            port: 8000
        readinessProbe:  # Ready for traffic
          httpGet:
            path: /ready
            port: 8000
```

---

## 🎤 **COMMON INTERVIEW QUESTIONS**

### **Q: Walk me through your project**

**STAR Method Answer:**

```
SITUATION:
"Banks lose $2 trillion annually to money laundering. Traditional rule-based
systems have high false positive rates (90%) and miss new fraud patterns."

TASK:
"Build an ML system to detect 5 types of AML fraud with 90%+ accuracy and
< 100ms latency for real-time transaction scoring."

ACTION:
1. Data: Generated 50K synthetic transactions with realistic fraud patterns
   - Used domain knowledge (e.g., structuring just below $10k threshold)

2. Features: Engineered 50+ features from tabular + text data
   - Numerical: log(amount), threshold flags
   - Text: TF-IDF on transaction descriptions
   - Temporal: hour, night/weekend flags

3. Models: Trained 10+ algorithms with PySpark ML
   - Random Forest (best: 92.3% accuracy)
   - Gradient Boosting (91.6%)
   - Neural Network (90.9%)

4. MLOps: Built complete pipeline
   - MLflow for experiment tracking
   - Docker for deployment
   - API for real-time scoring (45ms latency)
   - Monitoring for drift detection

RESULT:
- Achieved 92.3% accuracy, 91.9% F1-score
- Detects fraud with 90.5% recall (catches most fraud)
- 91.2% precision (few false alarms)
- Production-ready with < 100ms latency
- Estimated $4M+ annual savings for a mid-sized bank
```

---

### **Q: What was your biggest challenge?**

**Answer:**

```
CHALLENGE: Class Imbalance

PROBLEM:
- Real-world fraud: 0.1% of transactions
- Naive model: Predicts everything "legitimate" → 99.9% accuracy but useless!

SOLUTION ATTEMPT 1: SMOTE (Synthetic Minority Oversampling)
- Oversampled fraud to 30%
- Result: Improved recall but high false positives
- Issue: Synthetic samples not diverse enough

SOLUTION ATTEMPT 2: Class Weights
- Penalized fraud misclassification 100x
- Result: Better recall (95%) but poor precision (60%)
- Issue: Too aggressive, flagged too many legitimate transactions

FINAL SOLUTION: Ensemble + Threshold Tuning
1. Train multiple models on different resampled datasets
2. Ensemble predictions (weighted voting)
3. Tune decision threshold using Precision-Recall curve
   - Standard: threshold = 0.5
   - Optimized: threshold = 0.35 (prioritize recall)
4. Add business rules (e.g., auto-approve if amount < $100)

RESULT:
- Recall: 90.5% (catches most fraud)
- Precision: 91.2% (acceptable false positive rate)
- Balance achieved through ensemble + threshold tuning

LESSON LEARNED:
- No single technique solves imbalance perfectly
- Combine multiple strategies
- Optimize for business metrics, not just accuracy
```

---

### **Q: How would you improve this system?**

**Answer:**

```
IMPROVEMENTS (Priority Order):

1. GRAPH ANALYTICS (High Impact)
   - Current: Analyzes transactions independently
   - Improvement: Build transaction network graph
   - Benefit: Detect fraud rings (multiple related accounts)
   - Implementation:
     * Use GraphFrames (Spark)
     * PageRank to find central nodes (money mules)
     * Community detection (fraud networks)

2. ACTIVE LEARNING (Medium Impact)
   - Current: Static model, retrains monthly
   - Improvement: Learn from analyst feedback
   - Benefit: Continuously improve on edge cases
   - Implementation:
     * Identify high-uncertainty predictions
     * Human review → Labels
     * Retrain on new labels

3. SHAP EXPLAINABILITY (High Value)
   - Current: Black box predictions
   - Improvement: Explain why transaction flagged
   - Benefit: Regulatory compliance, analyst trust
   - Implementation:
     * SHAP values per prediction
     * "Flagged because: amount near $10k (0.42), late night (0.18)"

4. STREAMING ARCHITECTURE (Scalability)
   - Current: Batch processing (nightly)
   - Improvement: Real-time streaming
   - Benefit: Detect fraud as it happens
   - Implementation:
     * Kafka → Spark Streaming → Redis
     * Micro-batches (10s window)

5. FEDERATED LEARNING (Privacy)
   - Current: Centralized data
   - Improvement: Train on distributed data
   - Benefit: Privacy-preserving (GDPR compliant)
   - Implementation:
     * Train locally at each bank
     * Aggregate model updates
     * Never share raw data

QUICK WINS:
- Add more text features (sentiment analysis)
- Implement automated hyperparameter tuning (Hyperopt)
- A/B test different models in production
- Add customer risk profiles (historical behavior)
```

---

## 🐛 **DEBUGGING SCENARIOS**

### **Scenario 1: Model accuracy drops in production**

```
SYMPTOMS:
- Training: 92% accuracy
- Production (week 1): 91% accuracy
- Production (week 4): 78% accuracy ⚠️

DEBUGGING STEPS:

1. CHECK DATA DRIFT
   ```python
   # Compare distributions
   prod_stats = production_data.describe()
   train_stats = training_data.describe()

   # Statistical test
   from scipy.stats import ks_2samp
   stat, pval = ks_2samp(train_amounts, prod_amounts)
   if pval < 0.05:
       print("⚠️  Significant drift detected!")
   ```

2. CHECK LABEL DISTRIBUTION
   ```python
   prod_fraud_rate = production_data.filter('is_fraud = 1').count() / total
   if abs(prod_fraud_rate - train_fraud_rate) > 0.05:
       print("⚠️  Fraud rate changed!")
   ```

3. CHECK FEATURE QUALITY
   ```python
   # Missing values
   null_rate = production_data.select([
       F.sum(F.col(c).isNull().cast('int')) / F.count('*')
       for c in production_data.columns
   ])

   # Out-of-range values
   production_data.select(F.min('amount'), F.max('amount')).show()
   ```

4. CHECK MODEL STALENESS
   ```python
   # How old is the model?
   days_since_training = (datetime.now() - model_train_date).days
   if days_since_training > 30:
       print("⚠️  Model may be stale. Consider retraining.")
   ```

SOLUTIONS:
- Data drift → Retrain on recent data
- Label shift → Adjust decision threshold
- Feature issues → Fix data pipeline
- Model staleness → Scheduled retraining (weekly)
```

---

### **Scenario 2: High latency in production (> 1 second)**

```
SYMPTOMS:
- Expected: < 100ms
- Actual: 1-5 seconds

DEBUGGING STEPS:

1. PROFILE THE CODE
   ```python
   import time

   start = time.time()
   # Feature engineering
   t1 = time.time()
   print(f"Features: {t1 - start:.3f}s")

   # Model prediction
   t2 = time.time()
   print(f"Prediction: {t2 - t1:.3f}s")

   # Post-processing
   t3 = time.time()
   print(f"Post-process: {t3 - t2:.3f}s")
   ```

2. COMMON BOTTLENECKS

   a. Loading model on every request
      ```python
      # BAD: Load model per request
      def predict(data):
          model = load_model("path")  # Slow!
          return model.transform(data)

      # GOOD: Load once at startup
      MODEL = load_model("path")  # Global
      def predict(data):
          return MODEL.transform(data)
      ```

   b. Not using batch prediction
      ```python
      # BAD: One at a time
      for txn in transactions:
          predict(txn)  # 1000 requests = 1000 model calls

      # GOOD: Batch prediction
      predict(transactions)  # 1 request = 1 model call
      ```

   c. Expensive feature engineering
      ```python
      # BAD: Complex UDF
      @udf(returnType=FloatType())
      def compute_feature(x):
          # Complex calculation
          return result

      # GOOD: Precompute or use native Spark functions
      df.withColumn('feature', F.expr("simple_calculation"))
      ```

SOLUTIONS:
- Cache model in memory (singleton pattern)
- Use batch prediction API
- Optimize feature engineering (avoid UDFs)
- Add caching layer (Redis) for hot features
- Horizontal scaling (more API instances)
```

---

## 📝 **QUICK REFERENCE TABLES**

### **Algorithm Selection Guide**

| Use Case | Best Algorithm | Why |
|----------|---------------|-----|
| Real-time (< 10ms) | Logistic Regression | Simplest, fastest |
| Best accuracy | Random Forest | Robust, handles mixed data |
| Interpretability | Decision Tree | Visual, explainable |
| Large text data | Naive Bayes | Efficient with high-dim text |
| Complex patterns | Neural Network | Learns non-linear interactions |
| Imbalanced classes | GBT + Ensemble | Sequential error correction |

### **Hyperparameter Tuning Guide**

| Parameter | Typical Range | Effect | Tuning Tip |
|-----------|--------------|--------|-----------|
| `numTrees` (RF) | 50-200 | More = better (diminishing returns) | Start 100 |
| `maxDepth` (RF/GBT) | 5-20 | Deeper = more complex | Try [5, 10, 15] |
| `learningRate` (GBT) | 0.01-0.3 | Lower = slower but better | Start 0.1 |
| `layers` (NN) | [n, 128, 64, k] | More neurons = more capacity | Halve each layer |
| `regParam` (LogReg) | 0.001-0.1 | Higher = more regularization | Log scale search |

### **Metrics Interpretation**

| Metric | Formula | When to Use | Target |
|--------|---------|-------------|--------|
| Accuracy | (TP+TN) / Total | Balanced classes | > 90% |
| Precision | TP / (TP+FP) | Minimize false alarms | > 85% |
| Recall | TP / (TP+FN) | Catch all fraud (critical!) | > 90% |
| F1-Score | 2×(P×R)/(P+R) | Balance P and R | > 88% |
| AUC-ROC | Area under curve | Overall discrimination | > 0.95 |

---

## 🎓 **FINAL TIPS FOR INTERVIEWS**

### **Communication Strategy**

```
1. STRUCTURE YOUR ANSWER
   ├─ State the problem clearly
   ├─ Explain your approach
   ├─ Discuss trade-offs
   └─ Mention results/impact

2. USE EXAMPLES
   ├─ "For instance, in fraud detection..."
   ├─ Concrete numbers (92% accuracy, $4M savings)
   └─ Visual explanations (draw diagrams)

3. SHOW DEPTH
   ├─ Start high-level
   ├─ Dive deep when asked
   ├─ Mention alternatives considered
   └─ Discuss failure cases

4. BUSINESS AWARENESS
   ├─ "This improves recall from 70% to 90%"
   ├─ "Which means catching 20% more fraud"
   ├─ "Estimated $4M additional savings"
   └─ Connect technical to business value
```

### **What Interviewers Look For**

```
✅ TECHNICAL DEPTH
   - Can you explain concepts clearly?
   - Do you understand trade-offs?
   - Can you code on the spot?

✅ PROBLEM SOLVING
   - How do you approach new problems?
   - Can you break down complex systems?
   - Do you consider edge cases?

✅ PRACTICAL EXPERIENCE
   - Have you deployed models to production?
   - Do you understand MLOps?
   - Can you debug issues?

✅ COMMUNICATION
   - Can you explain to non-technical stakeholders?
   - Do you ask clarifying questions?
   - Can you work with others?
```

---

## 🚀 **CLOSING THOUGHTS**

**Key Takeaways:**

1. **Master the fundamentals** - Understand WHY, not just HOW
2. **Practice coding** - Write clean, efficient, production-ready code
3. **Think end-to-end** - From data to deployment
4. **Business impact** - Connect technical work to business value
5. **Stay curious** - ML evolves fast, keep learning

**Resources:**
- This project: Complete reference implementation
- Practice: Kaggle competitions
- Theory: "Hands-On Machine Learning" (Géron)
- Production: "Designing Data-Intensive Applications" (Kleppmann)

---

**Good luck with your interviews! 🍀**

*Remember: Confidence comes from preparation. You've built a complete ML system - you've got this!* 💪
