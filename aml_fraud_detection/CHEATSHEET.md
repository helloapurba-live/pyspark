# 🎯 AML Fraud Detection - Interview & Revision Cheat Sheet

## 📝 TABLE OF CONTENTS

1. [Core Concepts](#core-concepts)
2. [Technical Implementation](#technical-implementation)
3. [PySpark Essentials](#pyspark-essentials)
4. [Machine Learning Algorithms](#machine-learning-algorithms)
5. [Feature Engineering](#feature-engineering)
6. [Model Evaluation](#model-evaluation)
7. [MLOps & Production](#mlops--production)
8. [Interview Questions & Answers](#interview-questions--answers)
9. [Code Snippets](#code-snippets)
10. [Common Pitfalls](#common-pitfalls)

---

## 🎯 CORE CONCEPTS

### **What is AML (Anti-Money Laundering)?**

**Definition:** Process of detecting and preventing criminals from disguising illegally obtained money as legitimate income.

**Three Stages of Money Laundering:**

1. **Placement** - Introducing illegal money into financial system
   - Example: Depositing drug money into bank account

2. **Layering** - Moving money through complex transactions to hide origin
   - Example: Transferring through multiple accounts/countries

3. **Integration** - Making dirty money appear legitimate
   - Example: Buying real estate with laundered funds

### **Why Machine Learning for AML?**

| Traditional Approach | ML Approach |
|---------------------|-------------|
| Rule-based (if-then) | Pattern recognition |
| Misses new patterns | Adapts to new fraud |
| 60% false positives | 10-20% false positives |
| Manual updates | Auto-learning |
| Rigid thresholds | Probabilistic scores |

### **Why Clustering (Unsupervised) vs Classification (Supervised)?**

**Problem with Classification:**
- Requires labeled data (fraud/not fraud)
- Fraudsters constantly evolve → old labels become stale
- Class imbalance (1-5% fraud rate)

**Clustering Advantage:**
- No labels needed
- Finds "normal" patterns automatically
- Anything far from normal = suspicious
- Catches **novel/unknown** fraud patterns

---

## ⚙️ TECHNICAL IMPLEMENTATION

### **System Architecture Overview**

```
INPUT → FEATURE ENG → ALGORITHMS → ENSEMBLE → OUTPUT
 ↓         ↓             ↓            ↓         ↓
JSON    50 features   12 models   Voting   Fraud Score
```

### **Tech Stack Justification**

| Technology | Why Chosen |
|-----------|------------|
| **PySpark** | Distributed processing, scales to billions of rows |
| **MLlib** | Native Spark ML library, no data movement overhead |
| **TF-IDF** | Fast text vectorization, 100x faster than BERT |
| **YAML** | Configuration as code, change params without redeployment |
| **FastAPI** | Async API, handles 10K req/sec |
| **Docker** | Consistent deployment across environments |

### **Data Pipeline**

```python
# Input Schema
{
    "transaction_id": str,
    "customer_id": str,
    "amount": float,
    "timestamp": datetime,
    "description": str,
    "type": str,
    "channel": str
}

# Output Schema
{
    "transaction_id": str,
    "fraud_score": float,      # 0.0 - 1.0
    "is_suspicious": bool,
    "risk_level": str,         # LOW/MEDIUM/HIGH/CRITICAL
    "signals": List[str],      # ["structuring", "unusual_timing"]
    "cluster_id": int,
    "confidence": float
}
```

---

## 🔥 PYSPARK ESSENTIALS

### **Key PySpark Operations**

```python
# 1. WINDOW FUNCTIONS (Most Common in Fraud Detection)
from pyspark.sql.window import Window

# Rolling 24-hour window
window_24h = (Window
              .partitionBy("customer_id")          # Group by customer
              .orderBy(unix_timestamp("timestamp")) # Sort by time
              .rangeBetween(-86400, 0))            # 24h in seconds

df = df.withColumn("txn_count_24h", count("*").over(window_24h))

# 2. AGGREGATIONS
customer_stats = df.groupBy("customer_id").agg(
    avg("amount").alias("avg_amount"),
    stddev("amount").alias("std_amount"),
    count("*").alias("total_txns")
)

# 3. JOINS (Broadcast for small tables)
from pyspark.sql.functions import broadcast
df.join(broadcast(small_table), "key")  # Broadcast to all nodes

# 4. WHEN/OTHERWISE (SQL CASE statement)
df = df.withColumn("risk",
    when(col("amount") > 10000, "HIGH")
    .when(col("amount") > 5000, "MEDIUM")
    .otherwise("LOW")
)

# 5. UDFs (User Defined Functions)
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType

@udf(returnType=DoubleType())
def calculate_score(amount, avg):
    return float(amount / avg) if avg > 0 else 0.0

df = df.withColumn("ratio", calculate_score("amount", "customer_avg"))
```

### **Performance Optimization Tricks**

```python
# 1. CACHE wisely (if used 2+ times)
df.cache()  # Keep in memory
# ... use df multiple times ...
df.unpersist()  # Free memory

# 2. REPARTITION before heavy operations
df = df.repartition(200, "customer_id")  # Distribute by key

# 3. BROADCAST small tables (<10MB)
df.join(broadcast(lookup_table), "key")

# 4. AVOID collect() on large DataFrames
df.collect()  # ❌ Pulls all data to driver (OOM!)
df.take(10)   # ✅ Only 10 rows

# 5. USE .persist() for iterative algorithms
df.persist(StorageLevel.MEMORY_AND_DISK)  # Spill to disk if OOM
```

### **Common Pitfalls & Solutions**

| Pitfall | Why Bad | Solution |
|---------|---------|----------|
| `df.collect()` on large data | OOM error | Use `.take(n)` or `.show()` |
| Multiple `groupBy` without cache | Recompute each time | Cache between aggregations |
| No partitioning before join | Huge shuffle | `.repartition(key)` first |
| Too many small files | Metadata overhead | `.coalesce(1)` before write |
| String columns in ML | Algorithms need numbers | Use `StringIndexer` + `OneHotEncoder` |

---

## 🤖 MACHINE LEARNING ALGORITHMS

### **Algorithm Selection Matrix**

| Fraud Pattern | Best Algorithm | Why |
|--------------|----------------|-----|
| **Structuring** | Rule-Based + K-Means | Clear threshold ($10K), clusters similar amounts |
| **Layering** | Temporal + Graph | Time-series patterns, network analysis |
| **Smurfing** | Density-Based | Many small transactions in sparse region |
| **Novel Patterns** | Ensemble | Combines all signals for unknown fraud |
| **High Velocity** | Velocity Features + GMM | Probabilistic model for burst activity |

### **12 Algorithms Implemented**

#### **1. K-Means Clustering**
```python
# Concept: Partition into K spherical clusters
# Formula: Minimize Σ ||x_i - center_k||²

from pyspark.ml.clustering import KMeans

kmeans = KMeans(k=10, featuresCol="features", seed=42)
model = kmeans.fit(df)

# Anomaly Score = Distance to nearest cluster center
distance = ||transaction_features - cluster_center||
fraud_score = distance / max_distance  # Normalize to 0-1
```

**When to Use:** Fast general-purpose clustering, good baseline.

**Pros:** Fast, interpretable, works well for spherical clusters.

**Cons:** Must specify K, struggles with non-spherical clusters.

---

#### **2. Gaussian Mixture Model (GMM)**
```python
# Concept: Data comes from K Gaussian distributions
# Formula: P(x) = Σ π_k * N(μ_k, Σ_k)

from pyspark.ml.clustering import GaussianMixture

gmm = GaussianMixture(k=10, featuresCol="features", seed=42)
model = gmm.fit(df)

# Anomaly Score = -log(probability)
fraud_score = -log(max(probability_vector))
```

**When to Use:** Overlapping patterns, probabilistic scores needed.

**Pros:** Soft clustering, handles elliptical clusters, gives probabilities.

**Cons:** Slower than K-Means, sensitive to initialization.

---

#### **3. Ensemble Voting**
```python
# Concept: Combine multiple algorithm predictions
# Formula: score = Σ weight_i * score_i

ensemble_score = (
    0.3 * kmeans_score +
    0.4 * gmm_score +
    0.3 * rule_score
)
```

**When to Use:** Production systems requiring high confidence.

**Pros:** More robust, reduces false positives, diversity wins.

**Cons:** Slower (trains multiple models), harder to debug.

---

#### **4. Z-Score Anomaly Detection**
```python
# Concept: Statistical outlier detection
# Formula: z = (x - μ) / σ

z_score = (amount - customer_avg) / customer_std

# Rule: |z| > 3 → Outlier (99.7% of data within ±3σ)
is_outlier = abs(z_score) > 3
```

**When to Use:** Simple baseline, explainable results.

**Pros:** Fast, interpretable, no training needed.

**Cons:** Assumes normal distribution, misses multivariate patterns.

---

#### **5. PCA + Clustering**
```python
# Concept: Reduce dimensions, then cluster
# Formula: X' = X * eigenvectors

from pyspark.ml.feature import PCA

pca = PCA(k=20, inputCol="features", outputCol="pca_features")
pca_model = pca.fit(df)
reduced_df = pca_model.transform(df)

# Then apply K-Means on reduced dimensions
```

**When to Use:** High-dimensional data (50+ features).

**Pros:** Reduces noise, faster clustering, visualization.

**Cons:** Loses interpretability, may miss important features.

---

### **Ensemble Weight Tuning Strategy**

```python
# Weights based on algorithm strengths

# Accuracy-focused (F1 maximization)
weights = {
    'kmeans': 0.2,   # Fast but less accurate
    'gmm': 0.5,      # Most accurate
    'rules': 0.3     # Domain knowledge
}

# Explainability-focused (compliance)
weights = {
    'kmeans': 0.2,
    'gmm': 0.2,
    'rules': 0.6     # Rule-based is explainable
}

# Speed-focused (real-time)
weights = {
    'kmeans': 0.7,   # Fastest
    'gmm': 0.0,      # Disable slow GMM
    'rules': 0.3
}
```

---

## 🔧 FEATURE ENGINEERING

### **Feature Categories (50+ Total)**

#### **1. Temporal Features**
```python
# Extract time patterns
df = df.withColumn("hour", hour("timestamp"))
       .withColumn("day_of_week", dayofweek("timestamp"))
       .withColumn("is_weekend", when(dayofweek("timestamp").isin([1,7]), 1).otherwise(0))

# Cyclical encoding (hour 23 and hour 0 are neighbors!)
df = df.withColumn("hour_sin", sin(col("hour") * 2 * 3.14159 / 24))
       .withColumn("hour_cos", cos(col("hour") * 2 * 3.14159 / 24))

# Why? Linear: hour=23 and hour=0 are 23 apart
#      Cyclical: hour=23 and hour=0 are 1 hour apart ✅
```

#### **2. Velocity Features (Money Movement Speed)**
```python
# Rolling window aggregations
window_24h = Window.partitionBy("customer_id") \
                   .orderBy(unix_timestamp("timestamp")) \
                   .rangeBetween(-86400, 0)  # 24 hours in seconds

df = df.withColumn("txn_count_24h", count("*").over(window_24h))
       .withColumn("txn_sum_24h", sum("amount").over(window_24h))
       .withColumn("avg_amount_24h", avg("amount").over(window_24h))

# Structuring detection
df = df.withColumn("is_high_velocity",
                   when(col("txn_count_24h") > 5, 1).otherwise(0))
```

#### **3. Customer Baseline Features**
```python
# What's "normal" for this customer?
customer_window = Window.partitionBy("customer_id")

df = df.withColumn("customer_avg", avg("amount").over(customer_window))
       .withColumn("customer_std", stddev("amount").over(customer_window))
       .withColumn("customer_max", max("amount").over(customer_window))

# Deviation from norm (Z-score)
df = df.withColumn("amount_zscore",
                   (col("amount") - col("customer_avg")) /
                   (col("customer_std") + 1))  # +1 prevents div by 0
```

#### **4. AML Rule-Based Features**
```python
# Domain knowledge from compliance experts

# Structuring: Just under $10K threshold
df = df.withColumn("is_structuring",
    when((col("amount") >= 9000) & (col("amount") < 10000), 1)
    .otherwise(0)
)

# Round amounts (exactly $10K, $50K, etc.)
df = df.withColumn("is_round",
    when((col("amount") % 1000 == 0) & (col("amount") >= 10000), 1)
    .otherwise(0)
)

# Unusual hours (midnight to 6 AM)
df = df.withColumn("is_unusual_hour",
    when(col("hour").between(0, 5), 1).otherwise(0)
)

# Composite risk score
df = df.withColumn("rule_based_score",
    col("is_structuring") * 0.4 +
    col("is_round") * 0.3 +
    col("is_unusual_hour") * 0.2 +
    col("is_high_velocity") * 0.1
)
```

#### **5. Text Features (TF-IDF)**
```python
# Convert transaction descriptions to numbers

from pyspark.ml.feature import HashingTF, IDF

# Step 1: Tokenize
df = df.withColumn("words", split(lower(col("description")), "\\W+"))

# Step 2: Term Frequency
hashing_tf = HashingTF(inputCol="words", outputCol="tf", numFeatures=100)
df_tf = hashing_tf.transform(df)

# Step 3: Inverse Document Frequency
idf = IDF(inputCol="tf", outputCol="text_features")
idf_model = idf.fit(df_tf)
df_tfidf = idf_model.transform(df_tf)

# Now "structured", "layering" have high weights!
```

### **Feature Engineering Best Practices**

| Practice | Why Important |
|----------|---------------|
| **Cyclical encoding** | hour=23 and hour=0 should be close |
| **Normalization** | Make all features 0-1 scale |
| **Handle nulls** | `.fillna(0)` or `.dropna()` |
| **Feature selection** | Remove low-importance features |
| **Domain knowledge** | Add AML-specific rules |

---

## 📊 MODEL EVALUATION

### **Confusion Matrix for Fraud Detection**

```
                  ACTUAL
               Fraud | Normal
            ─────────┼─────────
PREDICTED  |         |
  Fraud    |   TP    |   FP   |  ← Alerts sent
           |   850   |  150   |
           ─────────┼─────────
  Normal   |   FN    |   TN   |
           |   150   | 84,850 |
           ─────────┼─────────
```

### **Key Metrics Formulas**

```python
# 1. Precision = How accurate are alerts?
precision = TP / (TP + FP)
          = 850 / (850 + 150)
          = 0.85 (85%)
# Interpretation: 85% of alerts are real fraud

# 2. Recall = How many frauds caught?
recall = TP / (TP + FN)
       = 850 / (850 + 150)
       = 0.85 (85%)
# Interpretation: Caught 85% of all fraud

# 3. F1-Score = Harmonic mean
f1 = 2 * (precision * recall) / (precision + recall)
   = 2 * (0.85 * 0.85) / (0.85 + 0.85)
   = 0.85
# Interpretation: Balanced performance

# 4. False Positive Rate = False alarm rate
fpr = FP / (FP + TN)
    = 150 / (150 + 84850)
    = 0.0018 (0.18%)
# Interpretation: 0.18% of normal transactions flagged

# 5. Business Cost
cost = (FP × $50) + (FN × $1000)
     = (150 × $50) + (150 × $1000)
     = $7,500 + $150,000
     = $157,500
```

### **Precision-Recall Tradeoff**

```python
# Adjust threshold to balance precision vs recall

threshold = 0.7   # Lower threshold
→ More alerts
→ Higher recall (95%) - catch more fraud ✅
→ Lower precision (70%) - more false alarms ❌

threshold = 0.9   # Higher threshold
→ Fewer alerts
→ Lower recall (70%) - miss some fraud ❌
→ Higher precision (95%) - fewer false alarms ✅

# Choose based on business priorities!
```

### **Model Comparison Metrics**

```python
# Evaluate clustering quality
from pyspark.ml.evaluation import ClusteringEvaluator

evaluator = ClusteringEvaluator(
    featuresCol="features",
    predictionCol="cluster",
    metricName="silhouette"
)

silhouette = evaluator.evaluate(predictions)

# Silhouette Score interpretation:
#   > 0.7  Excellent clustering
#   > 0.5  Good clustering
#   > 0.3  Acceptable clustering
#   < 0.3  Poor clustering
```

---

## 🚀 MLOPS & PRODUCTION

### **MLOps Pipeline Components**

```python
# 1. EXPERIMENT TRACKING
class ExperimentTracker:
    def log_experiment(self, run_id, config, metrics):
        experiment = {
            'run_id': run_id,
            'timestamp': datetime.now(),
            'config': config,
            'metrics': metrics
        }
        save_to_file(f"experiments/{run_id}.json", experiment)

# 2. MODEL REGISTRY
class ModelRegistry:
    def register_model(self, name, version, model, metrics):
        model_id = f"{name}_v{version}"
        save_model(model, f"models/{model_id}")
        self.metadata[model_id] = {
            'version': version,
            'metrics': metrics,
            'registered_at': datetime.now()
        }

    def promote_model(self, model_id, stage):  # dev/staging/production
        self.metadata[model_id]['stage'] = stage

# 3. MONITORING
class ModelMonitor:
    def detect_drift(self, production_df, baseline_stats):
        current_mean = production_df.select(mean("amount")).collect()[0][0]

        # Statistical distance
        drift_score = abs(current_mean - baseline_stats['mean']) / baseline_stats['std']

        if drift_score > 2.0:  # 2 standard deviations
            return {"alert": "DRIFT_DETECTED", "action": "RETRAIN"}

        return {"alert": "OK"}
```

### **Training vs Serving Architecture**

```python
# TRAINING (Nightly Batch)
# ─────────────────────────
def train_pipeline():
    # 1. Load last 30 days data
    df = spark.read.parquet("transactions_last_30d.parquet")

    # 2. Engineer features
    feature_df = engineer_features(df)

    # 3. Train models
    models = train_all_algorithms(feature_df)

    # 4. Evaluate
    best_model = select_best(models)

    # 5. Register
    registry.register_model("fraud_detector", version, best_model)

    # 6. Promote to production
    registry.promote_model(model_id, "production")

# SERVING (Real-time API)
# ─────────────────────────
from fastapi import FastAPI

app = FastAPI()
model = load_production_model()

@app.post("/predict")
async def predict(transaction: dict):
    # 1. Engineer features
    features = engineer_features_single(transaction)

    # 2. Score
    fraud_score = model.predict(features)

    # 3. Return result
    return {
        "fraud_score": fraud_score,
        "is_suspicious": fraud_score > 0.85
    }
```

### **Deployment Strategies**

```python
# 1. BATCH PROCESSING (Historical Data)
spark-submit \
  --master yarn \
  --deploy-mode cluster \
  --num-executors 10 \
  --executor-memory 8g \
  main_pipeline.py

# 2. STREAMING (Real-time)
stream = spark.readStream.format("kafka").load()
predictions = model.transform(stream)
predictions.writeStream.format("parquet").start()

# 3. REST API (On-demand)
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4

# 4. DOCKER CONTAINER
docker build -t fraud-detector .
docker run -p 8000:8000 fraud-detector
```

---

## 💬 INTERVIEW QUESTIONS & ANSWERS

### **Q1: Why use clustering instead of classification for fraud detection?**

**Answer:**
"Classification requires labeled data, but in AML:
1. **Labels are unreliable** - Fraud evolves daily, old labels become stale
2. **Extreme class imbalance** - Only 1-5% fraud rate
3. **Novel patterns** - Classification only catches known fraud, clustering catches unknown patterns

With clustering, we define 'normal' behavior and flag anything far from normal. This catches NEW fraud patterns without retraining."

---

### **Q2: How do you handle imbalanced data (95% normal, 5% fraud)?**

**Answer:**
"Three approaches:
1. **Ensemble weighting** - Give more weight to algorithms good at finding rare patterns (GMM)
2. **Anomaly scoring** - Don't classify, assign fraud scores 0-1
3. **Cost-sensitive learning** - Penalize false negatives more (missing fraud costs 20x more than false alarms)

We use ensemble with fraud_score instead of binary classification, then adjust threshold based on business priorities."

---

### **Q3: Explain TF-IDF and why it's useful for fraud detection.**

**Answer:**
"TF-IDF = Term Frequency × Inverse Document Frequency

Example:
- 'structured' appears in 50/100K transactions → IDF = log(100K/50) = 7.6 (high)
- 'payment' appears in 50K/100K transactions → IDF = log(100K/50K) = 0.7 (low)

So 'structured' gets 10x higher weight because it's RARE and therefore more informative. Words like 'payment' are common and ignored.

This helps us catch fraud based on suspicious keywords like 'layering', 'shell company', 'offshore'."

---

### **Q4: How do you detect structuring (transactions just under $10K)?**

**Answer:**
"Multiple signals:
1. **Amount threshold** - Flag if $9K ≤ amount < $10K
2. **Velocity** - Multiple such transactions in 24 hours
3. **Pattern** - Round amounts like $9,999.00 (not $9,247.83)
4. **Text** - Keywords like 'structured', 'split'

Code:
```python
is_structuring = (
    (amount >= 9000) & (amount < 10000) &  # Just under threshold
    (txn_count_24h > 3) &                   # Multiple transactions
    (amount % 100 == 0)                     # Round amount
)
```
"

---

### **Q5: What's the difference between K-Means and GMM?**

**Answer:**

| Aspect | K-Means | GMM |
|--------|---------|-----|
| **Type** | Hard clustering | Soft clustering |
| **Assignment** | Each point → 1 cluster | Each point → probability distribution |
| **Cluster shape** | Spherical | Elliptical |
| **Speed** | Fast | Slower |
| **Use case** | General patterns | Overlapping patterns |

Example:
- K-Means: Transaction belongs to Cluster 3 (100%)
- GMM: Transaction is 70% Cluster 3, 30% Cluster 5 (more nuanced)

For fraud, GMM is better because fraud patterns overlap with normal behavior."

---

### **Q6: How do you handle data drift in production?**

**Answer:**
"Three-step monitoring:

1. **Detect drift** - Compare production data distribution to training baseline
```python
drift_score = abs(current_mean - baseline_mean) / baseline_std
if drift_score > 2.0:
    trigger_alert()
```

2. **Alert** - Notify data science team

3. **Retrain** - Automatically trigger retraining if drift persists for 3+ days

Real example: During COVID-19, transaction patterns changed drastically (more online, less in-person). Our drift detector caught this and we retrained with recent data."

---

### **Q7: How would you scale this to 100M transactions/day?**

**Answer:**
"Three scaling strategies:

1. **Horizontal scaling** - Distribute across Spark cluster
```python
spark-submit \
  --num-executors 50 \
  --executor-cores 4 \
  --executor-memory 16g
```

2. **Streaming** - Process as data arrives (not batch)
```python
spark.readStream.format("kafka")
     .load()
     .transform(detect_fraud)
     .writeStream.start()
```

3. **Feature store** - Pre-compute features (customer_avg, velocity) and cache
```python
# Instead of computing on-the-fly:
customer_stats = spark.read.parquet("feature_store/customer_stats")
df.join(customer_stats, "customer_id")  # Fast lookup
```

Expected: 100M rows in ~1 hour with 50 executors."

---

### **Q8: Explain the ensemble voting mechanism.**

**Answer:**
"We combine 3 algorithms with weighted voting:

```python
fraud_score = (
    0.3 × kmeans_score +    # Fast, general patterns
    0.4 × gmm_score +       # Accurate, overlapping patterns
    0.3 × rule_score        # Explainable, compliance
)
```

**Why weights differ?**
- GMM (0.4) - Most accurate in our tests (F1=0.87)
- K-Means (0.3) - Fast but less accurate (F1=0.82)
- Rules (0.3) - Explainable for compliance

**Benefit:** Diversity! If K-Means misses a pattern, GMM or rules might catch it. This reduces false negatives by 15% compared to single model."

---

### **Q9: How do you explain predictions to compliance officers?**

**Answer:**
"Generate human-readable explanations:

```python
signals = []
if is_structuring:
    signals.append(f'Amount ${amount} is just under $10K threshold')
if is_high_velocity:
    signals.append(f'{txn_count_24h} transactions in 24h (normal: <3)')
if is_unusual_hour:
    signals.append(f'Transaction at {hour}:00 AM (unusual)')

explanation = {
    'fraud_score': 0.94,
    'signals': signals,
    'recommended_action': 'INVESTIGATE'
}
```

Output:
```
Fraud Score: 0.94 (CRITICAL)
Reasons:
  • Amount $9,999.00 is just under $10K threshold
  • 5 transactions in 24h (normal: <3)
  • Transaction at 3:00 AM (unusual)
Action: INVESTIGATE
```

This gives compliance officers SPECIFIC, ACTIONABLE reasons."

---

### **Q10: What's your model deployment workflow?**

**Answer:**
"Full CI/CD pipeline:

1. **Development**
   - Data scientist trains model locally
   - Logs experiment with metrics

2. **Testing**
   - Evaluate on holdout set
   - Validate F1 > 0.85 threshold

3. **Staging**
   - Deploy to staging environment
   - Run A/B test against current production model
   - Monitor for 7 days

4. **Production**
   - If staging metrics ≥ production, promote
   - Blue-green deployment (zero downtime)
   - Monitor for drift

5. **Monitoring**
   - Daily drift detection
   - Weekly performance reports
   - Auto-retrain if F1 drops below 0.80

All automated via Airflow DAGs."

---

## 💻 CODE SNIPPETS (Copy-Paste Ready)

### **1. Feature Engineering Template**

```python
from pyspark.sql.window import Window
from pyspark.sql.functions import *

def engineer_features(df):
    """Complete feature engineering in one function"""

    # Temporal
    df = df.withColumn("hour", hour("timestamp")) \
           .withColumn("hour_sin", sin(col("hour") * 2 * 3.14159 / 24)) \
           .withColumn("is_unusual_hour", when(col("hour").between(0, 5), 1).otherwise(0))

    # Velocity (24-hour window)
    window_24h = Window.partitionBy("customer_id") \
                       .orderBy(unix_timestamp("timestamp")) \
                       .rangeBetween(-86400, 0)

    df = df.withColumn("txn_count_24h", count("*").over(window_24h)) \
           .withColumn("txn_sum_24h", sum("amount").over(window_24h))

    # Customer baseline
    customer_window = Window.partitionBy("customer_id")

    df = df.withColumn("customer_avg", avg("amount").over(customer_window)) \
           .withColumn("customer_std", stddev("amount").over(customer_window)) \
           .withColumn("amount_zscore",
                      (col("amount") - col("customer_avg")) / (col("customer_std") + 1))

    # AML rules
    df = df.withColumn("is_structuring",
                      when((col("amount") >= 9000) & (col("amount") < 10000), 1).otherwise(0)) \
           .withColumn("is_round",
                      when((col("amount") % 1000 == 0) & (col("amount") >= 10000), 1).otherwise(0))

    # Composite risk score
    df = df.withColumn("rule_based_score",
                      col("is_structuring") * 0.4 +
                      col("is_round") * 0.3 +
                      col("is_unusual_hour") * 0.2)

    return df
```

### **2. Ensemble Prediction Template**

```python
def ensemble_predict(df, models, weights):
    """Combine multiple model predictions"""

    # Get predictions from each model
    df = models['kmeans'].transform(df)
    df = models['gmm'].transform(df)

    # Calculate normalized scores
    # K-Means: distance-based
    max_dist = df.agg(max("kmeans_distance")).collect()[0][0]
    df = df.withColumn("kmeans_score", col("kmeans_distance") / lit(max_dist))

    # GMM: probability-based
    df = df.withColumn("gmm_score",
                      -log10(array_max(col("probability")) + lit(1e-10)))
    max_gmm = df.agg(max("gmm_score")).collect()[0][0]
    df = df.withColumn("gmm_score", col("gmm_score") / lit(max_gmm))

    # Ensemble voting
    df = df.withColumn("fraud_score",
        col("kmeans_score") * lit(weights['kmeans']) +
        col("gmm_score") * lit(weights['gmm']) +
        col("rule_based_score") * lit(weights['rules'])
    )

    # Threshold
    df = df.withColumn("is_suspicious", when(col("fraud_score") > 0.85, 1).otherwise(0))

    return df
```

### **3. Real-time Prediction API**

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Transaction(BaseModel):
    transaction_id: str
    amount: float
    timestamp: str
    description: str

@app.post("/predict")
async def predict(txn: Transaction):
    # Convert to Spark DataFrame
    df = spark.createDataFrame([txn.dict()])

    # Engineer features
    features = engineer_features(df)

    # Predict
    result = ensemble_predict(features, models, weights)

    # Return
    row = result.collect()[0]
    return {
        "fraud_score": float(row.fraud_score),
        "is_suspicious": bool(row.is_suspicious),
        "risk_level": "CRITICAL" if row.fraud_score > 0.9 else "HIGH"
    }
```

### **4. Model Training Boilerplate**

```python
def train_fraud_detector(df):
    """Complete training pipeline"""

    # 1. Feature engineering
    feature_df = engineer_features(df)

    # 2. Assemble features
    assembler = VectorAssembler(
        inputCols=["amount", "hour_sin", "txn_count_24h", "amount_zscore"],
        outputCol="features"
    )
    assembled = assembler.transform(feature_df)

    # 3. Train K-Means
    kmeans = KMeans(k=10, featuresCol="features", seed=42)
    kmeans_model = kmeans.fit(assembled)

    # 4. Train GMM
    gmm = GaussianMixture(k=10, featuresCol="features", seed=42)
    gmm_model = gmm.fit(assembled)

    # 5. Save models
    kmeans_model.write().overwrite().save("models/kmeans")
    gmm_model.write().overwrite().save("models/gmm")

    return {'kmeans': kmeans_model, 'gmm': gmm_model}
```

---

## ⚠️ COMMON PITFALLS

### **1. Data Leakage**

```python
# ❌ WRONG: Using future information
df = df.withColumn("customer_avg",
                   avg("amount").over(Window.partitionBy("customer_id")))
# This includes FUTURE transactions!

# ✅ RIGHT: Only use past information
window = Window.partitionBy("customer_id") \
               .orderBy("timestamp") \
               .rowsBetween(Window.unboundedPreceding, -1)  # Exclude current row
df = df.withColumn("customer_avg", avg("amount").over(window))
```

### **2. Not Handling Nulls**

```python
# ❌ WRONG: Nulls crash ML algorithms
df.withColumn("zscore", (col("amount") - col("avg")) / col("std"))
# If std=NULL → zscore=NULL → Model fails!

# ✅ RIGHT: Handle nulls explicitly
df.withColumn("zscore", (col("amount") - col("avg")) / (col("std") + 1))
df.fillna({"zscore": 0})
```

### **3. Forgetting to Cache**

```python
# ❌ WRONG: Recomputes features 3 times!
features = engineer_features(df)  # Expensive operation
train_kmeans(features)  # Recomputes
train_gmm(features)     # Recomputes again!
evaluate(features)      # Recomputes again!!

# ✅ RIGHT: Cache once
features = engineer_features(df).cache()
train_kmeans(features)  # Uses cache
train_gmm(features)     # Uses cache
evaluate(features)      # Uses cache
features.unpersist()    # Clean up
```

### **4. Shuffling Too Much**

```python
# ❌ WRONG: Multiple shuffles
df.groupBy("customer_id").agg(avg("amount"))  # Shuffle 1
df.groupBy("customer_id").agg(count("*"))     # Shuffle 2
df.groupBy("customer_id").agg(max("amount"))  # Shuffle 3

# ✅ RIGHT: Single shuffle
df.groupBy("customer_id").agg(
    avg("amount").alias("avg"),
    count("*").alias("count"),
    max("amount").alias("max")
)  # Only 1 shuffle!
```

### **5. Not Normalizing Features**

```python
# ❌ WRONG: Features have different scales
# amount: 0-100,000
# hour: 0-23
# K-Means will be dominated by amount!

# ✅ RIGHT: Normalize to 0-1 or standardize
from pyspark.ml.feature import StandardScaler

scaler = StandardScaler(inputCol="features", outputCol="scaled_features",
                       withMean=True, withStd=True)
scaler.fit(df).transform(df)
```

---

## 🎓 SUMMARY: KEY TAKEAWAYS

1. **Clustering > Classification** for fraud (catches novel patterns)
2. **Features > Algorithms** (80% of performance)
3. **Ensemble > Single Model** (diversity wins)
4. **Explainability = Trust** (compliance requires interpretability)
5. **Monitor Everything** (fraud evolves, models degrade)
6. **PySpark Window Functions** (essential for time-series features)
7. **TF-IDF for Text** (fast, effective for fraud keywords)
8. **Precision-Recall Tradeoff** (adjust threshold based on business priorities)
9. **MLOps is Non-negotiable** (experiment tracking, versioning, monitoring)
10. **Start Simple, Scale Later** (K-Means + Rules → Add complexity as needed)

---

## 📚 QUICK REFERENCE FORMULAS

```python
# Ensemble Score
fraud_score = Σ(weight_i × score_i)

# Z-Score
z = (x - μ) / σ

# TF-IDF
tfidf = (count/total_words) × log(total_docs/docs_with_word)

# F1-Score
F1 = 2 × (Precision × Recall) / (Precision + Recall)

# Silhouette Score
s = (b - a) / max(a, b)
where a = avg distance within cluster
      b = avg distance to nearest cluster

# Business Cost
cost = (FP × $50) + (FN × $1000)
```

---

**This cheat sheet contains everything you need for interviews and quick revision! Keep it handy! 🚀**
