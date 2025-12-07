# 🎯 AML Fraud Detection - Quick Reference

## 📦 One-Line Commands

```bash
# Complete pipeline (automated)
./run_pipeline.sh

# Compact version (200 lines)
python src/compact_pipeline.py

# Full pipeline (detailed logging)
python src/main_pipeline.py
```

---

## 🎨 Code Patterns

### **Generate Data**
```python
from src.data.generate_data import BankingDataGenerator

gen = BankingDataGenerator(num_transactions=50000, fraud_ratio=0.2)
customers = gen.generate_customer_profiles()
transactions = gen.generate_all_transactions(customers)
```

### **Feature Engineering**
```python
from src.features.feature_engineering import FeatureEngineer

engineer = FeatureEngineer(spark)
df_featured = engineer.engineer_features(df)
pipeline = engineer.build_feature_pipeline()
```

### **Train Model**
```python
from src.models.model_trainer import ModelTrainer

trainer = ModelTrainer(spark)
train, val, test = trainer.split_data(df)
model = trainer.train_random_forest(train, "RF", num_trees=100)
```

### **Evaluate**
```python
from src.evaluation.model_evaluator import ModelEvaluator

evaluator = ModelEvaluator(spark)
results = evaluator.evaluate_model(model, test, "MyModel")
comparison = evaluator.compare_models(results)
```

### **Deploy**
```python
from src.deployment.model_inference import ModelInference

inference = ModelInference(spark)
inference.load_model("data/models/RandomForest")
predictions = inference.predict_from_raw(new_data)
```

---

## 📊 Data Patterns

### **Fraud Signatures**

| Type | Amount Range | Time Pattern | Description Keywords |
|------|-------------|--------------|---------------------|
| **STRUCTURING** | $7k-$10k | Night (10pm-6am) | "cash", "deposit", "threshold" |
| **LAYERING** | $10k-$500k | Any | "international", "wire", "offshore" |
| **SHELL_COMPANY** | Round ($10k, $25k, $50k) | Business hours | "consulting", "services", "advisory" |
| **ROUND_TRIPPING** | $20k-$1M | Any | "investment", "return", "loan" |
| **LEGITIMATE** | $5-$500 | 8am-8pm | "grocery", "gas", "restaurant" |

---

## 🔧 Feature Engineering

### **Quick Feature Creation**
```python
# Amount features (6)
df.withColumn('amount_log', F.log1p('amount'))
df.withColumn('is_near_threshold', F.col('amount').between(7000, 10000))
df.withColumn('is_large', F.col('amount') > 10000)
df.withColumn('is_round', F.col('amount') % 1000 < 10)
df.withColumn('amount_bin', F.when(F.col('amount') < 100, 'SMALL')
                             .when(F.col('amount') < 1000, 'MEDIUM')
                             .otherwise('LARGE'))

# Time features (8)
df.withColumn('hour', F.hour('timestamp'))
df.withColumn('is_night', F.col('hour').between(22, 6))
df.withColumn('is_weekend', F.dayofweek('timestamp').isin([1,7]))
df.withColumn('is_business_hours', F.col('hour').between(9, 17))

# Text features (100 via TF-IDF)
Pipeline(stages=[
    Tokenizer(inputCol="description", outputCol="words"),
    StopWordsRemover(inputCol="words", outputCol="filtered"),
    HashingTF(inputCol="filtered", outputCol="tf", numFeatures=100),
    IDF(inputCol="tf", outputCol="text_features")
])
```

---

## 🤖 Model Quick Reference

### **Algorithm Selection Matrix**

```
                Speed    Accuracy   Interpretability   Handles Text
─────────────────────────────────────────────────────────────────────
Logistic Reg    ⚡⚡⚡      ★★☆         ★★★              ★★☆
Decision Tree   ⚡⚡       ★★☆         ★★★              ★☆☆
Random Forest   ⚡        ★★★         ★★☆              ★★☆
GBT             ⚡        ★★★         ★☆☆              ★★☆
Naive Bayes     ⚡⚡⚡      ★★☆         ★★☆              ★★★
Neural Net      ⚡        ★★★         ★☆☆              ★★★
```

### **Hyperparameter Cheat Sheet**

```python
# Random Forest (best overall)
RandomForestClassifier(
    numTrees=100,        # 50-200 (more = better, slower)
    maxDepth=10,         # 5-20 (deeper = complex patterns)
    minInstancesPerNode=5  # 5-20 (higher = prevent overfit)
)

# Gradient Boosting (highest accuracy)
GBTClassifier(
    maxIter=50,          # 30-100 (more iterations)
    maxDepth=5,          # 3-8 (shallower than RF)
    stepSize=0.1         # 0.01-0.3 (learning rate)
)

# Neural Network (complex patterns)
MultilayerPerceptronClassifier(
    layers=[features, 128, 64, classes],  # [input, hidden..., output]
    maxIter=100,         # 50-200
    stepSize=0.03        # 0.01-0.1 (learning rate)
)
```

---

## 📈 Evaluation Metrics

### **Metric Interpretation**

```python
# Accuracy: Overall correctness
accuracy = correct_predictions / total_predictions

# Precision: Of flagged frauds, how many are real?
precision = true_positives / (true_positives + false_positives)

# Recall: Of real frauds, how many did we catch?
recall = true_positives / (true_positives + false_negatives)

# F1-Score: Harmonic mean of precision and recall
f1 = 2 * (precision * recall) / (precision + recall)
```

### **Quick Evaluation**

```python
evaluator = MulticlassClassificationEvaluator(labelCol='label')

# All metrics in one go
metrics = {
    'accuracy': evaluator.evaluate(preds, {evaluator.metricName: "accuracy"}),
    'f1': evaluator.evaluate(preds, {evaluator.metricName: "f1"}),
    'precision': evaluator.evaluate(preds, {evaluator.metricName: "weightedPrecision"}),
    'recall': evaluator.evaluate(preds, {evaluator.metricName: "weightedRecall"})
}
```

### **Confusion Matrix Quick Look**

```python
# Simple confusion matrix
preds.groupBy('label', 'prediction').count().show()

# Detailed per-class analysis
for class_id in range(num_classes):
    tp = preds.filter((F.col('prediction') == class_id) &
                     (F.col('label') == class_id)).count()
    fp = preds.filter((F.col('prediction') == class_id) &
                     (F.col('label') != class_id)).count()
    fn = preds.filter((F.col('prediction') != class_id) &
                     (F.col('label') == class_id)).count()
```

---

## 🚀 Deployment Patterns

### **Batch Prediction**
```python
# Load model
model = PipelineModel.load("data/models/RandomForest")

# Predict on large dataset
new_data = spark.read.parquet("data/new_transactions.parquet")
predictions = model.transform(new_data)

# Save results
predictions.select('transaction_id', 'prediction', 'probability') \
           .write.parquet("data/predictions/")
```

### **Real-time Prediction**
```python
def predict_single(transaction):
    """Predict single transaction"""
    df = spark.createDataFrame([transaction])
    result = model.transform(df).first()
    return {
        'fraud_probability': float(result.probability[1]),
        'predicted_class': int(result.prediction),
        'is_fraud': int(result.prediction) > 0
    }

# Example
predict_single({
    'amount': 9800,
    'description': 'Cash deposit',
    'hour': 23
})
```

---

## 💡 Performance Optimization

### **Spark Tuning**
```python
spark = SparkSession.builder \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "10") \
    .config("spark.default.parallelism", "10") \
    .getOrCreate()
```

### **Data Caching**
```python
# Cache frequently accessed data
df = df.cache()

# Unpersist when done
df.unpersist()
```

### **Partition Optimization**
```python
# Repartition for parallelism
df = df.repartition(10)

# Coalesce to reduce partitions
df = df.coalesce(1)  # For single output file
```

---

## 🎯 Common Issues & Fixes

### **Issue: Out of Memory**
```yaml
# In config/config.yaml
spark:
  config:
    "spark.driver.memory": "2g"  # Reduce
    "spark.executor.memory": "2g"

# Or reduce data size
data:
  generation:
    num_transactions: 10000  # Instead of 50000
```

### **Issue: Slow Training**
```python
# Use fewer trees/iterations
RandomForestClassifier(numTrees=50)  # Instead of 200
GBTClassifier(maxIter=30)  # Instead of 100

# Use smaller dataset for testing
df.sample(0.1).cache()  # 10% sample
```

### **Issue: Poor Performance**
```python
# Add more features
df = df.withColumn('new_feature', custom_logic)

# Try different algorithms
models = train_all_models(train_df)  # Compare multiple

# Tune hyperparameters
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
```

---

## 📊 Expected Results

### **Benchmark Performance**
```
Dataset Size: 50,000 transactions
Training Time: 10-15 minutes (local machine)
Best Model: Random Forest (tuned)
  ├─ Accuracy: 92-93%
  ├─ F1-Score: 91-92%
  ├─ Precision: 91-92%
  └─ Recall: 90-91%
```

### **Per-Class Targets**
```
LEGITIMATE:     95%+ recall (don't annoy customers)
STRUCTURING:    90%+ recall (critical to catch)
LAYERING:       90%+ recall (regulatory requirement)
SHELL_COMPANY:  88%+ recall (complex pattern)
ROUND_TRIPPING: 88%+ recall (hardest to detect)
```

---

## 🔗 Quick Links

```bash
# Documentation
cat README.md              # Project overview
cat QUICKSTART.md          # 5-minute setup
cat TUTORIAL.md            # Complete guide
cat COMPLETE_GUIDE.md      # Consolidated tutorial

# Results
cat data/reports/model_comparison.csv
cat data/reports/evaluation_report.md

# Logs
tail -f logs/pipeline.log
```

---

## 🎨 Customization Snippets

### **Change Dataset Balance**
```python
# In generate_data.py
BankingDataGenerator(
    num_transactions=100000,  # More data
    fraud_ratio=0.15          # Less fraud (more realistic)
)
```

### **Add Custom Fraud Pattern**
```python
# In FRAUD_PATTERNS dict
'MY_PATTERN': lambda: {
    'amount': custom_amount_logic(),
    'desc': 'Custom description',
    'hour': custom_time_logic()
}
```

### **Add Model Variant**
```python
# In train_models()
models['RF_Deep'] = RandomForestClassifier(
    numTrees=200,
    maxDepth=20,  # Much deeper
    minInstancesPerNode=2
).fit(train_df)
```

---

## ✅ Checklist

**Setup**
- [ ] Python 3.8+ installed
- [ ] Virtual environment created
- [ ] Dependencies installed (`pip install -r requirements.txt`)

**Run**
- [ ] Data generated (`python src/data/generate_data.py`)
- [ ] Pipeline executed (`python src/main_pipeline.py`)
- [ ] Results reviewed (`data/reports/`)

**Optimize**
- [ ] Hyperparameters tuned
- [ ] Custom features added
- [ ] Best model selected
- [ ] Performance benchmarked

**Deploy**
- [ ] Model saved
- [ ] Inference tested
- [ ] Monitoring setup
- [ ] Documentation updated

---

**🎯 Pro Tip:** Start with `compact_pipeline.py` (200 lines) to understand the flow, then explore the full modular version for production use!
