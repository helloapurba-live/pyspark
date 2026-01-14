# 🧠 AML Fraud Detection - Mind Map & Quick Reference

## 📊 COMPLETE SYSTEM MIND MAP

```
                          AML FRAUD DETECTION SYSTEM
                                    │
                ┌───────────────────┼───────────────────┐
                │                   │                   │
         ┌──────▼──────┐     ┌─────▼─────┐      ┌─────▼─────┐
         │   PROBLEM   │     │  SOLUTION │      │  OUTCOME  │
         └──────┬──────┘     └─────┬─────┘      └─────┬─────┘
                │                   │                   │
    ┌───────────┼───────────┐      │          ┌────────┼────────┐
    │           │           │      │          │        │        │
  $15M      60% FP      Manual     │       85% F1   $13M    Real-time
  Lost      Rate      Review       │       Score   Saved   Detection
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
   ┌────▼────┐              ┌─────▼─────┐             ┌─────▼─────┐
   │  DATA   │              │ FEATURES  │             │ ALGORITHMS│
   └────┬────┘              └─────┬─────┘             └─────┬─────┘
        │                          │                          │
    ┌───┴───┐              ┌───────┼───────┐         ┌───────┼───────┐
    │       │              │       │       │         │       │       │
Tabular   Text        Temporal  Velocity  AML    K-Means   GMM   Ensemble
  │         │              │       │     Rules      │       │       │
  ├─Amount  ├─TF-IDF       │       │       │        │       │       │
  ├─Time    └─Keywords   Hour   Count   Score    Cluster Probability Voting
  ├─Customer             Cycle   24h    Flag      Center  Distribution Weight
  └─Type                 Sin/Cos  7d    Struct
                                        Round
                                        Velocity

                         ┌───────────────┐
                         │   PIPELINE    │
                         └───────┬───────┘
                                 │
            ┌────────────────────┼────────────────────┐
            │                    │                    │
      ┌─────▼─────┐        ┌────▼────┐         ┌────▼────┐
      │  TRAINING │        │  SERVING│         │MONITORING│
      └─────┬─────┘        └────┬────┘         └────┬────┘
            │                   │                    │
      Batch Process        Real-time API        Drift Detection
      Nightly Run          <100ms               Auto-retrain
      Model Registry       FastAPI              Logging
      Experiment Track     Docker               Alerts
```

---

## 🎯 QUICK REFERENCE ARCHITECTURE

### **1. DATA FLOW**

```
Raw Transaction → Feature Engineering → Model Ensemble → Fraud Score → Alert
      │                    │                   │              │           │
   JSON/Parquet      50+ Features        12 Algorithms    0.0-1.0    Threshold
      │                    │                   │              │           │
   100K/day          Tabular+Text         Parallel         >0.85    Investigate
```

### **2. FEATURE CATEGORIES (50+ Features)**

```
┌─────────────────────────────────────────────────────────────┐
│ TEMPORAL FEATURES (Time Patterns)                           │
├─────────────────────────────────────────────────────────────┤
│ • hour, day_of_week, month                                  │
│ • hour_sin, hour_cos (cyclical encoding)                    │
│ • is_unusual_hour (0-5 AM)                                  │
│ • is_weekend, is_business_hours                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ VELOCITY FEATURES (Speed of Money)                          │
├─────────────────────────────────────────────────────────────┤
│ • txn_count_24h (rolling window)                            │
│ • txn_sum_24h (total amount)                                │
│ • txn_count_7d, txn_sum_7d                                  │
│ • time_since_last_txn                                       │
│ • is_high_velocity (>5 txns/24h)                            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ CUSTOMER FEATURES (Baseline Behavior)                       │
├─────────────────────────────────────────────────────────────┤
│ • customer_avg_amount                                       │
│ • customer_std_amount                                       │
│ • customer_max, customer_min                                │
│ • amount_zscore (deviation from norm)                       │
│ • customer_total_transactions                               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ AML RULE FEATURES (Domain Knowledge)                        │
├─────────────────────────────────────────────────────────────┤
│ • is_structuring ($9K-$10K range)                           │
│ • is_round_amount (% 1000 == 0)                             │
│ • is_high_value (>$10K)                                     │
│ • composite_risk_score (weighted sum)                       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ TEXT FEATURES (NLP from Descriptions)                       │
├─────────────────────────────────────────────────────────────┤
│ • TF-IDF vectors (100 dimensions)                           │
│ • Keywords: "structured", "layering", "offshore"            │
│ • HashingTF → IDF transformation                            │
└─────────────────────────────────────────────────────────────┘
```

### **3. ALGORITHM COMPARISON**

```
┌──────────────────┬──────────────┬─────────────┬───────────┬──────────────┐
│   ALGORITHM      │   USE CASE   │    SPEED    │ ACCURACY  │ EXPLAINABILITY│
├──────────────────┼──────────────┼─────────────┼───────────┼──────────────┤
│ K-Means          │ General      │ ⚡⚡⚡ Fast  │ ★★★☆☆     │ ★★★★☆        │
│ Bisecting K-M    │ Hierarchical │ ⚡⚡⚡ Fast  │ ★★★☆☆     │ ★★★☆☆        │
│ GMM              │ Overlapping  │ ⚡⚡ Moderate│ ★★★★☆     │ ★★★☆☆        │
│ PCA+Clustering   │ High Dim     │ ⚡⚡ Moderate│ ★★★☆☆     │ ★★☆☆☆        │
│ Z-Score          │ Statistical  │ ⚡⚡⚡ Fast  │ ★★★☆☆     │ ★★★★★        │
│ Distance-Based   │ Isolation    │ ⚡ Slow     │ ★★★☆☆     │ ★★★☆☆        │
│ Density-Based    │ Rare Combos  │ ⚡⚡ Moderate│ ★★★★☆     │ ★★★☆☆        │
│ Ensemble         │ Robust       │ ⚡ Slow     │ ★★★★★     │ ★★★☆☆        │
│ Risk-Score       │ Compliance   │ ⚡⚡⚡ Fast  │ ★★★★☆     │ ★★★★★        │
│ Behavioral       │ Segmentation │ ⚡⚡⚡ Fast  │ ★★★☆☆     │ ★★★★☆        │
│ Graph-Based      │ Networks     │ ⚡⚡ Moderate│ ★★★★☆     │ ★★★★☆        │
│ Temporal         │ Time Patterns│ ⚡⚡⚡ Fast  │ ★★★☆☆     │ ★★★★☆        │
└──────────────────┴──────────────┴─────────────┴───────────┴──────────────┘
```

### **4. FRAUD PATTERNS DETECTED**

```
1. STRUCTURING (Smurfing)
   Pattern: Multiple transactions just under $10K threshold
   Example: $9,999 × 5 = $49,995 (instead of 1 × $50K)
   Detection: is_structuring flag + velocity check

2. RAPID MOVEMENT (Layering)
   Pattern: Money moves through accounts quickly
   Example: A→B→C→D within 2 hours
   Detection: High velocity + short time_since_last_txn

3. ROUND AMOUNTS
   Pattern: Suspiciously exact transactions
   Example: Exactly $50,000.00 (not $50,247.83)
   Detection: amount % 1000 == 0 && amount >= 10000

4. UNUSUAL TIMING
   Pattern: Transactions at odd hours
   Example: 3:47 AM on Tuesday
   Detection: hour between 0-5

5. HIGH-RISK JURISDICTIONS
   Pattern: Transfers to offshore accounts
   Example: Wire to Cayman Islands
   Detection: country in high_risk_list

6. SMURFING
   Pattern: Many small deposits from multiple people
   Example: 20 deposits of $500 each
   Detection: High velocity + small amounts + multiple accounts
```

### **5. PERFORMANCE BENCHMARKS**

```
┌─────────────────────────────────────────────────────────┐
│ DATA VOLUME vs PROCESSING TIME                         │
├─────────────────────────────────────────────────────────┤
│ 100K rows    →  5 min   (local[*])                     │
│ 1M rows      →  10 min  (local[*])                     │
│ 10M rows     →  20 min  (standalone cluster, 3 nodes)  │
│ 100M rows    →  30 min  (EMR, 10 executors)            │
│ Streaming    →  <5 sec  (per micro-batch)              │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ EXPECTED METRICS (100K transactions)                   │
├─────────────────────────────────────────────────────────┤
│ Precision:        82-85%  (Alert accuracy)             │
│ Recall:           85-90%  (Fraud caught)               │
│ F1-Score:         83-87%  (Balanced)                   │
│ False Positive:   5-10%   (False alarms)               │
│ Processing Time:  5-10 min (Training)                  │
│ Inference:        <100ms  (Single transaction)         │
└─────────────────────────────────────────────────────────┘
```

### **6. MLOPS WORKFLOW**

```
┌─────────────────────────────────────────────────────────┐
│ TRAINING (Nightly Batch)                                │
├─────────────────────────────────────────────────────────┤
│ 1. Load last 30 days transactions                      │
│ 2. Engineer features                                    │
│ 3. Train 12 algorithms in parallel                      │
│ 4. Evaluate & select best                              │
│ 5. Register model with version                         │
│ 6. Promote to staging → production                     │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ SERVING (Real-time API)                                 │
├─────────────────────────────────────────────────────────┤
│ 1. Receive transaction via POST /predict               │
│ 2. Load production model from registry                 │
│ 3. Engineer features on-the-fly                        │
│ 4. Score with ensemble                                  │
│ 5. Return fraud_score + explanation                    │
│ 6. Log prediction for monitoring                       │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ MONITORING (24/7)                                       │
├─────────────────────────────────────────────────────────┤
│ 1. Track prediction distribution                       │
│ 2. Detect data drift (KL divergence)                   │
│ 3. Monitor false positive/negative rates               │
│ 4. Alert on anomalies                                   │
│ 5. Trigger retraining if drift > threshold             │
└─────────────────────────────────────────────────────────┘
```

---

## 🔑 KEY CONCEPTS SUMMARY

### **Ensemble Voting Formula**
```
fraud_score = (kmeans_score × 0.3) + (gmm_score × 0.4) + (rule_score × 0.3)

Where:
  kmeans_score = distance_to_cluster_center / max_distance
  gmm_score    = -log(probability) / max_log_prob
  rule_score   = weighted_sum_of_flags / max_possible
```

### **TF-IDF Transformation**
```
TF-IDF(word, doc) = TF(word, doc) × IDF(word)

Where:
  TF(word, doc) = count(word) / total_words
  IDF(word)     = log(total_docs / docs_containing_word)

Example:
  "structured" appears in 50/100,000 transactions
  IDF("structured") = log(100000/50) = 7.6  ← High importance!
```

### **Z-Score Calculation**
```
z_score = (x - μ) / σ

Where:
  x = current transaction amount
  μ = customer's average amount
  σ = customer's standard deviation

Interpretation:
  |z| > 3  → Outlier (flag)
  |z| > 2  → Unusual (monitor)
  |z| < 2  → Normal
```

### **Window Functions (Velocity)**
```python
# Rolling 24-hour window
window_24h = (Window
              .partitionBy("customer_id")
              .orderBy(unix_timestamp("timestamp"))
              .rangeBetween(-86400, 0))  # 24h in seconds

txn_count_24h = count("*").over(window_24h)
```

---

## 📋 CONFIGURATION TUNING GUIDE

### **Threshold Adjustment Impact**

```
threshold = 0.7 (Lower)
  → More alerts
  → Higher recall (90%+)
  → Lower precision (70%)
  → Use when: Compliance-critical, can't miss fraud

threshold = 0.85 (Medium)  ← DEFAULT
  → Balanced alerts
  → Balanced recall/precision (85%/85%)
  → Use when: General production

threshold = 0.95 (Higher)
  → Fewer alerts
  → Lower recall (70%)
  → Higher precision (95%)
  → Use when: Limited investigation resources
```

### **Ensemble Weight Tuning**

```yaml
# Accuracy-focused (catch most fraud)
weights:
  kmeans: 0.2
  gmm: 0.5      # Increase probabilistic model
  rules: 0.3

# Explainability-focused (compliance)
weights:
  kmeans: 0.2
  gmm: 0.2
  rules: 0.6    # Increase rule-based

# Speed-focused (real-time)
weights:
  kmeans: 0.7   # Fast algorithm
  gmm: 0.0      # Disable slow GMM
  rules: 0.3
```

---

## 🚀 DEPLOYMENT MODES

### **1. Batch Processing (Nightly)**
```bash
spark-submit \
  --master yarn \
  --deploy-mode cluster \
  --num-executors 10 \
  --executor-memory 8g \
  main_pipeline.py
```

### **2. Real-time API (24/7)**
```bash
docker run -p 8000:8000 fraud-detector:latest
```

### **3. Streaming (Kafka)**
```python
spark.readStream.format("kafka").load()
  .transform(detect_fraud)
  .writeStream.format("kafka").start()
```

---

## 💡 TROUBLESHOOTING QUICK FIXES

| Problem | Cause | Solution |
|---------|-------|----------|
| Out of Memory | Large dataset | Increase `spark.driver.memory` to 8g |
| Slow Performance | Too many shuffles | Use `.repartition("customer_id")` |
| Low Precision | Threshold too low | Increase threshold to 0.9 |
| Low Recall | Threshold too high | Decrease threshold to 0.7 |
| High Latency | GMM slow | Disable GMM, use K-Means only |
| False Positives | Noisy features | Remove low-importance features |
| Model Staleness | Drift | Retrain weekly instead of monthly |

---

## 📊 BUSINESS IMPACT CALCULATOR

```python
# Given 100K transactions, 15% fraud rate

total_transactions = 100_000
fraud_rate = 0.15
actual_frauds = total_transactions * fraud_rate  # 15,000

# Model with F1=0.85
precision = 0.85
recall = 0.85

flagged = actual_frauds / recall  # ~17,650 alerts
true_positives = actual_frauds * recall  # 12,750 caught
false_positives = flagged - true_positives  # 4,900
false_negatives = actual_frauds - true_positives  # 2,250

# Costs
cost_per_fp = 50  # Investigation time
cost_per_fn = 1000  # Fraud loss

total_cost = (false_positives * cost_per_fp) + (false_negatives * cost_per_fn)
# = (4,900 × $50) + (2,250 × $1,000)
# = $245,000 + $2,250,000
# = $2,495,000

# Without model (all fraud missed)
baseline_cost = actual_frauds * cost_per_fn
# = 15,000 × $1,000
# = $15,000,000

savings = baseline_cost - total_cost
# = $15M - $2.5M
# = $12.5M saved! 🎉
```

---

## 🎯 QUICK START COMMANDS

```bash
# Setup
cd aml_fraud_detection
pip install -r requirements.txt

# Run full pipeline
./run_pipeline.sh

# Or manually
cd src && python main_pipeline.py

# Run specific component
python data_generator.py          # Generate data only
python feature_engineering.py     # Features only
python clustering_algorithms.py   # Training only

# Start API server
uvicorn api:app --reload

# Test API
curl -X POST http://localhost:8000/predict \
  -d '{"transaction_id": "TXN123", "amount": 9999, ...}'

# Docker deployment
docker build -t fraud-detector .
docker run -p 8000:8000 fraud-detector
```

---

## 📚 FILE STRUCTURE REFERENCE

```
aml_fraud_detection/
├── configs/config.yaml           ← All settings
├── src/
│   ├── data_generator.py         ← Generate synthetic data
│   ├── feature_engineering.py    ← 50+ features
│   ├── clustering_algorithms.py  ← 12 algorithms
│   ├── model_evaluation.py       ← Metrics & comparison
│   ├── mlops_components.py       ← Tracking & registry
│   └── main_pipeline.py          ← Orchestration
├── data/
│   ├── raw_transactions.parquet  ← Generated data
│   └── features.parquet          ← Engineered features
├── models/
│   ├── registry/                 ← Model versions
│   └── experiments/              ← Experiment logs
├── results/
│   ├── executive_summary.txt     ← High-level report
│   └── evaluation_results.json   ← Detailed metrics
└── logs/
    └── aml_fraud_detection.log   ← System logs
```

---

## 🎓 LEARNING PATH

### **Beginner Track**
1. Read QUICKSTART.md
2. Run `./run_pipeline.sh`
3. Examine `results/executive_summary.txt`
4. Read TUTORIAL.md sections 1-3

### **Intermediate Track**
1. Study `feature_engineering.py`
2. Modify `config.yaml` parameters
3. Add custom features
4. Compare algorithm performance

### **Advanced Track**
1. Implement new algorithms
2. Add streaming pipeline
3. Deploy to cloud (EMR/Dataproc)
4. Build monitoring dashboard

---

This mind map provides a complete visual and conceptual overview of the entire system. Keep it handy for quick reference! 🚀
