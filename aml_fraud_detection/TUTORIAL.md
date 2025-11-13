# 🎓 AML Fraud Detection - Complete Tutorial

## Welcome, Future Data Scientist! 👋

This tutorial will guide you through the **entire AML fraud detection system**, explaining every concept like a friend teaching you over coffee. No robotic jargon - just clear, friendly explanations!

---

## 📚 Table of Contents

1. [What is AML Fraud Detection?](#what-is-aml-fraud-detection)
2. [Understanding the Data](#understanding-the-data)
3. [Feature Engineering Explained](#feature-engineering-explained)
4. [Clustering Algorithms - Simple Explanations](#clustering-algorithms)
5. [Evaluating Models](#evaluating-models)
6. [MLOps Pipeline](#mlops-pipeline)
7. [Running Your First Experiment](#running-your-first-experiment)
8. [Interpreting Results](#interpreting-results)
9. [Next Steps](#next-steps)

---

## 🕵️ What is AML Fraud Detection?

### The Problem

**Money laundering** is when criminals try to hide "dirty money" (from crimes) by moving it through the financial system to make it look "clean."

### Common Techniques Criminals Use:

1. **Structuring (Smurfing)**
   - Breaking $50,000 into multiple $9,999 transactions
   - Why? To avoid the $10,000 reporting threshold!
   - Example: Instead of one $50K deposit, make five $9,999 deposits

2. **Rapid Movement (Layering)**
   - Moving money quickly through multiple accounts
   - Like a shell game - hide the trail!
   - Example: Money enters Account A, moves to B, C, D within hours

3. **Round Amounts**
   - Transferring exactly $10,000 or $50,000
   - Real people rarely use exact round numbers
   - Example: Normal = $1,247.83, Suspicious = $50,000.00

4. **Unusual Timing**
   - Transactions at 3 AM on a Tuesday
   - Who transfers money at weird hours?
   - Example: Multiple large transfers between midnight and 5 AM

5. **High-Risk Jurisdictions**
   - Sending money to countries with weak AML controls
   - Following the money trail across borders
   - Example: Wire transfers to offshore tax havens

### Our Goal

Build a system that **automatically flags suspicious transactions** so investigators can review them!

---

## 📊 Understanding the Data

### What's in a Banking Transaction?

Let's look at a sample transaction:

```
Transaction ID: TXN0000012345
Customer ID: CUST001234
Account ID: ACC005678
Amount: $9,999.00
Timestamp: 2025-11-13 03:45:22
Type: CASH_DEPOSIT
Description: "Large cash deposit structured payment"
Channel: BRANCH
```

### Red Flags in This Transaction:

1. ⚠️ Amount is $9,999 (just under $10K threshold) - **Structuring!**
2. ⚠️ Time is 3:45 AM - **Unusual timing!**
3. ⚠️ Description mentions "structured" - **Keyword alert!**
4. ⚠️ Cash deposit (harder to trace) - **Higher risk!**

This is clearly suspicious! Our algorithms should flag this.

### Types of Data We Have:

**Tabular Data (Numbers & Categories):**
- Amount: $9,999.00
- Hour: 3
- Day of week: Tuesday
- Transaction type: CASH_DEPOSIT
- Customer risk level: HIGH

**Text Data (Descriptions):**
- "Large cash deposit structured payment"
- "Rapid funds movement layering transaction"
- "International wire transfer high risk jurisdiction"

Our system processes **both types** to catch fraud!

---

## 🔧 Feature Engineering Explained

### What is Feature Engineering?

Think of it as **translating** raw data into a language that machine learning algorithms understand.

### Example: From Raw to Features

**Raw Transaction:**
```
Amount: $9,999.00
Timestamp: 2025-11-13 03:45:22
Customer: CUST001234
```

**Engineered Features:**
```python
# Original features
amount = 9999.00
timestamp_hour = 3

# NEW FEATURES we create:

# 1. Log amount (better for ML)
amount_log = log10(9999.00) = 4.00

# 2. Is this unusual timing?
is_unusual_hour = 1 (yes, 3 AM is weird!)

# 3. Customer's average transaction
customer_avg_amount = $2,500 (from history)

# 4. How different is this from customer's normal?
deviation_from_avg = (9999 - 2500) / 2500 = 3.0
# This is 3x their normal amount!

# 5. How many transactions today?
txn_count_24h = 5  # They made 5 transactions today

# 6. Is amount suspiciously round?
is_round_amount = 1 (yes, it's close to $10K)

# 7. Text features from description
description_words = ["large", "cash", "structured", "payment"]
# "structured" is a red flag word!

# 8. Risk score (combining multiple signals)
composite_risk_score = 0.85  # Very high!
```

### Why This Matters:

Instead of just looking at `amount = 9999`, we now know:
- It's 3x the customer's normal amount ✅
- It happened at a weird time ✅
- It's suspiciously round ✅
- Description contains red flag words ✅
- Customer has high velocity today ✅

**This makes fraud MUCH easier to detect!**

---

## 🤖 Clustering Algorithms - Simple Explanations

### What is Clustering?

Clustering is **grouping similar things together**. Like organizing:
- Apples with apples
- Oranges with oranges
- Weird alien fruit separately (fraud!)

### Algorithm #1: K-Means

**What it does:**
Finds K groups (clusters) of similar transactions.

**How it works:**
1. Pick K random center points
2. Assign each transaction to nearest center
3. Move centers to average of their group
4. Repeat until centers stop moving

**Real-world analogy:**
Like organizing a classroom - students sit near friends with similar interests. Math lovers here, artists there, fraudsters in the corner!

**Best for:**
Finding general patterns in fraud behavior.

**Example:**
```
Cluster 1: Normal retail purchases ($10-$100)
Cluster 2: Bill payments ($50-$500)
Cluster 3: Salary deposits ($2K-$5K)
Cluster 4: Weird late-night $9,999 transactions ← FRAUD!
```

### Algorithm #2: Gaussian Mixture Model (GMM)

**What it does:**
Like K-Means but allows "fuzzy" membership - a transaction can partially belong to multiple clusters.

**How it works:**
Assumes each cluster is a bell curve (Gaussian distribution). Transactions can be 70% in Cluster A, 30% in Cluster B.

**Real-world analogy:**
Like music genres - a song can be 60% rock, 40% jazz. Some transactions are "suspicious-ish" rather than definitely fraud.

**Best for:**
Complex, overlapping fraud patterns.

**Example:**
```
Transaction X:
- 60% matches "normal" cluster
- 40% matches "suspicious" cluster
Conclusion: Moderately suspicious, worth reviewing
```

### Algorithm #3: Ensemble (Combining Multiple Algorithms)

**What it does:**
Asks multiple algorithms for their opinion, then combines them.

**How it works:**
1. K-Means says: "Suspicious"
2. GMM says: "Suspicious"
3. PCA says: "Normal"
Vote: 2 out of 3 say suspicious → Flag it!

**Real-world analogy:**
Like asking three doctors for a diagnosis. If 2 out of 3 say you're sick, you're probably sick!

**Best for:**
High-confidence fraud detection.

### Algorithm #4: Risk-Score Based

**What it does:**
Uses domain knowledge (AML rules) to score transactions.

**How it works:**
```python
risk_score = 0

# Check each red flag
if amount > 10000:
    risk_score += 0.2

if hour in [0,1,2,3,4,5]:  # Weird hours
    risk_score += 0.2

if amount % 1000 == 0:  # Round number
    risk_score += 0.3

if "structured" in description:
    risk_score += 0.3

# risk_score = 1.0 → Very suspicious!
```

**Real-world analogy:**
Like a checklist at airport security. Each red flag adds points. High score = extra screening!

**Best for:**
Explainable, compliance-friendly detection.

### All 12 Algorithms at a Glance:

| Algorithm | What It Does | Best For | Speed |
|-----------|--------------|----------|-------|
| K-Means | Groups similar transactions | General patterns | ⚡⚡⚡ |
| Bisecting K-Means | Hierarchical grouping | Nested patterns | ⚡⚡⚡ |
| GMM | Fuzzy clustering | Overlapping patterns | ⚡⚡ |
| PCA + Clustering | Simplify then cluster | High dimensions | ⚡⚡ |
| Z-Score | Statistical outliers | Simple anomalies | ⚡⚡⚡ |
| Distance-Based | Find isolated points | Unusual transactions | ⚡⚡ |
| Density-Based | Sparse regions | Rare combinations | ⚡⚡ |
| Ensemble | Combine multiple | High confidence | ⚡ |
| Risk-Score | Rule-based | Explainability | ⚡⚡⚡ |
| Behavioral | Customer profiles | Customer types | ⚡⚡⚡ |
| Graph-Based | Network analysis | Money networks | ⚡⚡ |
| Temporal | Time patterns | Timing-based fraud | ⚡⚡⚡ |

---

## 📊 Evaluating Models

### How Do We Know Which Algorithm is Best?

We grade them like students! Multiple tests:

### Test #1: Precision (Accuracy of Alerts)

**Question:** Of all the transactions we flagged, how many are actually fraud?

**Formula:**
```
Precision = True Frauds Flagged / Total Flagged
          = 85 / 100 = 0.85 (85%)
```

**What this means:**
- High precision (85%+) = Few false alarms ✅
- Low precision (50%) = Lots of false alarms ❌

**Real-world impact:**
Low precision means investigators waste time checking normal transactions!

### Test #2: Recall (Coverage of Fraud)

**Question:** Of all the actual frauds, how many did we catch?

**Formula:**
```
Recall = True Frauds Flagged / Total Actual Frauds
       = 90 / 100 = 0.90 (90%)
```

**What this means:**
- High recall (90%+) = Catching most frauds ✅
- Low recall (60%) = Missing many frauds ❌

**Real-world impact:**
Low recall means real criminals are getting away!

### Test #3: F1-Score (Balance)

**Question:** What's the balance between precision and recall?

**Formula:**
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
   = 2 * (0.85 * 0.90) / (0.85 + 0.90)
   = 0.87 (87%)
```

**What this means:**
F1-Score is the "overall grade" - higher is better!

### Test #4: Business Cost

**The Reality:**
- False Positive (false alarm) costs $50 (investigation time)
- False Negative (missed fraud) costs $1000 (actual fraud loss)

**Example:**
```
Algorithm A: 100 false positives, 10 false negatives
Cost = (100 * $50) + (10 * $1000) = $15,000

Algorithm B: 200 false positives, 5 false negatives
Cost = (200 * $50) + (5 * $1000) = $15,000

They're equal from a business perspective!
```

### Our Leaderboard:

```
Rank  Algorithm              F1      Business Cost
────────────────────────────────────────────────────
1     Ensemble               0.850   $12,000  🏆
2     GMM                    0.840   $13,500
3     Risk-Score             0.830   $14,000
4     K-Means                0.820   $15,000
```

**Winner:** Ensemble (best F1-score AND lowest cost)!

---

## 🚀 MLOps Pipeline

### What is MLOps?

**MLOps = Machine Learning Operations**

Think of it as the "infrastructure" that makes ML production-ready. Like how DevOps helps deploy software, MLOps helps deploy ML models!

### Key Components:

#### 1. Experiment Tracking

**What:** Record every training run
**Why:** So you can compare and reproduce results

**Example:**
```
Run 1 (2025-11-13 10:00):
  - Config: K-Means, k=10
  - F1-Score: 0.82
  - Time: 5 minutes

Run 2 (2025-11-13 11:00):
  - Config: GMM, k=10
  - F1-Score: 0.84  ← Better!
  - Time: 8 minutes
```

#### 2. Model Registry

**What:** Version control for models
**Why:** Track which model is in production

**Example:**
```
Models:
  - kmeans_v1.0 [dev]
  - gmm_v1.0 [staging]
  - ensemble_v1.0 [production] ← Currently live
  - ensemble_v1.1 [testing]
```

#### 3. Monitoring

**What:** Watch model performance over time
**Why:** Detect when it starts failing

**Example:**
```
Week 1: F1 = 0.85 ✅
Week 2: F1 = 0.84 ✅
Week 3: F1 = 0.70 ⚠️ Alert! Performance dropped!
```

**What happened?** Maybe fraud patterns changed, or data distribution shifted!

#### 4. Data Drift Detection

**What:** Detect when input data changes
**Why:** Models trained on old data might not work on new data

**Example:**
```
Training data: Average transaction = $100
Production data: Average transaction = $500
⚠️ DRIFT DETECTED!

Possible causes:
- Inflation
- Different customer segment
- Seasonal changes
- Data quality issues
```

---

## 🏃 Running Your First Experiment

### Step 1: Installation

```bash
# Clone the repository
git clone <repository-url>
cd aml_fraud_detection

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Check Configuration

Open `configs/config.yaml` and verify settings:

```yaml
data:
  num_transactions: 100000  # Start small for testing: 10000
  fraud_ratio: 0.15         # 15% suspicious transactions

algorithms:
  kmeans:
    enabled: true           # Try K-Means
    k: [5, 10, 15]         # Test 3 values of K
```

### Step 3: Run the Pipeline

```bash
# Easy mode (recommended for beginners)
chmod +x run_pipeline.sh
./run_pipeline.sh

# Or manually
cd src
python main_pipeline.py
```

### Step 4: Watch the Magic! ✨

You'll see output like:

```
╔══════════════════════════════════════════════════════╗
║       AML FRAUD DETECTION PIPELINE                   ║
╚══════════════════════════════════════════════════════╝

📊 STAGE 1: DATA GENERATION
─────────────────────────────
🎲 Generating synthetic data...
   ✓ Created 5000 customer profiles
   ✓ Generated 85000 normal transactions
   ✓ Generated 15000 suspicious transactions

🔧 STAGE 2: FEATURE ENGINEERING
─────────────────────────────────
⏰ Creating time-based features...
   ✓ Created time-based features
👥 Creating customer features...
   ✓ Created customer aggregated features
📝 Processing text descriptions...
   ✓ Created text features (TF-IDF)

🤖 STAGE 3: MODEL TRAINING
────────────────────────────
1️⃣  K-MEANS CLUSTERING
   Training K-Means with k=5...
      Cost: 1234.56 | Time: 2.3s
   Training K-Means with k=10...
      Cost: 987.65 | Time: 3.1s
   ✅ Best K-Means: k=10, cost=987.65

...

📊 STAGE 4: MODEL EVALUATION
──────────────────────────────
📈 Evaluating KMEANS...
   Precision: 0.820
   Recall: 0.840
   F1-Score: 0.830

...

🏆 STAGE 5: MODEL SELECTION
─────────────────────────────
   🏆 Best model: ensemble
   📊 F1-Score: 0.8500
   ✓ Model registered: aml_fraud_ensemble_v1.0
   ✓ Model promoted to production

✅ PIPELINE COMPLETE!
```

### Step 5: Check Results

```bash
# Read the executive summary
cat results/executive_summary.txt

# Read detailed evaluation
cat results/evaluation_report.txt

# Check logs
cat logs/aml_fraud_detection.log
```

---

## 📈 Interpreting Results

### Understanding the Leaderboard

```
Rank  Algorithm              F1      Precision  Recall
──────────────────────────────────────────────────────
1     ensemble               0.850   0.820      0.880
2     gmm                    0.840   0.810      0.870
3     risk_score_clustering  0.830   0.850      0.810
```

**What this tells us:**

**Ensemble (Rank 1):**
- F1 = 0.850: Excellent overall performance
- Precision = 0.820: 82% of alerts are real fraud
- Recall = 0.880: Catches 88% of all fraud
- **Verdict:** Best all-around performer!

**Risk-Score (Rank 3):**
- Precision = 0.850: Highest precision!
- Recall = 0.810: Slightly lower recall
- **Verdict:** Fewest false alarms, but misses some fraud

### Choosing the Right Model

**For Your Organization:**

**If false alarms are expensive:**
→ Choose **Risk-Score** (highest precision)
→ You'll have fewer false alarms but might miss some fraud

**If missing fraud is unacceptable:**
→ Choose **Ensemble** or **GMM** (highest recall)
→ You'll catch more fraud but have more false alarms

**If you need explainability:**
→ Choose **Risk-Score** or **Z-Score**
→ You can explain WHY a transaction was flagged

### Real-World Example

**Your Results:**
```
Ensemble Algorithm:
- Precision: 0.820 (82%)
- Recall: 0.880 (88%)
- 15,000 transactions flagged out of 100,000
```

**Translation:**
```
Out of 15,000 alerts:
- 12,300 are real fraud (0.82 * 15,000)
- 2,700 are false alarms

Out of 15,000 actual frauds in data:
- 13,200 were caught (0.88 * 15,000)
- 1,800 were missed
```

**Business Impact:**
```
Cost of false alarms: 2,700 * $50 = $135,000
Cost of missed fraud: 1,800 * $1,000 = $1,800,000
Total cost: $1,935,000

Without the system:
Cost of missing all fraud: 15,000 * $1,000 = $15,000,000

Savings: $13,065,000! 💰
```

---

## 🎯 Next Steps

### For Beginners

1. **✅ Run the default pipeline**
   - Understand what each stage does
   - Read the code comments

2. **🔧 Experiment with config.yaml**
   - Try different values of K
   - Enable/disable algorithms
   - Change fraud ratio

3. **📚 Read the code**
   - Start with `data_generator.py`
   - Then `feature_engineering.py`
   - Understand one algorithm at a time

### For Intermediate Users

1. **📊 Add custom features**
   - Create your own fraud indicators
   - Test if they improve performance

2. **🤖 Tune algorithms**
   - Experiment with hyperparameters
   - Try different threshold values

3. **📈 Analyze results**
   - Which features are most important?
   - Which fraud patterns are hardest to detect?

### For Advanced Users

1. **🔬 Add new algorithms**
   - Implement DBSCAN or HDBSCAN
   - Try deep learning approaches
   - Create custom ensemble methods

2. **🚀 Deploy to production**
   - Set up on a Spark cluster
   - Implement real-time scoring
   - Add A/B testing

3. **📡 Enhance monitoring**
   - Build dashboards
   - Set up alerting
   - Implement automated retraining

---

## 💡 Key Takeaways

1. **AML fraud detection is a cat-and-mouse game**
   - Criminals adapt, so models must too
   - No single algorithm catches everything
   - Ensemble methods work best

2. **Features matter more than algorithms**
   - Good features make any algorithm work better
   - Domain knowledge is crucial
   - Text + tabular data is powerful

3. **MLOps is essential for production**
   - Track experiments
   - Version models
   - Monitor performance
   - Detect drift

4. **Business context matters**
   - Balance precision vs. recall based on costs
   - Consider interpretability requirements
   - Think about investigator workload

5. **Continuous improvement**
   - Models degrade over time
   - Fraud patterns evolve
   - Regular retraining is necessary

---

## 🎓 Additional Resources

### Learning More About:

**PySpark:**
- [PySpark Official Docs](https://spark.apache.org/docs/latest/api/python/)
- [PySpark ML Guide](https://spark.apache.org/docs/latest/ml-guide.html)

**AML & Financial Crime:**
- [FATF Guidelines](https://www.fatf-gafi.org/)
- [FinCEN AML Resources](https://www.fincen.gov/)

**Machine Learning:**
- [scikit-learn docs](https://scikit-learn.org/) (similar concepts)
- [Clustering Algorithms Guide](https://scikit-learn.org/stable/modules/clustering.html)

**MLOps:**
- [MLOps.org](https://ml-ops.org/)
- [Google MLOps Guide](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)

---

## 🤝 Getting Help

**Stuck? Here's what to do:**

1. **Check the logs**
   ```bash
   cat logs/aml_fraud_detection.log
   ```

2. **Read error messages carefully**
   - They usually tell you exactly what's wrong!

3. **Start small**
   - Reduce `num_transactions` to 1,000
   - Test with fewer algorithms
   - Disable complex features

4. **Use print statements**
   - Add `df.show()` to see data
   - Add `print(metrics)` to debug

5. **Ask for help**
   - Open an issue on GitHub
   - Include error messages and logs

---

## 🎉 Congratulations!

You've completed the tutorial! You now understand:

✅ What AML fraud detection is and why it matters
✅ How to engineer features from raw banking data
✅ How 12+ different algorithms detect fraud
✅ How to evaluate and compare models
✅ How MLOps makes systems production-ready
✅ How to run experiments and interpret results

**Now go catch some fraudsters! 🕵️‍♂️💰**

---

**Remember:** This is a learning project with synthetic data. Real production AML systems require:
- Regulatory compliance
- Privacy controls
- Expert validation
- Continuous monitoring
- Regular audits

But the concepts you've learned here form the foundation!

**Happy Learning! 🚀**
