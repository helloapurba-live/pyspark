# 🎓 Complete Tutorial: Banking AML Fraud Detection

**Welcome! Let's build an ML system together, step by step!**

---

## 📖 Table of Contents

1. [Getting Started](#getting-started)
2. [Understanding the Problem](#understanding-the-problem)
3. [Project Walkthrough](#project-walkthrough)
4. [Running Your First Pipeline](#running-your-first-pipeline)
5. [Understanding the Results](#understanding-the-results)
6. [Customizing the System](#customizing-the-system)
7. [Troubleshooting](#troubleshooting)
8. [Next Steps](#next-steps)

---

## 🚀 Getting Started

### Prerequisites

Before we begin, make sure you have:
- **Python 3.8+** installed ([Download](https://www.python.org/downloads/))
- **4GB+ RAM** (8GB recommended)
- **2GB free disk space**
- **Text editor** (VS Code, PyCharm, or even Notepad++)

### Quick Installation

```bash
# Clone or navigate to the project directory
cd pyspark

# Create a virtual environment (isolated Python environment)
python3 -m venv venv

# Activate it
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**What just happened?**
- Created a "bubble" (virtual environment) for our project
- Installed all necessary Python packages
- Ready to run our code!

---

## 🎯 Understanding the Problem

### What is Anti-Money Laundering (AML)?

Imagine you're a detective at a bank. Your job is to catch criminals trying to "clean" their dirty money. Here's how they do it:

#### 1. 💰 **Structuring** (Smurfing)
```
Criminal has: $50,000 from illegal activities
Problem: Banks report transactions > $10,000
Solution for criminal: Deposit $9,000 five times (under the radar!)
Our job: Detect this pattern!
```

#### 2. 🌍 **Layering**
```
Money flow:
Criminal account → Offshore account (Cayman Islands)
                 → Swiss bank
                 → Back to criminal (now "clean")
Our job: Follow the trail!
```

#### 3. 🎭 **Shell Company**
```
Criminal creates: "ABC Consulting LLC" (fake business)
Transaction: $100,000 to ABC for "consulting services"
Reality: No actual consulting happened
Our job: Spot fake businesses!
```

#### 4. 🔄 **Round Tripping**
```
Money goes out → Comes back through complex path
Looks like: Investment return or loan repayment
Reality: Money laundering
Our job: Identify circular patterns!
```

### Why Machine Learning?

**Traditional Approach:**
- Manual rules: "Flag if amount > $10,000 AND international"
- Problems:
  - Too many false alarms (annoyed customers)
  - Criminals adapt and evade rules
  - Can't catch complex patterns

**ML Approach:**
- Learn patterns from examples
- Adapt as fraud evolves
- Find complex relationships humans miss
- Balance accuracy with fewer false alarms

---

## 📂 Project Walkthrough

Let's explore what each file does:

### Directory Structure

```
pyspark/
│
├── 📄 README.md                  ← Start here! Project overview
├── 📄 TUTORIAL.md                ← You are here!
├── 📄 requirements.txt           ← Python packages needed
├── 📄 run_pipeline.sh            ← One-click run everything
│
├── 📁 src/                       ← All source code
│   ├── 📄 main_pipeline.py       ← 🌟 Main orchestrator
│   │
│   ├── 📁 data/                  ← Data handling
│   │   └── 📄 generate_data.py   ← Creates synthetic transactions
│   │
│   ├── 📁 features/              ← Feature engineering
│   │   └── 📄 feature_engineering.py  ← Transforms raw data
│   │
│   ├── 📁 models/                ← ML algorithms
│   │   └── 📄 model_trainer.py   ← Trains 10+ models
│   │
│   ├── 📁 evaluation/            ← Model assessment
│   │   └── 📄 model_evaluator.py ← Measures performance
│   │
│   └── 📁 deployment/            ← Using trained models
│       └── 📄 model_inference.py ← Makes predictions
│
├── 📁 config/                    ← Configuration
│   └── 📄 config.yaml            ← All settings
│
├── 📁 data/                      ← Data storage
│   ├── 📁 raw/                   ← Original data
│   ├── 📁 processed/             ← Transformed data
│   ├── 📁 models/                ← Saved models
│   └── 📁 reports/               ← Results and metrics
│
├── 📁 notebooks/                 ← Jupyter notebooks
└── 📁 logs/                      ← Execution logs
```

### Key Files Explained

#### 1. `src/data/generate_data.py` - The Data Factory

**What it does:**
- Creates 50,000 fake but realistic banking transactions
- Generates 5 types: 1 legitimate + 4 fraud types
- Adds realistic details (names, amounts, descriptions)

**Run it:**
```bash
python src/data/generate_data.py
```

**Output:**
- `data/raw/transactions.csv` - Human-readable
- `data/raw/transactions.parquet` - Efficient format for PySpark
- `data/raw/metadata.json` - Data statistics

**Look inside:**
```python
# Example transaction (legitimate)
{
  "transaction_id": "TXN_00001234",
  "customer_id": "CUST_000567",
  "amount": 87.43,
  "description": "Grocery store purchase",
  "category": "LEGITIMATE",
  "timestamp": "2024-03-15 14:32:18"
}

# Example transaction (fraud - structuring)
{
  "transaction_id": "TXN_00005678",
  "customer_id": "CUST_000234",
  "amount": 9,847.00,  # Just under $10k!
  "description": "Cash deposit below threshold",
  "category": "STRUCTURING",
  "timestamp": "2024-03-15 22:45:33"  # Late night!
}
```

#### 2. `src/features/feature_engineering.py` - The Transformer

**What it does:**
- Converts raw data into ML-ready features
- Handles both numbers (amount, time) and text (descriptions)
- Scales and encodes everything

**Example transformation:**

**Before (Raw):**
```
Amount: $9,847.23
Description: "Cash deposit below threshold"
Time: 2024-03-15 22:45:33
```

**After (Features):**
```
- amount: 9847.23
- amount_log: 9.19 (log scale)
- is_near_threshold: 1 (yes!)
- is_large_amount: 0
- is_round_amount: 1
- hour: 22 (late night)
- is_night: 1
- description_length: 29
- has_cash_keyword: 1
- ... + 50 more features from text processing
```

#### 3. `src/models/model_trainer.py` - The Teacher

**What it does:**
- Trains 10+ different ML algorithms
- Each algorithm learns patterns differently
- Saves all models for later use

**The 10+ Algorithms:**

| Algorithm | Strength | Speed | Complexity |
|-----------|----------|-------|------------|
| Logistic Regression | Fast baseline | ⚡⚡⚡ | Simple |
| Decision Tree | Interpretable | ⚡⚡ | Medium |
| Random Forest | Robust, accurate | ⚡ | Medium |
| Gradient Boosting | High performance | ⚡ | Complex |
| Naive Bayes | Fast, good with text | ⚡⚡⚡ | Simple |
| Linear SVC | High-dimensional data | ⚡⚡ | Medium |
| Neural Network | Complex patterns | ⚡ | Complex |
| + More variants | Different configurations | - | - |

#### 4. `src/evaluation/model_evaluator.py` - The Judge

**What it does:**
- Tests all models on unseen data
- Calculates performance metrics
- Creates comparison reports

**Key Metrics:**

- **Accuracy**: How often is the model right?
  - Formula: (Correct predictions) / (Total predictions)
  - Example: 90% accuracy = right 9 out of 10 times

- **Precision**: When model says "fraud," how often is it right?
  - Formula: (True frauds caught) / (All flagged as fraud)
  - Example: 85% precision = 85 out of 100 flags are real fraud

- **Recall**: Of all real frauds, how many did we catch?
  - Formula: (True frauds caught) / (All real frauds)
  - Example: 92% recall = caught 92 out of 100 real frauds
  - **Most important for AML!** Missing fraud is costly!

- **F1-Score**: Balance of precision and recall
  - Formula: 2 × (Precision × Recall) / (Precision + Recall)
  - Example: 88% F1 = good overall performance

#### 5. `src/main_pipeline.py` - The Conductor

**What it does:**
- Orchestrates everything
- Runs steps in correct order
- Handles errors gracefully
- Generates final report

**The Pipeline Flow:**
```
Start
  ↓
Load Data (50,000 transactions)
  ↓
Engineer Features (tabular + text)
  ↓
Split Data (70% train, 15% validation, 15% test)
  ↓
Train Models (10+ algorithms) ← Takes time!
  ↓
Evaluate Models (on test set)
  ↓
Compare Performance (find best model)
  ↓
Save Everything (models, reports, metrics)
  ↓
Generate Report
  ↓
Done! 🎉
```

---

## ⚡ Running Your First Pipeline

### Method 1: Automated Script (Easiest!)

```bash
# Make script executable
chmod +x run_pipeline.sh

# Run everything!
./run_pipeline.sh
```

**What happens:**
1. ✅ Checks Python installation
2. ✅ Creates virtual environment
3. ✅ Installs dependencies
4. ✅ Generates data
5. ✅ Runs ML pipeline
6. ✅ Shows results

**Time:** 5-15 minutes (depending on your computer)

### Method 2: Manual Steps (More control)

**Step 1: Generate Data**
```bash
python src/data/generate_data.py
```

**Step 2: Run Pipeline**
```bash
python src/main_pipeline.py
```

**Step 3: View Results**
```bash
# View comparison
cat data/reports/model_comparison.csv

# View detailed report
cat data/reports/evaluation_report.md
```

---

## 📊 Understanding the Results

### Where to Find Results

```
data/reports/
├── model_comparison.csv         ← Quick comparison table
├── evaluation_report.md         ← Detailed report
└── evaluation_results.json      ← Raw metrics (for programs)
```

### Reading the Comparison Table

```csv
Rank,Model,Accuracy,Precision,Recall,F1-Score
1,Random_Forest_Tuned,0.9234,0.9187,0.9201,0.9194
2,Gradient_Boosted_Trees,0.9156,0.9098,0.9145,0.9121
3,Random_Forest_Large,0.9089,0.9034,0.9076,0.9055
...
```

**How to read:**
- **Rank 1 = Best model!**
- Look at F1-Score (balanced metric)
- For AML, also check Recall (don't miss frauds!)

### Understanding a Confusion Matrix

```
                    Predicted
                L    S    L    SC   RT
Actual    L  [8900  45   32   18   5]
          S  [38   892  15   10   5]
          L  [25   12   885  20   8]
          SC [15   8    18   924  10]
          RT [12   10   15   8    930]
```

**Reading:**
- Diagonal (bold) = Correct predictions
- L = Legitimate, S = Structuring, L = Layering, SC = Shell Company, RT = Round Tripping
- Example: Row 1, Col 1 = 8900 legitimate transactions correctly identified

### What Makes a Good Model?

**For AML Detection:**

✅ **Good Model:**
- Recall > 90% (catch most frauds)
- Precision > 80% (few false alarms)
- F1-Score > 85% (balanced)
- Fast inference (< 100ms per transaction)

❌ **Bad Model:**
- Recall < 70% (missing too much fraud!)
- Precision < 50% (too many false alarms)
- F1-Score < 60% (poor overall)

---

## 🎨 Customizing the System

### Adjusting Settings

Edit `config/config.yaml`:

```yaml
# Generate more data
data:
  generation:
    num_transactions: 100000  # More data = better models

# Tune Random Forest
models:
  random_forest:
    num_trees: 200  # More trees = better accuracy (but slower)
    max_depth: 15   # Deeper = more complex patterns
```

### Trying Different Algorithms

Edit `src/main_pipeline.py`:

```python
# Add a new model variant
self.train_random_forest(
    train_df,
    "My_Custom_RF",
    num_trees=150,
    max_depth=20
)
```

### Creating Custom Features

Edit `src/features/feature_engineering.py`:

```python
def create_custom_features(self, df):
    # Add velocity: transactions per day
    df = df.withColumn('daily_transaction_count', ...)

    # Add amount ratio: current transaction vs average
    df = df.withColumn('amount_vs_average', ...)

    return df
```

---

## 🔧 Troubleshooting

### Common Issues

#### Issue 1: "Data not found"
```
Error: FileNotFoundError: data/raw/transactions.parquet
```

**Solution:**
```bash
python src/data/generate_data.py
```

#### Issue 2: "Out of memory"
```
Error: java.lang.OutOfMemoryError
```

**Solution:**
Edit `config/config.yaml`:
```yaml
spark:
  config:
    "spark.driver.memory": "2g"  # Reduce if needed
```

Or generate less data:
```yaml
data:
  generation:
    num_transactions: 10000  # Smaller dataset
```

#### Issue 3: "Module not found"
```
Error: ModuleNotFoundError: No module named 'pyspark'
```

**Solution:**
```bash
pip install -r requirements.txt
```

#### Issue 4: Pipeline is too slow
```
Taking forever to train models...
```

**Solution:**
- Reduce number of models (comment out some in main_pipeline.py)
- Reduce model complexity (fewer trees, less depth)
- Use smaller dataset for testing

---

## 🎯 Next Steps

### Beginner Path

1. ✅ Run the pipeline successfully
2. 📊 Understand the results
3. 🎨 Modify configuration (change parameters)
4. 🔄 Run again and compare results
5. 📚 Read code comments to understand each part

### Intermediate Path

1. ✅ All beginner steps
2. 🎨 Create custom features
3. 🤖 Add new ML algorithms
4. 📊 Implement cross-validation
5. 🚀 Deploy a model (use inference.py)

### Advanced Path

1. ✅ All intermediate steps
2. 🔬 Hyperparameter optimization (Grid Search)
3. 🎪 Implement ensemble methods (stacking, blending)
4. 📈 Add SHAP for explainability
5. 🌐 Create REST API for model serving
6. 📊 Implement monitoring and drift detection
7. 🏗️ Deploy to production (AWS, Azure, GCP)

### Learning Resources

**PySpark:**
- [Official PySpark Documentation](https://spark.apache.org/docs/latest/api/python/)
- [PySpark Tutorial](https://sparkbyexamples.com/)

**Machine Learning:**
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Fast.ai Course](https://course.fast.ai/) (Free!)
- [Coursera ML Course](https://www.coursera.org/learn/machine-learning)

**MLOps:**
- [MLflow Documentation](https://mlflow.org/)
- [Made With ML](https://madewithml.com/)

**AML/Fraud Detection:**
- [Kaggle Credit Card Fraud](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- [Fraud Detection Papers](https://paperswithcode.com/task/fraud-detection)

---

## 💡 Pro Tips

### 1. Start Small
Don't generate 1 million transactions immediately! Start with 10,000, understand everything, then scale up.

### 2. Use Notebooks
Jupyter notebooks are great for exploration:
```bash
jupyter notebook
# Create new notebook in notebooks/ directory
```

### 3. Track Experiments
Keep a log:
```
Experiment 1: Random Forest with 50 trees → F1 = 0.87
Experiment 2: Random Forest with 100 trees → F1 = 0.91 (better!)
Experiment 3: Added new feature (velocity) → F1 = 0.93 (even better!)
```

### 4. Understand, Don't Just Run
Read the code comments! They explain WHY we do things, not just WHAT.

### 5. Make It Your Own
- Add your own features
- Try different algorithms
- Experiment with parameters
- Break things and fix them (best way to learn!)

---

## 🤝 Getting Help

**Found a bug?**
- Check logs in `logs/`
- Read error messages carefully
- Google the error (StackOverflow is your friend!)

**Want to contribute?**
- Fix bugs
- Add features
- Improve documentation
- Share your learnings!

**Have questions?**
- Re-read this tutorial
- Check code comments
- Consult documentation
- Experiment and learn!

---

## 🎉 Congratulations!

You now have:
- ✅ A complete AML fraud detection system
- ✅ 10+ trained ML models
- ✅ Understanding of end-to-end ML pipeline
- ✅ MLOps best practices
- ✅ Deployable models

**You're now ready to:**
- Build your own ML projects
- Understand production ML systems
- Contribute to ML projects
- Continue learning advanced topics

---

## 📜 Appendix: Command Cheat Sheet

```bash
# Data generation
python src/data/generate_data.py

# Run complete pipeline
python src/main_pipeline.py

# Quick start (automated)
./run_pipeline.sh

# Feature engineering only
python src/features/feature_engineering.py

# Model training only
python src/models/model_trainer.py

# Model inference (predictions)
python src/deployment/model_inference.py

# View results
cat data/reports/model_comparison.csv
cat data/reports/evaluation_report.md

# Check logs
tail -f logs/pipeline.log
```

---

**Happy Learning! 🚀**

*Remember: The best way to learn is by doing. Experiment, break things, fix them, and learn!*

---

*"Give a person a model and they can make predictions for a day. Teach them ML and they can build systems for a lifetime."* 😊
