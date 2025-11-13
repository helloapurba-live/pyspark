# 🏦 AML Fraud Detection - Complete MLOps Pipeline with PySpark

## 📚 Table of Contents
- [Overview](#overview)
- [What You'll Learn](#what-youll-learn)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Guide](#detailed-guide)
- [Understanding the Code](#understanding-the-code)
- [MLOps Best Practices](#mlops-best-practices)
- [Results and Evaluation](#results-and-evaluation)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

This is a **complete, production-ready MLOps pipeline** for Anti-Money Laundering (AML) fraud detection. It's designed as an educational project that teaches you how to build real-world machine learning systems from scratch.

### What Makes This Special?

✨ **Complete End-to-End Pipeline**: From raw data to deployed model
🤖 **10+ Machine Learning Algorithms**: Compare different approaches
📊 **Mixed Data Types**: Works with both tabular (numbers) and text data
🔬 **Full MLOps**: Experiment tracking, model registry, monitoring
📖 **Beginner-Friendly**: Extensive comments explaining every concept
🎓 **Learn by Doing**: Follows teaching approach, not just code

### The Problem We're Solving

**Banking fraud costs billions annually.** This system:
- 🔍 Analyzes banking transactions in real-time
- 🚨 Detects fraudulent patterns using ML
- 💡 Explains WHY a transaction is flagged
- 📈 Improves over time with new data

---

## 🎓 What You'll Learn

### 1. **Data Engineering**
- Generating realistic synthetic data
- Handling mixed data types (numbers + text)
- Creating meaningful features from raw data
- Data validation and quality checks

### 2. **Machine Learning**
- **10+ Algorithms**:
  - Logistic Regression (baseline)
  - Decision Trees (interpretable)
  - Random Forests (ensemble)
  - Gradient Boosting (GBT, XGBoost, LightGBM, CatBoost)
  - Neural Networks (MLP)
  - Support Vector Machines (SVM)
  - Naive Bayes (probabilistic)
  - Ensemble methods (Stacking, Voting)

### 3. **Feature Engineering**
- Text vectorization (TF-IDF)
- Categorical encoding (One-Hot)
- Numerical scaling (Standardization)
- Advanced feature creation
- Feature selection

### 4. **MLOps (Machine Learning Operations)**
- Experiment tracking (MLflow)
- Model versioning and registry
- Performance monitoring
- Drift detection
- Automated retraining pipelines

### 5. **Model Evaluation**
- Metrics for imbalanced data
- Confusion matrices
- ROC and Precision-Recall curves
- Feature importance analysis
- Business metrics (cost of false positives/negatives)

### 6. **Deployment**
- Batch prediction
- Real-time inference
- Model serving
- API endpoints

### 7. **PySpark**
- Distributed data processing
- ML pipelines
- Performance optimization
- Working with big data

---

## 📁 Project Structure

```
pyspark/
│
├── config/                          # Configuration files
│   └── pipeline_config.yaml         # Main pipeline configuration
│
├── data/                            # Data storage
│   ├── raw/                         # Raw generated data
│   ├── processed/                   # Processed features
│   └── checkpoints/                 # Pipeline checkpoints
│
├── src/                             # Source code
│   ├── data_generation/             # Data generation modules
│   │   └── generate_banking_data.py # Creates synthetic banking data
│   │
│   ├── feature_engineering/         # Feature engineering
│   │   └── feature_pipeline.py      # Feature transformation pipeline
│   │
│   ├── models/                      # ML models
│   │   ├── ml_algorithms.py         # 10+ ML algorithms
│   │   └── model_evaluation.py      # Model evaluation & comparison
│   │
│   ├── mlops/                       # MLOps components
│   │   └── experiment_tracking.py   # Experiment tracking & model registry
│   │
│   ├── deployment/                  # Deployment modules
│   │   └── model_serving.py         # Model serving & inference
│   │
│   └── monitoring/                  # Monitoring modules
│       └── model_monitoring.py      # Performance & drift monitoring
│
├── experiments/                     # MLflow experiment data
├── model_registry/                  # Saved models with versions
├── logs/                            # Application logs
├── reports/                         # Generated reports and plots
│   └── figures/                     # Evaluation plots
│
├── main_pipeline.py                 # 🚀 MAIN SCRIPT - Run this!
├── requirements.txt                 # Python dependencies
└── README.md                        # This file!
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Java 8 or 11 (required for PySpark)
- 8GB+ RAM recommended

### Step 1: Clone or Download
```bash
cd /home/user/pyspark
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation
```bash
python -c "import pyspark; print(f'PySpark version: {pyspark.__version__}')"
```

---

## 🚀 Quick Start

### Run the Complete Pipeline

```bash
python main_pipeline.py
```

This single command will:
1. ✅ Generate 100,000 synthetic banking transactions
2. ✅ Engineer features (tabular + text)
3. ✅ Train 10+ ML models
4. ✅ Evaluate and compare all models
5. ✅ Select the best model
6. ✅ Save everything (models, experiments, reports)

**Expected runtime:** 10-30 minutes (depending on your hardware)

### What You'll See

The pipeline will print detailed logs showing:
- 📊 Data generation progress
- 🔧 Feature engineering steps
- 🤖 Training progress for each model
- 📈 Evaluation metrics
- 🏆 Best model selection

### Results Location

After running, you'll find:
- **Models**: `./model_registry/`
- **Experiments**: `./experiments/`
- **Plots**: `./reports/figures/`
- **Logs**: `./logs/pipeline.log`

---

## 📖 Detailed Guide

### Understanding Banking Fraud Detection

#### What is AML (Anti-Money Laundering)?
Banks must detect when criminals try to:
- Transfer stolen money
- Hide illegal income
- Finance terrorism
- Launder money through fake transactions

#### The Challenge
- **Rare Events**: Only ~2% of transactions are fraudulent
- **Costly Mistakes**:
  - False Negatives (missing fraud) = Financial loss
  - False Positives (false alarms) = Annoying legitimate customers
- **Evolving Patterns**: Fraudsters constantly adapt
- **Real-Time**: Need decisions in milliseconds

#### Our Solution
Use machine learning to:
1. Learn patterns from historical fraud cases
2. Predict fraud probability for new transactions
3. Flag suspicious transactions for review
4. Adapt as fraud patterns change

---

### Understanding the Data

#### Tabular Features (Numbers & Categories)
```python
# Transaction characteristics
- transaction_amount: How much money?
- transaction_hour: What time of day?
- account_age_days: How old is the account?
- num_transactions_24h: How active recently?
- credit_score: Customer creditworthiness
- source_country / destination_country: International?

# Derived features
- is_weekend: Weekend transactions more risky
- is_night: Late-night transactions suspicious
- amount_to_avg_ratio: Unusual amount for this customer?
- transaction_velocity: How fast is account transacting?
```

#### Text Features
```python
# Natural language data
- transaction_description: "Wire transfer to overseas account"
- merchant_category: "Casino", "Charity", "Electronics"
- customer_notes: "Disputed transaction"

# Converted to numbers using TF-IDF
- Captures important words
- "urgent", "overseas", "casino" → higher fraud signal
```

---

### Understanding the Algorithms

#### 1. **Logistic Regression** (Baseline)
**What**: Linear model that outputs probability

**How it works**: Finds a line that separates fraud from legitimate

**Pros**: Fast, interpretable, good baseline

**Cons**: Assumes linear relationships

**When to use**: Starting point, when you need interpretability

---

#### 2. **Decision Trees**
**What**: Series of yes/no questions

**How it works**:
```
Is amount > $5000?
  ├─ YES → Is at night?
  │         ├─ YES → Is international?
  │         │         ├─ YES → FRAUD!
  │         │         └─ NO → Check credit score...
  │         └─ NO → LEGITIMATE
  └─ NO → LEGITIMATE
```

**Pros**: Very interpretable, handles non-linear patterns

**Cons**: Can overfit, unstable

---

#### 3. **Random Forest** (Recommended!)
**What**: 100+ decision trees voting together

**How it works**:
- Train many trees on random subsets of data
- Each tree votes
- Majority wins!

**Pros**: Very accurate, robust, handles complex patterns

**Cons**: Less interpretable, slower

**Why use**: Often the best performer for tabular data

---

#### 4. **Gradient Boosted Trees (GBT, XGBoost, LightGBM)**
**What**: Sequential learning - each tree fixes previous mistakes

**How it works**:
- Tree 1 learns from data
- Tree 2 focuses on Tree 1's errors
- Tree 3 focuses on Tree 2's errors
- Combine all trees

**Pros**: Often highest accuracy, handles complex patterns

**Cons**: Can overfit, requires tuning

**Why use**: Winning Kaggle competitions, maximum performance

---

#### 5. **Neural Networks (MLP)**
**What**: Brain-inspired network of neurons

**How it works**:
```
Input Layer → Hidden Layers → Output Layer
[100 nodes] → [50 nodes] → [2 nodes: Fraud/Legit]
```

**Pros**: Can learn any pattern, powerful

**Cons**: Needs more data, "black box", slower

**Why use**: Complex non-linear relationships

---

### Understanding MLOps

#### Why MLOps?

Traditional ML: "I trained a model and got 95% accuracy!"

Production ML:
- Which model version is this?
- What data was it trained on?
- Can you reproduce these results?
- Is it still accurate a month later?
- How do you deploy updates?

**MLOps solves these problems!**

---

#### Experiment Tracking (MLflow)

**Problem**: You train 100 models - which was best?

**Solution**: Log everything automatically
```python
mlflow.log_param("learning_rate", 0.01)
mlflow.log_metric("f1_score", 0.87)
mlflow.log_model(model, "random_forest_v1")
```

**Benefits**:
- Compare models fairly
- Reproduce results
- Share with team
- Track what works

---

#### Model Registry

**Problem**: Multiple model versions in production - which is which?

**Solution**: Version control for models
```python
registry.register_model(
    model=trained_model,
    version="1.0.0",
    stage="production"
)
```

**Benefits**:
- Track model versions
- Promote/rollback models
- Know what's in production
- Audit trail

---

#### Monitoring

**Problem**: Model works great in training, degrades in production

**Solution**: Monitor performance and data drift
```python
monitor.detect_data_drift(new_data)
monitor.monitor_performance(predictions, true_labels)
```

**Alerts**:
- ⚠️ Data drift detected → Patterns changing
- ⚠️ Performance drop → Model degrading
- ⚠️ System issues → Service unhealthy

---

### Understanding Evaluation Metrics

#### For Fraud Detection, We Care About:

#### 1. **Accuracy**
```
Accuracy = (Correct Predictions) / (Total Predictions)
```
**Problem**: Misleading for imbalanced data!

Example: If 98% transactions are legitimate, a model that always predicts "legitimate" gets 98% accuracy but catches ZERO fraud!

---

#### 2. **Precision** (Avoid False Alarms)
```
Precision = True Positives / (True Positives + False Positives)
```
**Question**: "When we flag fraud, how often are we right?"

**Business impact**: High false positives = angry customers

---

#### 3. **Recall** (Catch the Fraudsters!)
```
Recall = True Positives / (True Positives + False Negatives)
```
**Question**: "Of all actual fraud, how much did we catch?"

**Business impact**: Low recall = fraud slips through

---

#### 4. **F1-Score** (The Sweet Spot)
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```
**The balance**: Catch fraud WITHOUT annoying customers

**Why we use it**: Best single metric for fraud detection

---

#### 5. **AUC-ROC** (Overall Ability)
- Area Under ROC Curve
- 0.5 = Random guessing
- 1.0 = Perfect classification
- 0.8+ = Good model

---

#### 6. **AUC-PR** (Better for Imbalanced Data)
- Area Under Precision-Recall Curve
- Better than AUC-ROC for rare events like fraud

---

#### Confusion Matrix

```
                  Predicted
                Legit  Fraud
Actual  Legit    TN     FP    ← False Positive = False Alarm
        Fraud    FN     TP    ← False Negative = Missed Fraud
                 ↑
            False Negative
```

**Goal**: Maximize TP (catch fraud) and TN (approve legit)
**Minimize**: FP (false alarms) and FN (missed fraud)

---

## 🔬 MLOps Best Practices

### 1. Version Everything
- ✅ Code (Git)
- ✅ Data (checksums, versions)
- ✅ Models (MLflow registry)
- ✅ Configurations (YAML files)

### 2. Make It Reproducible
- ✅ Set random seeds
- ✅ Log all parameters
- ✅ Save data snapshots
- ✅ Document environment

### 3. Monitor in Production
- ✅ Data drift detection
- ✅ Performance tracking
- ✅ Alert on degradation
- ✅ Automated retraining

### 4. Test Everything
- ✅ Unit tests for functions
- ✅ Integration tests for pipeline
- ✅ Data quality tests
- ✅ Model performance tests

### 5. Automate
- ✅ Automated training pipelines
- ✅ Automated deployment
- ✅ Automated monitoring
- ✅ Automated alerts

---

## 📊 Results and Evaluation

After running the pipeline, check:

### 1. **Model Comparison**
Open `./logs/pipeline.log` to see leaderboard:
```
MODEL COMPARISON LEADERBOARD
============================================================
Model                    F1     AUC-ROC  Precision  Recall
Random Forest         0.8745    0.9321     0.8934  0.8567
Gradient Boosted      0.8698    0.9287     0.8812  0.8589
XGBoost               0.8654    0.9245     0.8756  0.8556
...
```

### 2. **Best Model Performance**
```
WINNER: Random Forest
  F1-Score:  0.8745
  AUC-ROC:   0.9321
  Precision: 0.8934
  Recall:    0.8567
```

### 3. **Confusion Matrix**
Check `./reports/figures/confusion_matrix_*.png`:
- True Positives: Fraud caught ✅
- True Negatives: Legit approved ✅
- False Positives: False alarms ⚠️
- False Negatives: Fraud missed ❌

### 4. **Feature Importance**
See which features matter most:
- Transaction amount
- Account age
- Transaction time
- International transfers
- Text signals (keywords)

---

## 🐛 Troubleshooting

### Common Issues

#### 1. **Memory Error**
```
OutOfMemoryError: Java heap space
```
**Solution**: Reduce data size in config:
```yaml
data:
  num_transactions: 50000  # Reduce from 100000
```

#### 2. **Spark Not Starting**
```
Exception: Java gateway process exited
```
**Solution**: Check Java installation:
```bash
java -version  # Should show Java 8 or 11
```

#### 3. **Module Not Found**
```
ModuleNotFoundError: No module named 'pyspark'
```
**Solution**: Install requirements:
```bash
pip install -r requirements.txt
```

#### 4. **Slow Performance**
**Solution**: Reduce data or increase Spark memory:
```yaml
spark:
  executor_memory: "8g"  # Increase if you have RAM
```

---

## 🎓 Learning Path

### Beginner → Intermediate → Advanced

#### Week 1: Understand the Code
- [ ] Read all code comments
- [ ] Run the pipeline
- [ ] Understand each component
- [ ] Modify configuration

#### Week 2: Experiment
- [ ] Try different algorithms
- [ ] Adjust hyperparameters
- [ ] Add new features
- [ ] Compare results

#### Week 3: Extend
- [ ] Add new ML algorithms
- [ ] Implement ensemble methods
- [ ] Add feature selection
- [ ] Improve evaluation

#### Week 4: Production
- [ ] Set up monitoring
- [ ] Deploy as API
- [ ] Implement CI/CD
- [ ] Add automated retraining

---

## 📚 Additional Resources

### Learning Materials
- **PySpark**: [Official Documentation](https://spark.apache.org/docs/latest/api/python/)
- **MLflow**: [Tracking Guide](https://mlflow.org/docs/latest/tracking.html)
- **Fraud Detection**: [Research Papers](https://paperswithcode.com/task/fraud-detection)

### Next Steps
1. **Deploy as API**: Flask/FastAPI endpoint
2. **Real-Time Processing**: Kafka + Spark Streaming
3. **AutoML**: Automated hyperparameter tuning
4. **Deep Learning**: LSTM for sequential patterns
5. **Explainability**: SHAP for model interpretation

---

## 🤝 Contributing

Have improvements? Found bugs?
1. Document the issue
2. Propose solution
3. Test thoroughly
4. Submit with clear explanation

---

## 📄 License

MIT License - Feel free to use for learning and projects!

---

## 🙏 Acknowledgments

Built with:
- ⚡ Apache Spark & PySpark
- 🔬 MLflow for experiment tracking
- 🤖 scikit-learn, XGBoost, LightGBM, CatBoost
- 📊 Matplotlib, Seaborn for visualization
- 🎨 Faker for synthetic data

---

## 💡 Final Thoughts

**You've just built a production-ready ML system!** 🎉

This project demonstrates:
- ✅ End-to-end ML pipeline
- ✅ MLOps best practices
- ✅ Real-world problem solving
- ✅ Scalable architecture
- ✅ Comprehensive documentation

**What you learned:**
- Data engineering at scale
- Multiple ML algorithms
- Feature engineering
- Model evaluation
- MLOps practices
- Production deployment

**Next challenge**: Apply this to YOUR domain!
- E-commerce recommendation
- Healthcare diagnostics
- IoT anomaly detection
- Financial forecasting
- Your idea here!

---

**Happy Learning! 🚀**

*Remember: The best way to learn is by doing. Run the code, break it, fix it, improve it!*
