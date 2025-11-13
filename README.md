# 🏦 Banking AML Fraud Detection - Complete MLOps Pipeline

**Hey there, future ML engineer!** 👋

Welcome to your comprehensive guide for building a production-ready Anti-Money Laundering (AML) fraud detection system. I'm going to walk you through this like a friendly teacher, explaining every concept along the way.

## 📚 What You'll Learn

This project isn't just about running some algorithms - it's about understanding the **entire journey** from raw data to a deployed model in production. Think of it as your personal ML bootcamp!

### The Big Picture 🎯

Imagine you work at a bank. Every day, millions of transactions happen:
- Sarah buys groceries ($50)
- John pays his rent ($1,500)
- A company deposits a large payment ($100,000)

But hidden among these normal transactions might be **money laundering** activities:
- 💰 Structuring: Breaking large amounts into smaller transactions to avoid detection
- 🌍 Layering: Moving money through complex chains to hide its origin
- 🎭 Shell Companies: Using fake businesses for illegal transfers
- 🔄 Round-tripping: Moving money in circles to make it look legitimate

**Our Mission:** Build an AI system that automatically flags suspicious transactions!

## 🎓 What is AML (Anti-Money Laundering)?

Money laundering is when criminals try to make illegally-obtained money appear legitimate. Banks are legally required to detect and report suspicious activities. That's where our ML system comes in!

### Transaction Categories We'll Detect:
1. **LEGITIMATE** - Normal, everyday transactions
2. **STRUCTURING** - Breaking up large amounts to avoid reporting thresholds
3. **LAYERING** - Complex chains of transactions to obscure money trail
4. **SHELL_COMPANY** - Suspicious business transactions
5. **ROUND_TRIPPING** - Circular money movements

## 🏗️ Project Architecture

```
📦 Banking AML Detection System
│
├── 📊 DATA LAYER
│   ├── Synthetic transaction generation (realistic patterns)
│   ├── Mixed data: Numbers + Text descriptions
│   └── Balanced classes for fair learning
│
├── 🔧 FEATURE ENGINEERING
│   ├── Tabular features: Amount, time patterns, frequency
│   ├── Text features: NLP on transaction descriptions
│   └── Feature scaling and encoding
│
├── 🤖 MODEL LAYER (10+ Algorithms!)
│   ├── Traditional ML: Logistic Regression, Decision Trees
│   ├── Ensemble: Random Forest, Gradient Boosting
│   ├── Deep Learning: Neural Networks
│   └── Advanced: XGBoost, Stacking
│
├── 📈 EVALUATION & MONITORING
│   ├── Multiple metrics (Accuracy, Precision, Recall, F1)
│   ├── Confusion matrices
│   └── ROC curves and AUC
│
└── 🚀 MLOPS PIPELINE
    ├── Experiment tracking
    ├── Model versioning
    ├── Automated training pipelines
    └── Deployment readiness
```

## 🛠️ Technologies Used

- **PySpark**: For big data processing (imagine handling millions of transactions!)
- **PySpark ML**: Scalable machine learning
- **MLflow**: Track experiments like a scientist's lab notebook
- **Python**: The language of modern ML

## 📂 Project Structure Explained

```
pyspark/
│
├── src/                          # All our source code lives here
│   ├── data/                     # Data generation and loading
│   ├── features/                 # Feature engineering magic
│   ├── models/                   # ML algorithms
│   ├── evaluation/               # How good are our models?
│   ├── deployment/               # Taking models to production
│   └── utils/                    # Helper functions
│
├── notebooks/                    # Jupyter notebooks for exploration
├── config/                       # Configuration files
├── data/                         # Where data is stored
│   ├── raw/                      # Original data
│   ├── processed/                # Cleaned and feature-engineered data
│   ├── models/                   # Saved models
│   └── reports/                  # Evaluation reports
│
└── logs/                         # System logs and tracking
```

## 🚀 Getting Started

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Generate Data
```bash
python src/data/generate_data.py
```

### Step 3: Run the Complete Pipeline
```bash
python src/main_pipeline.py
```

### Step 4: Explore Results
```bash
# Check the reports in data/reports/
# View MLflow UI for experiment tracking
mlflow ui
```

## 🎯 The 10+ Algorithms We'll Use

### Why Multiple Algorithms?

Think of it like this: If you're sick, would you trust just one doctor's opinion? Probably not! Same with ML - different algorithms see patterns differently.

1. **Logistic Regression** 📊
   - The simplest starter (like learning to ride a bike with training wheels)
   - Fast and interpretable
   - Great baseline

2. **Decision Tree** 🌲
   - Makes decisions like a flowchart
   - Easy to understand and visualize
   - Can overfit (memorize instead of learn)

3. **Random Forest** 🌳🌳🌳
   - Many decision trees voting together (democracy!)
   - More robust, less overfitting
   - One of the most reliable algorithms

4. **Gradient Boosted Trees (GBT)** 🚀
   - Learns from mistakes iteratively
   - Often wins Kaggle competitions
   - Powerful but needs tuning

5. **Naive Bayes** 🎲
   - Based on probability theory
   - Fast and works well with text
   - "Naive" but surprisingly effective

6. **Linear Support Vector Classifier (SVC)** 📐
   - Finds the best boundary between classes
   - Works well in high dimensions
   - Solid performer

7. **Multilayer Perceptron (MLP)** 🧠
   - A neural network (baby version of deep learning)
   - Can learn complex patterns
   - Needs more data and tuning

8. **One-vs-Rest (OvR)** 🎯
   - Breaks multiclass into multiple binary problems
   - Ensemble approach
   - Great for imbalanced classes

9. **Feature Engineering Variants** ✨
   - Same algorithm, different feature combinations
   - Tests what features matter most
   - Critical for performance

10. **Ensemble Stacking** 🎪
    - Combines predictions from multiple models
    - The "wisdom of crowds" approach
    - Often gives best performance

## 📖 MLOps Best Practices Included

### What is MLOps?

MLOps = Machine Learning + Operations

It's about building ML systems that work reliably in production, not just in notebooks!

**Key Principles We Follow:**

1. **Reproducibility** 🔁
   - Same code + same data = same results
   - Version everything: data, code, models

2. **Experimentation** 🧪
   - Track every experiment
   - Compare models scientifically
   - Never lose your best model

3. **Automation** 🤖
   - Automate training pipelines
   - Scheduled retraining
   - Continuous integration

4. **Monitoring** 📊
   - Track model performance over time
   - Detect model drift
   - Alert when things go wrong

5. **Validation** ✅
   - Multiple validation techniques
   - Cross-validation
   - Test on unseen data

6. **Deployment** 🚀
   - Model serving infrastructure
   - A/B testing capabilities
   - Rollback mechanisms

## 🎓 Learning Path

### If You're Brand New:
1. Start with `notebooks/01_understanding_data.ipynb`
2. Read through `src/data/generate_data.py` (heavily commented!)
3. Run the simple pipeline first
4. Gradually explore more complex components

### If You Have Some Experience:
1. Jump into `src/main_pipeline.py`
2. Experiment with different algorithms
3. Try feature engineering
4. Explore MLflow tracking

### If You're Advanced:
1. Optimize hyperparameters
2. Build custom transformers
3. Implement model ensembles
4. Deploy to production

## 💡 Key Concepts Explained Simply

### What is a Feature?
A feature is just a piece of information the model uses to make decisions. Like how you might judge if someone is trustworthy by their handshake, eye contact, and words - each of these is a "feature."

### What is Training?
Training is showing the model thousands of examples until it learns patterns. Like teaching a child to identify animals by showing them many pictures.

### What is Validation?
Testing the model on data it hasn't seen before. Like a practice exam before the real test.

### What is Overfitting?
When a model memorizes training data instead of learning patterns. Like a student who memorizes answers instead of understanding concepts - they fail on new questions!

### What is Feature Engineering?
Creating new useful features from raw data. Like combining "transaction amount" and "account balance" to create "percentage of balance spent" - often more informative!

## 📊 Understanding the Results

After running the pipeline, you'll see:

1. **Accuracy**: How often is the model right? (But can be misleading!)
2. **Precision**: When it says "fraud," how often is it actually fraud?
3. **Recall**: Of all actual frauds, how many did we catch?
4. **F1-Score**: Harmonic mean of precision and recall (the balanced view)
5. **AUC-ROC**: How well can the model distinguish between classes?

**Important**: In fraud detection, **recall is often most important** - we don't want to miss real fraud!

## 🔍 What Makes This Project Special?

1. **Production-Ready**: Not just a toy example - follows industry practices
2. **Comprehensive**: Covers the entire ML lifecycle
3. **Educational**: Every line explained for learners
4. **Scalable**: Built with PySpark for big data
5. **Modern**: Uses current best practices and tools

## 🤝 Contributing

This is a learning project! Feel free to:
- Add more algorithms
- Improve feature engineering
- Add more evaluation metrics
- Enhance documentation
- Fix bugs

## 📚 Further Reading

- [PySpark ML Guide](https://spark.apache.org/docs/latest/ml-guide.html)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [AML Compliance Basics](https://www.fincen.gov/)
- [Imbalanced Learning](https://imbalanced-learn.org/)

## ⚖️ Legal & Ethical Considerations

**Important**: This is an educational project with synthetic data. Real AML systems must:
- Comply with regulations (Bank Secrecy Act, etc.)
- Protect customer privacy (GDPR, CCPA)
- Avoid bias and discrimination
- Have human oversight for final decisions
- Be explainable to regulators

**Never** use this on real banking data without proper authorization, security, and compliance reviews!

## 🎉 Let's Get Started!

Remember: **Everyone starts as a beginner.** Take your time, experiment, break things, and learn! The best way to learn ML is by doing.

Got questions? Read the code comments - they're written like a friendly conversation!

**Happy Learning! 🚀**

---

*"The journey of a thousand models begins with a single line of code."* 😊
