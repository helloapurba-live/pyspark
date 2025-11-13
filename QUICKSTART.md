# ⚡ Quick Start Guide - Banking AML Fraud Detection

**Get started in 5 minutes!** ⏱️

---

## 🚀 Installation (3 Steps)

### Step 1: Check Prerequisites

```bash
# Check Python version (need 3.8+)
python3 --version

# If not installed, download from: https://www.python.org/downloads/
```

### Step 2: Create Virtual Environment

```bash
# Create isolated Python environment
python3 -m venv venv

# Activate it
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt

# This will install:
# - PySpark (big data processing)
# - Pandas, NumPy (data manipulation)
# - Scikit-learn (ML algorithms)
# - MLflow (experiment tracking)
# - And more...
```

**⚠️ Note**: Installation may take 5-10 minutes depending on your internet speed.

---

## ⚡ Quick Run (2 Options)

### Option 1: Automated Script (Easiest!)

```bash
# Make executable
chmod +x run_pipeline.sh

# Run everything!
./run_pipeline.sh
```

**This will:**
1. ✅ Generate 50,000 synthetic transactions
2. ✅ Train 10+ ML models
3. ✅ Evaluate and compare models
4. ✅ Generate detailed reports

**Time**: 5-15 minutes

---

### Option 2: Manual Steps (More Control)

```bash
# Step 1: Generate data
python src/data/generate_data.py

# Step 2: Run ML pipeline
python src/main_pipeline.py

# Step 3: View results
cat data/reports/model_comparison.csv
```

---

## 📊 View Results

After running, check these files:

```bash
# Quick comparison of all models
data/reports/model_comparison.csv

# Detailed evaluation report
data/reports/evaluation_report.md

# Raw metrics (JSON)
data/reports/evaluation_results.json
```

**Example output:**
```
Rank | Model                    | Accuracy | F1-Score
-----|--------------------------|----------|----------
1    | Random_Forest_Tuned      | 0.9234   | 0.9194
2    | Gradient_Boosted_Trees   | 0.9156   | 0.9121
3    | Random_Forest_Large      | 0.9089   | 0.9055
```

---

## 🎯 What You Get

After running the pipeline, you'll have:

✅ **50,000 synthetic banking transactions**
   - 5 categories: Legitimate + 4 fraud types
   - Realistic patterns and amounts
   - Mixed data: numbers + text descriptions

✅ **10+ Trained ML Models**
   - Logistic Regression
   - Decision Trees (multiple variants)
   - Random Forests (multiple variants)
   - Gradient Boosted Trees
   - Naive Bayes
   - Linear SVC
   - Neural Networks
   - And more!

✅ **Comprehensive Evaluation**
   - Accuracy, Precision, Recall, F1-Score
   - Confusion matrices
   - Per-class performance
   - Model comparison

✅ **Production-Ready Pipeline**
   - Feature engineering
   - Model training
   - Evaluation
   - Deployment infrastructure

✅ **Complete Documentation**
   - README: Project overview
   - TUTORIAL: Step-by-step guide
   - Code comments: Every line explained

---

## 🎓 Learning Path

### 🟢 Beginner (Start Here!)

1. **Read** `README.md` - Understand the project
2. **Run** the automated script
3. **View** results in `data/reports/`
4. **Explore** `src/data/generate_data.py` - Read the comments

### 🟡 Intermediate

1. **Modify** `config/config.yaml` - Change settings
2. **Run** pipeline again - Compare results
3. **Add** custom features in feature_engineering.py
4. **Experiment** with different algorithms

### 🔴 Advanced

1. **Implement** hyperparameter tuning
2. **Add** new ML algorithms
3. **Create** ensemble methods
4. **Deploy** models to production
5. **Build** REST API for model serving

---

## 🔧 Troubleshooting

### Issue: "Data not found"

```bash
# Solution: Generate data first
python src/data/generate_data.py
```

### Issue: "Module not found"

```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

### Issue: "Out of memory"

Edit `config/config.yaml`:
```yaml
data:
  generation:
    num_transactions: 10000  # Reduce dataset size

spark:
  config:
    "spark.driver.memory": "2g"  # Reduce memory
```

### Issue: "Too slow"

**Quick fixes:**
- Reduce number of transactions (10,000 instead of 50,000)
- Train fewer models (comment out some in main_pipeline.py)
- Use simpler models (fewer trees, less depth)

---

## 📁 Project Structure

```
pyspark/
├── 📄 README.md              ← Project overview
├── 📄 QUICKSTART.md          ← This file!
├── 📄 TUTORIAL.md            ← Detailed guide
├── 📄 requirements.txt       ← Dependencies
├── 📄 run_pipeline.sh        ← Automated runner
│
├── 📁 src/                   ← Source code
│   ├── main_pipeline.py      ← Main orchestrator
│   ├── data/                 ← Data generation
│   ├── features/             ← Feature engineering
│   ├── models/               ← ML algorithms
│   ├── evaluation/           ← Model assessment
│   └── deployment/           ← Model serving
│
├── 📁 config/                ← Settings
│   └── config.yaml           ← Configuration
│
├── 📁 data/                  ← Data storage
│   ├── raw/                  ← Original data
│   ├── models/               ← Trained models
│   └── reports/              ← Results
│
└── 📁 logs/                  ← Execution logs
```

---

## 💡 Pro Tips

### Tip 1: Start Small
```bash
# In config/config.yaml, change:
num_transactions: 10000  # Instead of 50000
```

### Tip 2: Use Notebooks
```bash
# Launch Jupyter
jupyter notebook

# Create new notebook in notebooks/
# Great for exploration and visualization!
```

### Tip 3: Track Experiments
Keep a log of your experiments:
```
Experiment 1: Default settings → F1 = 0.87
Experiment 2: More trees (100 → 200) → F1 = 0.91 (better!)
Experiment 3: Added velocity feature → F1 = 0.93 (even better!)
```

### Tip 4: Read the Code
**Every file has detailed comments** explaining:
- **What** the code does
- **Why** we do it that way
- **How** it works

Don't just run it - understand it!

---

## 🆘 Getting Help

### Option 1: Check Documentation
- `README.md` - Project overview
- `TUTORIAL.md` - Detailed walkthrough
- Code comments - Line-by-line explanations

### Option 2: Check Logs
```bash
# View execution logs
cat logs/pipeline.log

# Follow logs in real-time
tail -f logs/pipeline.log
```

### Option 3: Debug Mode
```python
# In any Python file, add:
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## 🎯 Next Steps

After successfully running the pipeline:

### 📊 Analyze Results
1. Compare model performance
2. Identify best model for your use case
3. Understand which features matter most

### 🎨 Customize
1. Modify configuration in `config/config.yaml`
2. Add custom features
3. Try different algorithms
4. Tune hyperparameters

### 🚀 Deploy
1. Load best model
2. Make predictions on new data
3. Create REST API
4. Monitor performance

### 📚 Learn More
1. Read PySpark documentation
2. Explore ML algorithms in depth
3. Study feature engineering techniques
4. Learn about MLOps best practices

---

## ✅ Success Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] Data generated successfully
- [ ] Pipeline executed without errors
- [ ] Results viewed and understood
- [ ] Ready to customize and experiment!

---

## 🎉 Congratulations!

You've successfully set up a complete, production-ready ML system for fraud detection!

**What you've learned:**
- ✅ End-to-end ML pipeline
- ✅ Feature engineering
- ✅ Multiple ML algorithms
- ✅ Model evaluation
- ✅ MLOps best practices

**You're now ready to:**
- Build your own ML projects
- Contribute to ML systems
- Understand production ML
- Continue learning advanced topics

---

## 📞 Support

**Found a bug?** Check logs in `logs/`

**Have questions?** Read `TUTORIAL.md`

**Want to contribute?** Add features, fix bugs, improve docs!

---

**Happy Learning! 🚀**

*"The journey of a thousand models begins with a single `python src/main_pipeline.py`"* 😊
