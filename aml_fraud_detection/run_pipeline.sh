#!/bin/bash

################################################################################
# AML FRAUD DETECTION PIPELINE - RUN SCRIPT
################################################################################
# This script makes it easy to run the complete pipeline!
# Just execute: ./run_pipeline.sh
################################################################################

echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                          ║"
echo "║            🏦 AML FRAUD DETECTION PIPELINE 🏦                           ║"
echo "║                                                                          ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null
then
    echo "❌ Error: Python 3 is not installed!"
    echo "   Please install Python 3.8 or higher"
    exit 1
fi

echo "✓ Python found: $(python3 --version)"

# Check if Java is installed (required for PySpark)
if ! command -v java &> /dev/null
then
    echo "⚠️  Warning: Java is not installed!"
    echo "   PySpark requires Java 8 or 11"
    echo "   Install from: https://www.oracle.com/java/technologies/downloads/"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]
    then
        exit 1
    fi
else
    echo "✓ Java found: $(java -version 2>&1 | head -n 1)"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 STEP 1: Checking Dependencies"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if requirements are installed
if python3 -c "import pyspark" 2>/dev/null; then
    echo "✓ PySpark is installed"
else
    echo "⚠️  PySpark not found. Installing dependencies..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Error installing dependencies"
        exit 1
    fi
fi

if python3 -c "import yaml" 2>/dev/null; then
    echo "✓ PyYAML is installed"
fi

if python3 -c "import numpy" 2>/dev/null; then
    echo "✓ NumPy is installed"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 STEP 2: Running the Pipeline"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Starting AML Fraud Detection Pipeline..."
echo "This will take approximately 5-10 minutes..."
echo ""

# Navigate to src directory
cd src

# Run the main pipeline
python3 main_pipeline.py

# Check if pipeline succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✅ PIPELINE COMPLETED SUCCESSFULLY!"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "📂 Results are available in:"
    echo "   • ../data/           - Generated data"
    echo "   • ../models/         - Trained models"
    echo "   • ../results/        - Evaluation reports"
    echo "   • ../logs/           - System logs"
    echo ""
    echo "📄 Key files to check:"
    echo "   • ../results/executive_summary.txt"
    echo "   • ../results/evaluation_report.txt"
    echo "   • ../logs/aml_fraud_detection.log"
    echo ""
    echo "🎉 Thank you for using the AML Fraud Detection Pipeline!"
else
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "❌ PIPELINE FAILED"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Please check the logs for errors:"
    echo "   cat ../logs/aml_fraud_detection.log"
    exit 1
fi
