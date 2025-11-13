#!/bin/bash

# ============================================================================
# QUICK START SCRIPT - Run the Complete ML Pipeline
# ============================================================================
# This script automates the entire workflow!
#
# What it does:
# 1. Checks Python environment
# 2. Generates synthetic data
# 3. Runs the ML pipeline
# 4. Opens results
#
# Usage:
#   chmod +x run_pipeline.sh
#   ./run_pipeline.sh
# ============================================================================

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                                                              ║"
echo "║     🏦  BANKING AML FRAUD DETECTION PIPELINE  🏦            ║"
echo "║                                                              ║"
echo "║              Quick Start Script                             ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if Python is installed
print_info "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed!"
    echo "Please install Python 3.8+ and try again."
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
print_success "Python $PYTHON_VERSION found"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    print_error "pip3 is not installed!"
    echo "Please install pip and try again."
    exit 1
fi

print_success "pip3 found"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    print_warning "Virtual environment not found. Creating one..."
    python3 -m venv venv
    print_success "Virtual environment created"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies
print_info "Checking dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

print_success "Dependencies ready"

# Create necessary directories
print_info "Creating directories..."
mkdir -p data/{raw,processed,models,reports}
mkdir -p logs
mkdir -p notebooks

print_success "Directory structure ready"

# Step 1: Generate Data
print_info "Step 1/3: Generating synthetic banking data..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -f "data/raw/transactions.parquet" ]; then
    print_warning "Data already exists. Skipping generation."
    read -p "Do you want to regenerate data? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python3 src/data/generate_data.py
    fi
else
    python3 src/data/generate_data.py
fi

if [ $? -ne 0 ]; then
    print_error "Data generation failed!"
    exit 1
fi

print_success "Data generation complete"
echo ""

# Step 2: Run ML Pipeline
print_info "Step 2/3: Running ML Pipeline..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
print_warning "This will train 10+ models and may take several minutes..."
echo ""

python3 src/main_pipeline.py

if [ $? -ne 0 ]; then
    print_error "Pipeline execution failed!"
    exit 1
fi

print_success "Pipeline execution complete"
echo ""

# Step 3: Show Results
print_info "Step 3/3: Results Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -f "data/reports/model_comparison.csv" ]; then
    print_success "Model comparison results:"
    echo ""
    head -5 data/reports/model_comparison.csv | column -t -s','
    echo ""
fi

# Summary
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                     🎉 SUCCESS! 🎉                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
print_success "Pipeline completed successfully!"
echo ""
echo "📂 Outputs:"
echo "  ├─ Models:       data/models/"
echo "  ├─ Reports:      data/reports/"
echo "  └─ Logs:         logs/"
echo ""
echo "📊 Next Steps:"
echo "  1. View detailed report: cat data/reports/evaluation_report.md"
echo "  2. Compare models:       cat data/reports/model_comparison.csv"
echo "  3. Check logs:           cat logs/pipeline.log"
echo ""
echo "💡 Pro Tips:"
echo "  - Adjust settings in config/config.yaml"
echo "  - Try different hyperparameters"
echo "  - Add more training data for better results"
echo ""

# Deactivate virtual environment
deactivate

print_success "All done! Happy ML! 🚀"
