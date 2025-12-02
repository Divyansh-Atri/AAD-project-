#!/bin/bash

# Deutsch-Jozsa Algorithm Project Setup Script
# Team: Dhinchak Dikstra

echo "=========================================="
echo "Deutsch-Jozsa Algorithm Project Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the demonstration, use:"
echo "  python src/deutsch_jozsa.py"
echo ""
echo "To run tests, use:"
echo "  pytest tests/"
echo ""
echo "To start Jupyter notebook:"
echo "  jupyter notebook notebooks/deutsch_jozsa_tutorial.ipynb"
echo ""
echo "Happy Quantum Computing! 🚀"
