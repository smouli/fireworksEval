#!/bin/bash

set -e

echo "Setting up Fireworks QueryGPT..."
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv_querygpt" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv_querygpt
fi

# Activate virtual environment
source venv_querygpt/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q openai

echo ""
echo "✓ Setup complete!"
echo ""
echo "To use QueryGPT:"
echo "  1. Activate virtual environment: source venv_querygpt/bin/activate"
echo "  2. Set API key: export FIREWORKS_API_KEY='your-key-here'"
echo "  3. Run: python3 example_querygpt_usage.py"
echo ""

