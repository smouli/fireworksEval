#!/bin/bash
# Launcher script for SFT workflow - ensures venv is activated

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment
if [ -d "venv_querygpt" ]; then
    source venv_querygpt/bin/activate
else
    echo "Error: venv_querygpt directory not found"
    echo "Please create it with: python3 -m venv venv_querygpt"
    exit 1
fi

# Run the SFT workflow script
python3 run_sft_workflow.py "$@"

