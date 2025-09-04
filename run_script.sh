#!/bin/bash
# Activate virtual environment and run the synthetic data generator

cd "$(dirname "$0")"
source venv/bin/activate
python data/scripts/generate_synthetic_data.py
