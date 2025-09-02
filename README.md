# MSAT Quality Dashboard

This project generates synthetic data for MSAT (Manufacturing Science and Technology) quality dashboards.

## Setup

1. **Clone the repository** (if not already done)
   ```bash
   git clone <repository-url>
   cd MSAT_quality_dashboard
   ```

2. **Set up the Python environment**
   ```bash
   # The virtual environment is already created
   source venv/bin/activate
   ```

3. **Install dependencies** (already done, but if needed)
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Option 1: Using the convenience script
```bash
./run_script.sh
```

### Option 2: Manual activation
```bash
source venv/bin/activate
python data/scripts/generate_synthetic_data.py
```

## Output

The script generates the following CSV files:
- `lots_large.csv` - Production lot data
- `qc_results_large.csv` - Quality control test results
- `deviations_large.csv` - Deviation and CAPA data
- `data_dictionary_large.csv` - Data dictionary

## Requirements

- Python 3.8+
- pandas >= 1.5.0
- numpy >= 1.21.0
