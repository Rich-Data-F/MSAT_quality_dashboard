import pandas as pd
import numpy as np
import datetime as dt
from datetime import timedelta
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

print("🔄 Generating LARGE synthetic MSAT quality dataset...")
print("📅 Time period: 32 months (Jan 2023 - Aug 2025)")

# Define products and parameters (expanded)
products = ['octanate', 'octanate_LV', 'fibryga']
lines = ['Line_A', 'Line_B', 'Line_C']  # Added third line
parameters = ['assay_percent', 'pH', 'osmolality_mOsm_kg', 'fill_volume_ml', 
             'particulates_per_ml', 'endotoxin_eu_ml', 'protein_mg_ml']  # Added parameters

# Define realistic specs and means for each parameter
param_specs = {
    'assay_percent': {'mean': 102.5, 'std': 2.8, 'lsl': 95.0, 'usl': 110.0, 'unit': '%'},
    'pH': {'mean': 7.15, 'std': 0.08, 'lsl': 6.9, 'usl': 7.4, 'unit': 'pH'},
    'osmolality_mOsm_kg': {'mean': 295, 'std': 12, 'lsl': 270, 'usl': 320, 'unit': 'mOsm/kg'},
    'fill_volume_ml': {'mean': 10.05, 'std': 0.15, 'lsl': 9.5, 'usl': 10.5, 'unit': 'ml'},
    'particulates_per_ml': {'mean': 25, 'std': 8, 'lsl': 0, 'usl': 50, 'unit': 'count/ml'},
    'endotoxin_eu_ml': {'mean': 0.15, 'std': 0.05, 'lsl': 0.0, 'usl': 0.5, 'unit': 'EU/ml'},
    'protein_mg_ml': {'mean': 8.5, 'std': 0.8, 'lsl': 7.0, 'usl': 10.0, 'unit': 'mg/ml'}
}

# =============================================================================
# GENERATE LOTS DATA - 32 months, more frequent production
# =============================================================================
print("📦 Creating lots data (32 months)...")

start_date = dt.datetime(2023, 1, 1)
end_date = dt.datetime(2025, 8, 31)
lots_data = []
lot_id = 1000

# Generate lots every 1-3 days (more realistic production schedule)
current_date = start_date
lot_counter = 0

while current_date <= end_date:
    for product in products:
        for line in lines:
            # Each product-line combination produces ~2-3 lots per week
            if random.random() < 0.4:  # 40% chance of production on any given day
                lot_start = current_date
                planned_release = lot_start + timedelta(days=random.randint(7, 16))

                # Add realistic delays and early releases
                delay_prob = random.random()
                if delay_prob < 0.15:  # 15% major delays
                    actual_delay = random.randint(3, 10)
                elif delay_prob < 0.35:  # 20% minor delays  
                    actual_delay = random.randint(1, 2)
                else:  # 65% on-time or early
                    actual_delay = random.randint(-2, 1)

                disposition_date = planned_release + timedelta(days=actual_delay)

                lots_data.append({
                    'lot_id': f"{product[:3].upper()}{lot_id:04d}",
                    'product': product,
                    'line': line,
                    'lot_start_date': lot_start,
                    'planned_release_date': planned_release,
                    'disposition_date': disposition_date,
                    'status': 'Released' if disposition_date <= dt.datetime.now() else 'In_Process',
                    'batch_size_L': random.randint(500, 2000),  # Batch size in liters
                    'campaign_id': f"CAMP_{lot_start.year}_{lot_start.month:02d}"
                })
                lot_id += 1
                lot_counter += 1

    # Move to next day
    current_date += timedelta(days=1)

lots_df = pd.DataFrame(lots_data)
print(f"   ✅ Generated {len(lots_df)} lots across 32 months")

# =============================================================================
# GENERATE QC TEST RESULTS - 7 parameters per lot
# =============================================================================
print("🧪 Creating QC test results (7 parameters per lot)...")

def process_lot(lot_row):
    lot_id = lot_row['lot_id']
    product = lot_row['product']
    line = lot_row['line']
    lot_start = lot_row['lot_start_date']
    results = []
    for param in parameters:
        # Different parameters have different testing schedules
        if param in ['assay_percent', 'pH', 'fill_volume_ml']:
            # Quick tests - done early
            sample_delay = random.randint(6, 18)  # 6-18 hours
            test_delay = random.randint(2, 8)     # 2-8 hours
        else:
            # Complex tests - take longer
            sample_delay = random.randint(12, 36)  # 12-36 hours
            test_delay = random.randint(8, 48)     # 8-48 hours

        sample_date = lot_start + timedelta(hours=sample_delay)
        result_date = sample_date + timedelta(hours=test_delay)
        review_date = result_date + timedelta(hours=random.randint(1, 24))

        specs = param_specs[param]
        base_mean = specs['mean']
        base_std = specs['std']

        # Add realistic process variations and trends over time
        lot_num = int(lot_id[-4:])
        month_num = lot_start.month

        # Seasonal effects for some parameters
        if param == 'osmolality_mOsm_kg':
            # Slight seasonal variation (summer vs winter)
            seasonal_adj = 5 * np.sin((month_num - 3) * np.pi / 6)
            base_mean += seasonal_adj

        # Process improvements over time
        if param == 'particulates_per_ml' and lot_start >= dt.datetime(2023, 6, 1):
            # Improvement implemented mid-2023
            base_mean *= 0.7
            base_std *= 0.8

        # Equipment changes causing shifts
        if param == 'assay_percent' and dt.datetime(2024, 1, 1) <= lot_start <= dt.datetime(2024, 2, 28):
            # Equipment calibration issue in Jan-Feb 2024
            base_mean += 2.0
            base_std *= 1.3

        # Generate correlated results (process memory)
        if results and random.random() < 0.4:  # 40% correlation
            similar_recent = [r for r in results[-20:] 
                            if r['test_name'] == param and r['product'] == product and r['line'] == line]
            if similar_recent:
                last_values = [r['result_value'] for r in similar_recent[-3:]]
                trend_mean = np.mean(last_values)
                # Weighted towards recent trend
                value = 0.6 * np.random.normal(base_mean, base_std) + 0.4 * np.random.normal(trend_mean, base_std)
            else:
                value = np.random.normal(base_mean, base_std)
        else:
            value = np.random.normal(base_mean, base_std)

        # Occasional special causes (outliers) - increased for more extreme values
        if random.random() < 0.025:  # Increased from 1.5% to 2.5% special cause rate
            extreme_factor = random.uniform(4, 8)  # More extreme outliers
            value += random.choice([-1, 1]) * base_std * extreme_factor

        # Ensure positive values for certain parameters
        if param in ['particulates_per_ml', 'endotoxin_eu_ml']:
            value = max(0, value)

        results.append({
            'test_id': f"T{len(results)+1:07d}",
            'lot_id': lot_id,
            'product': product,
            'line': line,
            'test_name': param,
            'sample_date': sample_date,
            'result_date': result_date,
            'review_date': review_date,
            'result_value': round(value, 4),
            'unit': specs['unit'],
            'lsl': specs['lsl'],
            'usl': specs['usl'],
            'oos_flag': 1 if (value < specs['lsl'] or value > specs['usl']) else 0,
            'reviewer': random.choice(['QC_Analyst_A', 'QC_Analyst_B', 'QC_Analyst_C', 
                                    'QC_Senior_1', 'QC_Senior_2', 'QC_Lead']),
            'status': random.choices(['Reviewed', 'Under_Review'], weights=[0.95, 0.05])[0],
            'method_id': f"M{param[:3].upper()}_001",
            'instrument_id': f"INST_{random.randint(1, 5):02d}"
        })
    return results

qc_results_nested = lots_df.apply(process_lot, axis=1)
qc_results = [item for sublist in qc_results_nested for item in sublist]
qc_df = pd.DataFrame(qc_results)
print(f"   ✅ Generated {len(qc_df)} QC test results")

# =============================================================================
# GENERATE DEVIATIONS AND CAPA DATA - More comprehensive
# =============================================================================
print("⚠️  Creating deviations and CAPA data...")

def process_oos_row(qc_row):
    if qc_row['oos_flag'] == 1:
        opened_date = qc_row['result_date'] + timedelta(hours=random.randint(1, 48))
        severity = random.choices(['Minor', 'Major', 'Critical'], weights=[0.5, 0.4, 0.1])[0]

        # Realistic closure logic with aging consideration
        current_date = dt.datetime.now()
        days_since_opened = (current_date - opened_date).days

        # Adjust closure probability based on age
        base_closure_prob = 0.85 if severity == 'Minor' else 0.75 if severity == 'Major' else 0.65
        age_penalty = min(days_since_opened / 365, 0.2)  # Max 20% penalty for old deviations
        adjusted_closure_prob = base_closure_prob + age_penalty

        if random.random() < adjusted_closure_prob:
            # Closed deviation
            if days_since_opened < 30:
                # Recent - shorter closure
                closure_days = random.randint(1, 14)
            elif days_since_opened < 90:
                # Medium age - normal closure
                closure_days = random.randint(3, 21) if severity == 'Minor' else random.randint(5, 30) if severity == 'Major' else random.randint(7, 45)
            else:
                # Older - longer closure time
                closure_days = random.randint(14, min(days_since_opened - 1, 90))
            closed_date = opened_date + timedelta(days=closure_days)
        else:
            # Open deviation - make it recent to avoid unrealistic aging
            closed_date = None
            if days_since_opened > 120:  # If opened more than 4 months ago
                # Move opened_date to be within last 60 days for open OOS deviations
                max_age_days = 60
                opened_date = current_date - timedelta(days=random.randint(7, max_age_days))

        return {
            'deviation_id': f"DEV{len(deviations)+1:05d}",
            'lot_id': qc_row['lot_id'],
            'category': 'OOS_Result',
            'severity': severity,
            'description': f"OOS result: {qc_row['test_name']} = {qc_row['result_value']} {qc_row['unit']} (LSL:{qc_row['lsl']}, USL:{qc_row['usl']})",
            'opened_at': opened_date,
            'closed_at': closed_date,
            'capa_flag': 1 if severity in ['Major', 'Critical'] else random.randint(0, 1),
            'assignee': random.choice(['QA_Invest_1', 'QA_Invest_2', 'QA_Invest_3', 'MSAT_Lead', 'QA_Manager']),
            'root_cause': random.choice(['Equipment', 'Material', 'Method', 'Environment', 'Personnel', 'Under_Investigation']) if closed_date else 'Under_Investigation'
        }
    return None

deviations = []
oos_deviations = qc_df.apply(process_oos_row, axis=1).dropna()
deviations.extend(oos_deviations)

# Additional process deviations (not OOS-related)
deviation_categories = {
    'Equipment': ['Pump malfunction', 'Filter integrity', 'Temperature excursion', 'Pressure deviation'],
    'Environment': ['Clean room breach', 'HVAC issue', 'Contamination risk'],
    'Material': ['Raw material OOS', 'Component shortage', 'Vendor quality issue'],
    'Personnel': ['Training deviation', 'Procedure not followed', 'Documentation error'],
    'System': ['Database error', 'Label printing issue', 'Software malfunction']
}

# Generate 150 additional process deviations
for _ in range(150):
    random_lot = lots_df.sample(1).iloc[0]
    category = random.choice(list(deviation_categories.keys()))
    issue_type = random.choice(deviation_categories[category])

    # Deviations can occur throughout the process
    opened_date = (random_lot['lot_start_date'] + 
                  timedelta(days=random.randint(0, 
                    (random_lot['disposition_date'] - random_lot['lot_start_date']).days)))

    severity = random.choices(['Minor', 'Major', 'Critical'], weights=[0.65, 0.3, 0.05])[0]

    # Closure patterns with realistic aging
    current_date = dt.datetime.now()
    days_since_opened = (current_date - opened_date).days

    # Adjust closure probability based on age - older deviations more likely to be closed
    base_closure_prob = 0.82
    age_penalty = min(days_since_opened / 365, 0.3)  # Max 30% penalty for very old deviations
    adjusted_closure_prob = base_closure_prob + age_penalty

    if random.random() < adjusted_closure_prob:
        # Closed deviation
        if days_since_opened < 30:
            # Recent deviation - shorter closure time
            closure_days = random.randint(1, 14)
        elif days_since_opened < 90:
            # Medium age - normal closure time
            closure_days = random.randint(7, 30)
        else:
            # Older deviation - longer closure time (already closed)
            closure_days = random.randint(14, min(days_since_opened - 1, 120))
        closed_date = opened_date + timedelta(days=closure_days)
    else:
        # Open deviation - make it more recent to avoid unrealistic aging
        closed_date = None
        if days_since_opened > 180:  # If opened more than 6 months ago
            # Move opened_date to be within last 90 days for open deviations
            max_age_days = 90
            opened_date = current_date - timedelta(days=random.randint(7, max_age_days))

    deviations.append({
        'deviation_id': f"DEV{len(deviations)+1:05d}",
        'lot_id': random_lot['lot_id'],
        'category': category,
        'severity': severity,
        'description': f"{issue_type} during lot {random_lot['lot_id']} processing",
        'opened_at': opened_date,
        'closed_at': closed_date,
        'capa_flag': 1 if (severity == 'Critical' or 
                          (severity == 'Major' and random.random() < 0.7) or
                          (severity == 'Minor' and random.random() < 0.2)) else 0,
        'assignee': random.choice(['Prod_Super_1', 'Prod_Super_2', 'QA_Invest_1', 'QA_Invest_2', 
                                 'Maint_Lead', 'MSAT_Engineer', 'QA_Manager']),
        'root_cause': random.choice(['Equipment', 'Material', 'Method', 'Environment', 'Personnel', 'System', 'Under_Investigation']) if closed_date else 'Under_Investigation'
    })

deviations_df = pd.DataFrame(deviations)
print(f"   ✅ Generated {len(deviations_df)} deviations")

# =============================================================================
# UPDATE LOTS STATUS FOR REJECTED LOTS
# =============================================================================
print("🔄 Updating lot statuses for rejected lots...")

# Identify lots with critical deviations
critical_deviations = deviations_df[deviations_df['severity'] == 'Critical']
lots_with_critical = critical_deviations['lot_id'].unique()

# Identify lots with multiple OOS results (more than 2 OOS tests)
oos_counts = qc_df[qc_df['oos_flag'] == 1].groupby('lot_id').size()
lots_with_multiple_oos = oos_counts[oos_counts > 2].index.tolist()

# Combine lots to reject
lots_to_reject = set(lots_with_critical).union(set(lots_with_multiple_oos))

# Select ~12 lots to reject (or all if fewer)
if len(lots_to_reject) > 12:
    lots_to_reject = random.sample(list(lots_to_reject), 12)
elif len(lots_to_reject) < 12:
    # Add some additional lots with major deviations or high OOS
    major_deviations = deviations_df[deviations_df['severity'] == 'Major']
    lots_with_major = major_deviations['lot_id'].unique()
    additional_lots = [lot for lot in lots_with_major if lot not in lots_to_reject]
    if len(additional_lots) > 0:
        needed = 12 - len(lots_to_reject)
        lots_to_reject.update(random.sample(list(additional_lots), min(needed, len(additional_lots))))

# Update status to 'Rejected' for selected lots
lots_df.loc[lots_df['lot_id'].isin(lots_to_reject), 'status'] = 'Rejected'

# For rejected lots, update disposition_date to reflect rejection timing
for lot_id in lots_to_reject:
    lot_row = lots_df[lots_df['lot_id'] == lot_id]
    if not lot_row.empty:
        lot_start = pd.to_datetime(lot_row['lot_start_date'].iloc[0])
        # Set rejection date to 1-7 days after critical issue detection
        rejection_delay = random.randint(1, 7)
        rejection_date = lot_start + timedelta(days=rejection_delay)
        lots_df.loc[lots_df['lot_id'] == lot_id, 'disposition_date'] = rejection_date.strftime('%Y-%m-%d')

print(f"   ✅ Marked {len(lots_to_reject)} lots as 'Rejected' due to critical issues and CAPA requirements")

# =============================================================================
# SAVE LARGE DATASET CSV FILES
# =============================================================================
print("💾 Saving large dataset CSV files...")

# Format dates for Qlik compatibility
lots_df['lot_start_date'] = lots_df['lot_start_date'].dt.strftime('%Y-%m-%d')
lots_df['planned_release_date'] = lots_df['planned_release_date'].dt.strftime('%Y-%m-%d')
lots_df['disposition_date'] = lots_df['disposition_date'].dt.strftime('%Y-%m-%d')

qc_df['sample_date'] = qc_df['sample_date'].dt.strftime('%Y-%m-%d %H:%M:%S')
qc_df['result_date'] = qc_df['result_date'].dt.strftime('%Y-%m-%d %H:%M:%S')
qc_df['review_date'] = qc_df['review_date'].dt.strftime('%Y-%m-%d %H:%M:%S')

deviations_df['opened_at'] = deviations_df['opened_at'].dt.strftime('%Y-%m-%d %H:%M:%S')
deviations_df['closed_at'] = deviations_df['closed_at'].dt.strftime('%Y-%m-%d %H:%M:%S')

# Save with large dataset suffix
lots_df.to_csv('data/lots_large.csv', index=False)
qc_df.to_csv('data/qc_results_large.csv', index=False)
deviations_df.to_csv('data/deviations_large.csv', index=False)

# Extended data dictionary
data_dict_large = {
    'Table': (['lots'] * 9 + ['qc_results'] * 17 + ['deviations'] * 10),
    'Field': ['lot_id', 'product', 'line', 'lot_start_date', 'planned_release_date',
             'disposition_date', 'status', 'batch_size_L', 'campaign_id',
             'test_id', 'lot_id', 'product', 'line', 'test_name', 'sample_date',
             'result_date', 'review_date', 'result_value', 'unit', 'lsl', 'usl',
             'oos_flag', 'reviewer', 'status', 'method_id', 'instrument_id',
             'deviation_id', 'lot_id', 'category', 'severity', 'description',
             'opened_at', 'closed_at', 'capa_flag', 'assignee', 'root_cause'],
    'Description': ['Unique lot identifier', 'Product (octanate/octanate_LV/fibryga)',
                   'Production line (Line_A/B/C)', 'Lot start date', 'Planned release date',
                   'Actual disposition date',                    'Status (Released/In_Process/Rejected)',
                   'Batch size in liters', 'Monthly campaign identifier',
                   'Unique test identifier', 'Associated lot ID', 'Product name',
                   'Production line', 'QC parameter name', 'Sample collection timestamp',
                   'Test result timestamp', 'QC review timestamp', 'Numerical test result',
                   'Unit of measurement', 'Lower specification limit', 'Upper specification limit',
                   'Out of spec flag (1=OOS)', 'QC reviewer name', 'Test status',
                   'Test method identifier', 'Instrument used',
                   'Unique deviation identifier', 'Associated lot ID', 'Deviation category',
                   'Severity (Minor/Major/Critical)', 'Deviation description',
                   'Timestamp opened', 'Timestamp closed (null if open)',
                   'CAPA required flag', 'Assigned investigator', 'Root cause (if determined)']
}

data_dict_large_df = pd.DataFrame(data_dict_large)
data_dict_large_df.to_csv('data/data_dictionary_large.csv', index=False)

# =============================================================================
# SUMMARY STATISTICS FOR LARGE DATASET
# =============================================================================
print("\n" + "="*70)
print("📊 LARGE SYNTHETIC DATASET GENERATION COMPLETE!")
print("="*70)
print(f"✅ data/lots_large.csv: {len(lots_df):,} records")
print(f"✅ data/qc_results_large.csv: {len(qc_df):,} records") 
print(f"✅ data/deviations_large.csv: {len(deviations_df):,} records")
print(f"✅ data/data_dictionary_large.csv: {len(data_dict_large_df)} records")

print("\n📈 Large Dataset Summary:")
print(f"   • Time Period: 32 months (Jan 2023 - Aug 2025)")
print(f"   • Products: {lots_df['product'].nunique()} ({', '.join(lots_df['product'].unique())})")
print(f"   • Production Lines: {lots_df['line'].nunique()} ({', '.join(lots_df['line'].unique())})")
print(f"   • QC Parameters: {qc_df['test_name'].nunique()} parameters")
print(f"   • OOS Results: {qc_df['oos_flag'].sum():,} ({qc_df['oos_flag'].mean()*100:.1f}%)")
print(f"   • Open Deviations: {deviations_df['closed_at'].isna().sum():,}")
print(f"   • CAPAs Required: {deviations_df['capa_flag'].sum():,}")
print(f"   • Avg Lots/Month: {len(lots_df)/18:.0f}")

print("\n🎯 Large dataset ready for enterprise-scale Qlik demo!")
print("📁 Files: data/lots_large.csv, data/qc_results_large.csv, data/deviations_large.csv, data/data_dictionary_large.csv")
