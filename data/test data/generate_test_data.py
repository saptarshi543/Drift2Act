"""Generate test CSVs calibrated to actual PhysioNet reference distributions."""
import pandas as pd, numpy as np, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
np.random.seed(42)

# Actual PhysioNet Phase 1 distributions (from patient_level.csv)
REF = {
    'HR':       {'mean': 84.86, 'std': 14.00},
    'O2Sat':    {'mean': 97.10, 'std': 1.85},
    'Temp':     {'mean': 36.88, 'std': 0.53},
    'SBP':      {'mean': 119.83, 'std': 16.15},
    'MAP':      {'mean': 78.31, 'std': 10.84},
    'DBP':      {'mean': 60.03, 'std': 10.43},
    'Resp':     {'mean': 18.63, 'std': 3.62},
    'EtCO2':    {'mean': 33.0,  'std': 5.0},
    'BaseExcess': {'mean': -1.5, 'std': 3.5},
    'HCO3':     {'mean': 23.0, 'std': 3.5},
    'FiO2':     {'mean': 0.5,  'std': 0.2},
    'pH':       {'mean': 7.38, 'std': 0.06},
    'PaCO2':    {'mean': 40.0, 'std': 7.0},
    'SaO2':     {'mean': 96.5, 'std': 2.5},
    'AST':      {'mean': 60.0, 'std': 80.0},
    'BUN':      {'mean': 23.65, 'std': 19.38},
    'Alkalinephos': {'mean': 80.0, 'std': 40.0},
    'Calcium':  {'mean': 8.2,  'std': 0.7},
    'Chloride': {'mean': 103.0, 'std': 5.0},
    'Creatinine': {'mean': 1.35, 'std': 1.43},
    'Bilirubin_direct': {'mean': 0.5, 'std': 0.8},
    'Glucose':  {'mean': 133.44, 'std': 36.54},
    'Lactate':  {'mean': 2.09, 'std': 1.57},
    'Magnesium': {'mean': 2.0, 'std': 0.4},
    'Phosphate': {'mean': 3.5, 'std': 1.2},
    'Potassium': {'mean': 4.1, 'std': 0.6},
    'Bilirubin_total': {'mean': 1.0, 'std': 1.5},
    'TroponinI': {'mean': 0.5, 'std': 2.0},
    'Hct':      {'mean': 32.0, 'std': 5.0},
    'Hgb':      {'mean': 10.72, 'std': 1.64},
    'PTT':      {'mean': 35.0, 'std': 15.0},
    'WBC':      {'mean': 11.71, 'std': 7.21},
    'Fibrinogen': {'mean': 280.0, 'std': 100.0},
    'Platelets': {'mean': 212.46, 'std': 105.28},
}

# Sparsity: fraction of hours with a measurement
SPARSITY = {
    'HR': 0.99, 'O2Sat': 0.99, 'Temp': 0.95, 'SBP': 0.99, 'MAP': 0.99,
    'DBP': 0.99, 'Resp': 0.99, 'EtCO2': 0.01,
    'BaseExcess': 0.15, 'HCO3': 0.15, 'FiO2': 0.20, 'pH': 0.15,
    'PaCO2': 0.15, 'SaO2': 0.15, 'AST': 0.10, 'BUN': 0.30,
    'Alkalinephos': 0.10, 'Calcium': 0.20, 'Chloride': 0.25,
    'Creatinine': 0.30, 'Bilirubin_direct': 0.05, 'Glucose': 0.35,
    'Lactate': 0.15, 'Magnesium': 0.15, 'Phosphate': 0.10,
    'Potassium': 0.25, 'Bilirubin_total': 0.08, 'TroponinI': 0.03,
    'Hct': 0.30, 'Hgb': 0.30, 'PTT': 0.15, 'WBC': 0.30,
    'Fibrinogen': 0.08, 'Platelets': 0.30,
}

COLUMNS = ['Patient_ID', 'Hour'] + list(REF.keys()) + ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS', 'SepsisLabel']

# Correlation map (latent severity theta effect)
CORR = {
    'HR': 0.4, 'Temp': 0.2, 'Resp': 0.4, 'O2Sat': -0.2,
    'MAP': -0.4, 'SBP': -0.4, 'DBP': -0.3,
    'Creatinine': 0.3, 'Lactate': 0.4, 'WBC': 0.3, 'BUN': 0.3,
    'pH': -0.3, 'BaseExcess': -0.3, 'HCO3': -0.3,
    'Glucose': 0.1, 'Platelets': -0.2, 'FiO2': 0.2, 'PaCO2': 0.2,
}

def generate_batch(n_patients, shifts=None, label_flip_rate=0.0, name="batch"):
    """Generate a batch of hourly ICU data with clinical panel draws."""
    if shifts is None:
        shifts = {}
    rows = []
    
    # Feature group categorizations
    vitals = {'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp'}
    daily_labs = {'BUN', 'Calcium', 'Chloride', 'Creatinine', 'Glucose', 'Magnesium', 
                  'Phosphate', 'Potassium', 'Hct', 'Hgb', 'PTT', 'WBC', 'Platelets'}
    lfts = {'AST', 'Alkalinephos', 'Bilirubin_total', 'Bilirubin_direct'}
    abg = {'pH', 'PaCO2', 'BaseExcess', 'HCO3', 'SaO2', 'FiO2'}
    lactate = {'Lactate'}
    troponin = {'TroponinI'}
    fibrinogen = {'Fibrinogen'}
    etco2 = {'EtCO2'}

    for pid in range(n_patients):
        n_hours = np.random.randint(12, 72)
        age = np.clip(np.random.normal(61.88, 16.03), 18, 89)
        gender = np.random.choice([0, 1], p=[0.44, 0.56])
        unit = np.random.choice([0, 1])
        hosp_adm = np.random.uniform(-100, 0)
        is_sepsis = np.random.random() < 0.087
        sepsis_onset = np.random.randint(max(6, n_hours // 2), n_hours) if is_sepsis else n_hours + 100

        # Draw patient-specific mean and volatility
        theta = np.random.normal(0, 0.8)  # Latent patient severity factor
        p_means = {}
        p_vols = {}
        for feat, ref in REF.items():
            c = CORR.get(feat, 0.0)
            p_means[feat] = np.random.normal(ref['mean'] + c * theta * ref['std'], ref['std'] * np.sqrt(1 - c**2) * 0.9)
            p_vols[feat] = np.maximum(ref['std'] * 0.1, np.random.normal(ref['std'] * 0.55, ref['std'] * 0.15))

        # Panel ordering flags for this patient
        has_daily_labs = np.random.random() < 0.97
        has_lft_labs = np.random.random() < 0.28
        has_abg_labs = np.random.random() < 0.62
        has_lactate = np.random.random() < 0.40
        has_troponin = np.random.random() < 0.03
        has_fibrinogen = np.random.random() < 0.13

        # Lab draw hours (draw panel every 24 hours, starting at hour 6)
        draw_hours = set(range(6, n_hours, 24))

        for h in range(n_hours):
            row = {'Patient_ID': pid, 'Hour': h}

            for feat, ref in REF.items():
                # Sparsity / Draw check
                is_observed = False
                if feat in vitals:
                    is_observed = np.random.random() < 0.99
                elif feat in daily_labs:
                    is_observed = (h in draw_hours) and has_daily_labs
                elif feat in lfts:
                    is_observed = (h in draw_hours) and has_lft_labs
                elif feat in abg:
                    is_observed = (h in draw_hours) and has_abg_labs
                elif feat in lactate:
                    is_observed = (h in draw_hours) and has_lactate
                elif feat in troponin:
                    is_observed = (h in draw_hours) and has_troponin
                elif feat in fibrinogen:
                    is_observed = (h in draw_hours) and has_fibrinogen
                elif feat in etco2:
                    is_observed = np.random.random() < 0.01

                if not is_observed:
                    row[feat] = np.nan
                    continue

                base = p_means[feat]
                noise = np.random.normal(0, p_vols[feat])

                # Sepsis effect near onset
                if is_sepsis and h >= sepsis_onset - 4:
                    if feat in ('HR', 'Resp', 'Lactate', 'WBC', 'Creatinine'):
                        base *= 1.15
                    elif feat in ('O2Sat', 'SBP', 'MAP'):
                        base *= 0.92
                    elif feat == 'Temp':
                        base += 1.0

                # Apply drift shifts
                if feat in shifts:
                    add_val, mult_val = shifts[feat]
                    base = base * mult_val + add_val

                val = base + noise

                # Clip to physiological ranges
                if feat == 'O2Sat': val = np.clip(val, 60, 100)
                elif feat == 'HR': val = np.clip(val, 30, 200)
                elif feat == 'Temp': val = np.clip(val, 33, 42)
                elif feat == 'pH': val = np.clip(val, 6.8, 7.8)
                elif feat == 'FiO2': val = np.clip(val, 0.21, 1.0)
                elif feat in ('SBP', 'MAP', 'DBP'): val = max(20, val)
                elif feat == 'Resp': val = max(5, val)

                # General clipping to keep positive features positive
                if feat in ('AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 
                            'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 
                            'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 
                            'Fibrinogen', 'Platelets'):
                    val = max(0.01, val)

                row[feat] = round(val, 2)

            row['Age'] = round(age, 1)
            row['Gender'] = gender
            row['Unit1'] = 1 if unit == 0 else 0
            row['Unit2'] = 1 if unit == 1 else 0
            row['HospAdmTime'] = round(hosp_adm, 1)
            row['ICULOS'] = h + 1

            sep_label = 1 if is_sepsis and h >= sepsis_onset else 0
            if label_flip_rate > 0 and sep_label == 0 and np.random.random() < label_flip_rate:
                sep_label = 1
            row['SepsisLabel'] = sep_label

            rows.append(row)

    df = pd.DataFrame(rows, columns=COLUMNS)
    return df


print("Generating calibrated test data...")

# 1. Normal batch - matches PhysioNet distributions
df_normal = generate_batch(200, shifts={}, name="normal")
path_normal = ROOT / "normal_batch.csv"
df_normal.to_csv(path_normal, index=False)
n_sep = df_normal.groupby('Patient_ID')['SepsisLabel'].max().sum()
print(f"  normal_batch.csv: {len(df_normal)} rows, {df_normal['Patient_ID'].nunique()} patients, {n_sep} sepsis")
for f in ['HR', 'O2Sat', 'Temp', 'Resp']:
    v = df_normal[f].dropna()
    print(f"    {f}: mean={v.mean():.2f} std={v.std():.2f}")

# 2. Moderate drift - respiratory features shifted
moderate_shifts = {
    'O2Sat': (-3.0, 1.0),    # O2Sat drops by ~3 points
    'Resp': (4.0, 1.0),      # Resp increases by ~4
    'FiO2': (0.1, 1.0),      # FiO2 slightly up
    'PaCO2': (5.0, 1.0),     # PaCO2 up
}
df_moderate = generate_batch(200, shifts=moderate_shifts, name="moderate")
path_moderate = ROOT / "moderate_drift.csv"
df_moderate.to_csv(path_moderate, index=False)
n_sep = df_moderate.groupby('Patient_ID')['SepsisLabel'].max().sum()
print(f"  moderate_drift.csv: {len(df_moderate)} rows, {df_moderate['Patient_ID'].nunique()} patients, {n_sep} sepsis")
for f in ['HR', 'O2Sat', 'Temp', 'Resp']:
    v = df_moderate[f].dropna()
    print(f"    {f}: mean={v.mean():.2f} std={v.std():.2f}")

# 3. Severe drift - multi-system
severe_shifts = {
    'O2Sat': (-8.0, 1.0),
    'Resp': (8.0, 1.0),
    'FiO2': (0.2, 1.0),
    'PaCO2': (10.0, 1.0),
    'Lactate': (3.0, 1.0),
    'WBC': (5.0, 1.0),
    'Creatinine': (1.0, 1.0),
    'Fibrinogen': (150.0, 1.0),
    'SBP': (-20.0, 1.0),
    'MAP': (-15.0, 1.0),
}
df_severe = generate_batch(200, shifts=severe_shifts, label_flip_rate=0.03, name="severe")
path_severe = ROOT / "severe_drift.csv"
df_severe.to_csv(path_severe, index=False)
n_sep = df_severe.groupby('Patient_ID')['SepsisLabel'].max().sum()
print(f"  severe_drift.csv: {len(df_severe)} rows, {df_severe['Patient_ID'].nunique()} patients, {n_sep} sepsis")
for f in ['HR', 'O2Sat', 'Temp', 'Resp', 'Lactate', 'WBC']:
    v = df_severe[f].dropna()
    print(f"    {f}: mean={v.mean():.2f} std={v.std():.2f}")

print("\nDone! Files saved to:", ROOT)
