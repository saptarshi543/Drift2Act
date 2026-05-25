"""Generate 3 pipeline test data scenarios for the Drift2Act technical dashboard."""
import pandas as pd, numpy as np, json, os, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
np.random.seed(42)

RESP_FEATS = ["O2Sat_mean","O2Sat_std","Resp_mean","Resp_std","FiO2_mean","PaCO2_mean","PaCO2_std"]
LAB_FEATS = ["Lactate_mean","Lactate_max","WBC_mean","WBC_std","Creatinine_mean","BUN_mean","Platelets_mean","Hgb_mean"]
MIXED_FEATS = ["HR_mean","HR_std","Temp_mean","SBP_mean","MAP_mean","Glucose_mean","Calcium_mean",
               "pH_mean","Chloride_mean","Potassium_mean","DBP_mean","Hct_mean","AST_mean","Magnesium_mean"]
ALL_FEATS = RESP_FEATS + LAB_FEATS + MIXED_FEATS

ACTIONS = {0:"NO_ACTION",1:"ALERT",2:"RECALIBRATE",3:"PARTIAL_RETRAIN",4:"FULL_RETRAIN"}

def rj(feats, n=5):
    return json.dumps(list(np.random.choice(feats, min(n,len(feats)), replace=False)))

def make_alert(w, phase, lv, action, dt, rs, ea):
    return {"window_idx":w,"phase":phase,"level":lv,"action":action,"D_total":round(dt,4),
            "risk_score":round(rs,4),"est_auprc":round(ea,4),
            "alert_text":f"\u2554{'='*46}\u2557\n\u2551  DRIFT2ACT CLINICAL ALERT - Level {lv:<11}\u2551\n\u2560{'='*46}\u2563\n\u2551  Window    : {w:<34}\u2551\n\u2551  Phase     : {phase:<34}\u2551\n\u2551  D_total   : {dt:<34.4f}\u2551\n\u2551  Action    : {action:<34}\u2551\n\u255a{'='*46}\u255d"}

def make_tables(scenario_type):
    if scenario_type == "covid":
        t1 = pd.DataFrame({"Detector":["SADI (ours)","KS Test","PSI"],"Precision":[0.82,0.65,0.71],
                           "Recall":[0.95,0.80,0.75],"F1":[0.88,0.72,0.73],"Detection_Latency_Windows":[0,2,1]})
        t2 = pd.DataFrame({"Method":["SADI (ours)","KS Test","PSI"],"Precision":[0.71,0.45,0.50],"Recall":[0.86,0.57,0.43]})
    elif scenario_type == "seasonal":
        t1 = pd.DataFrame({"Detector":["SADI (ours)","KS Test","PSI"],"Precision":[0.90,0.40,0.55],
                           "Recall":[0.10,0.60,0.50],"F1":[0.18,0.48,0.52],"Detection_Latency_Windows":[0,0,0]})
        t2 = pd.DataFrame({"Method":["SADI (ours)","KS Test","PSI"],"Precision":[0.80,0.30,0.35],"Recall":[0.10,0.40,0.30]})
    else:
        t1 = pd.DataFrame({"Detector":["SADI (ours)","KS Test","PSI"],"Precision":[0.78,0.60,0.68],
                           "Recall":[0.92,0.88,0.75],"F1":[0.84,0.71,0.71],"Detection_Latency_Windows":[0,1,1]})
        t2 = pd.DataFrame({"Method":["SADI (ours)","KS Test","PSI"],"Precision":[0.67,0.40,0.55],"Recall":[0.75,0.50,0.38]})
    t3 = pd.DataFrame({"Config":["KL only","Rank shift only","Direction flip only","Equal weights","Drift2Act SADI (ours)"],
                        "alpha":[1,0,0,0.33,0.5],"beta":[0,1,0,0.33,0.3],"gamma":[0,0,1,0.34,0.2],
                        "Precision":[0.65,0.50,0.40,0.70,0.82],"Recall":[0.78,0.60,0.35,0.82,0.95],
                        "F1":[0.71,0.55,0.37,0.76,0.88]})
    return t1, t2, t3

def save_scenario(name, rows, alerts, gt, baseline_auprc, scenario_type, phases_list):
    out = ROOT / name
    os.makedirs(out / "tables", exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "drift2act_results.csv", index=False)
    with open(out / "alert_log.json","w") as f: json.dump(alerts, f, indent=2)
    with open(out / "drift_ground_truth.json","w") as f: json.dump(gt, f, indent=2)
    bm = {"xgboost":{"auprc":baseline_auprc,"auroc":0.96,"brier_score":0.04,"optimal_threshold":0.42},
          "xgboost_calibrated":{"auprc":baseline_auprc,"auroc":0.96,"brier_score":0.04,"optimal_threshold":0.42},
          "logistic_regression":{"auprc":0.48,"auroc":0.85,"brier_score":0.11,"optimal_threshold":0.50}}
    with open(out / "baseline_metrics.json","w") as f: json.dump(bm, f, indent=2)
    sadi_f1 = df[df['true_phase']!='phase1']['D_total'].apply(lambda x: 1 if x>0.3 else 0).mean() if len(df[df['true_phase']!='phase1'])>0 else 0
    ss = {"sadi_f1_mean":round(sadi_f1,3),"sadi_f1_std":0.052,"ks_f1_mean":round(sadi_f1-0.12,3),
          "ks_f1_std":0.068,"wilcoxon_p_value":0.0023,"significant":True}
    with open(out / "statistical_significance.json","w") as f: json.dump(ss, f, indent=2)
    t1,t2,t3 = make_tables(scenario_type)
    t1.to_csv(out / "tables/table1_detector_comparison.csv", index=False)
    t2.to_csv(out / "tables/table2_attribution_accuracy.csv", index=False)
    t3.to_csv(out / "tables/table3_ablation.csv", index=False)
    unique_phases = sorted(df['true_phase'].unique())
    t4_rows = []
    for p in unique_phases:
        pdata = df[df['true_phase']==p]
        t4_rows.append({"Phase":p,"Mean_DPD":round(pdata['dpd'].mean(),4),"Max_DPD":round(pdata['dpd'].max(),4),
                        "Mean_EOD":round(pdata['eod'].mean(),4),"Max_EOD":round(pdata['eod'].max(),4)})
    pd.DataFrame(t4_rows).to_csv(out / "tables/table4_fairness.csv", index=False)
    # Create ZIP
    zip_path = ROOT / f"{name}.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fp in out.rglob('*'):
            if fp.is_file():
                zf.write(fp, f"{name}/{fp.relative_to(out)}")
    print(f"  OK {name}: {len(df)} windows, {len(alerts)} alerts, ZIP={zip_path.name}")
    return df

# ══════════════════════════════════════════════════════════════════════════
# SCENARIO 1: COVID Respiratory Drift
# ══════════════════════════════════════════════════════════════════════════
print("Generating pipeline test scenarios...")
rows, alerts = [], []
for i in range(50):
    if i < 20: phase = "phase1"
    elif i < 35: phase = "phase2"
    else: phase = "phase3"
    if phase == "phase1":
        dt = 0.12 + np.random.normal(0, 0.04)
        ds = dt * 0.7; adv = 0.50 + np.random.normal(0, 0.02)
        lv = 0; ea = 0.84 + np.random.normal(0, 0.005)
        dpd = abs(np.random.normal(0.02, 0.008)); eod = abs(np.random.normal(0.015, 0.006))
        top = rj(MIXED_FEATS)
    elif phase == "phase2":
        prog = (i - 20) / 14
        dt = 0.30 + prog * 0.40 + np.random.normal(0, 0.05)
        ds = dt * 0.75; adv = 0.55 + prog * 0.15 + np.random.normal(0, 0.02)
        lv = 1 if dt < 0.45 else 2
        ea = 0.84 - prog * 0.15 + np.random.normal(0, 0.01)
        dpd = 0.03 + prog * 0.05 + abs(np.random.normal(0, 0.01))
        eod = 0.02 + prog * 0.04 + abs(np.random.normal(0, 0.008))
        top = rj(RESP_FEATS)
    else:
        dt = 0.80 + (i-35)*0.05 + np.random.normal(0, 0.08)
        ds = dt * 0.8; adv = 0.70 + np.random.normal(0, 0.05)
        lv = 3 if dt < 1.1 else 4
        ea = 0.65 - (i-35)*0.015 + np.random.normal(0, 0.01)
        dpd = 0.08 + (i-35)*0.005 + abs(np.random.normal(0, 0.012))
        eod = 0.06 + (i-35)*0.004 + abs(np.random.normal(0, 0.01))
        top = rj(RESP_FEATS + LAB_FEATS[:3])
    dt = max(0.05, dt); ds = max(0.02, ds); adv = np.clip(adv, 0.45, 0.90)
    ea = np.clip(ea, 0.40, 0.85)
    db = np.clip(0.6*min(dt,2)/2 + 0.4*max(0,(adv-0.5)*2), 0, 1)
    rs = max(0, (dt-0.3)*0.05 + np.random.normal(0, 0.003))
    row = {"window_idx":i,"window_num":i,"true_phase":phase,"D_total":round(dt,4),"D_shap":round(ds,4),
           "adv_score":round(adv,4),"drift_belief":round(db,4),"risk_score":round(rs,4),
           "est_auprc":round(ea,4),"baseline_auprc":0.84,"intervention_level":lv,
           "intervention_action":ACTIONS[lv],"top_sadi_features":top,
           "ks_detected":dt>0.4,"psi_detected":dt>0.3,
           "ks_flagged_features":rj(ALL_FEATS, np.random.randint(1,6)),
           "psi_flagged_features":rj(ALL_FEATS, np.random.randint(2,8)),
           "dpd":round(dpd,4),"eod":round(eod,4),"n_samples":200}
    rows.append(row)
    if lv >= 2:
        alerts.append(make_alert(i, phase, lv, ACTIONS[lv], dt, rs, ea))
gt = {f:{"base_feature":f.split("_")[0],"phase":"phase2","component":"respiratory"} for f in RESP_FEATS}
save_scenario("covid_respiratory_drift", rows, alerts, gt, 0.84, "covid", ["phase1","phase2","phase3"])

# ══════════════════════════════════════════════════════════════════════════
# SCENARIO 2: Seasonal Baseline (Minimal Drift)
# ══════════════════════════════════════════════════════════════════════════
rows, alerts = [], []
for i in range(40):
    dt = 0.10 + 0.06*np.sin(i*0.3) + np.random.normal(0, 0.03)
    dt = max(0.05, dt); ds = dt * 0.65
    adv = 0.50 + np.random.normal(0, 0.015)
    ea = 0.82 + np.random.normal(0, 0.004)
    db = np.clip(0.6*min(dt,2)/2 + 0.4*max(0,(adv-0.5)*2), 0, 1)
    rs = max(0, (dt-0.3)*0.03)
    dpd = abs(np.random.normal(0.015, 0.008)); eod = abs(np.random.normal(0.01, 0.005))
    row = {"window_idx":i,"window_num":i,"true_phase":"phase1","D_total":round(dt,4),"D_shap":round(ds,4),
           "adv_score":round(np.clip(adv,0.45,0.55),4),"drift_belief":round(db,4),"risk_score":round(rs,4),
           "est_auprc":round(np.clip(ea,0.79,0.84),4),"baseline_auprc":0.82,"intervention_level":0,
           "intervention_action":"NO_ACTION","top_sadi_features":rj(MIXED_FEATS),
           "ks_detected":False,"psi_detected":False,
           "ks_flagged_features":"[]","psi_flagged_features":"[]",
           "dpd":round(dpd,4),"eod":round(eod,4),"n_samples":200}
    rows.append(row)
save_scenario("seasonal_baseline", rows, [], {}, 0.82, "seasonal", ["phase1"])

# ══════════════════════════════════════════════════════════════════════════
# SCENARIO 3: Equipment Calibration Drift
# ══════════════════════════════════════════════════════════════════════════
rows, alerts = [], []
for i in range(45):
    if i < 18: phase = "phase1"
    elif i < 30: phase = "phase2"
    else: phase = "phase3"
    if phase == "phase1":
        dt = 0.12 + np.random.normal(0, 0.03); ds = dt*0.6; adv = 0.50 + np.random.normal(0, 0.02)
        lv = 0; ea = 0.80 + np.random.normal(0, 0.005)
        dpd = abs(np.random.normal(0.02, 0.008)); eod = abs(np.random.normal(0.015, 0.006))
        top = rj(MIXED_FEATS)
    elif phase == "phase2":
        if i == 18:
            dt = 1.1  # sudden spike
        elif i == 19:
            dt = 0.95
        else:
            dt = 0.60 + np.random.normal(0, 0.08)
        ds = dt*0.7; adv = (0.78 if i<=19 else 0.62+np.random.normal(0,0.03))
        lv = 4 if i<=19 else (3 if dt>0.7 else 2)
        ea = 0.70 - (i-18)*0.01 + np.random.normal(0, 0.008)
        dpd = 0.12 + np.random.normal(0, 0.02); eod = 0.09 + np.random.normal(0, 0.015)
        top = rj(LAB_FEATS)
    else:
        dt = 0.50 + np.random.normal(0, 0.06)
        ds = dt*0.65; adv = 0.60 + np.random.normal(0, 0.03)
        lv = 2 if dt > 0.5 else 1
        ea = 0.72 + np.random.normal(0, 0.008)
        dpd = 0.06 + np.random.normal(0, 0.015); eod = 0.04 + np.random.normal(0, 0.01)
        top = rj(LAB_FEATS + MIXED_FEATS[:3])
    dt = max(0.05, dt); ds = max(0.02, ds); adv = np.clip(adv, 0.45, 0.85); ea = np.clip(ea, 0.45, 0.82)
    db = np.clip(0.6*min(dt,2)/2 + 0.4*max(0,(adv-0.5)*2), 0, 1)
    rs = max(0, (dt-0.3)*0.05 + np.random.normal(0, 0.003))
    dpd = abs(dpd); eod = abs(eod)
    row = {"window_idx":i,"window_num":i,"true_phase":phase,"D_total":round(dt,4),"D_shap":round(ds,4),
           "adv_score":round(adv,4),"drift_belief":round(db,4),"risk_score":round(rs,4),
           "est_auprc":round(ea,4),"baseline_auprc":0.80,"intervention_level":lv,
           "intervention_action":ACTIONS[lv],"top_sadi_features":top,
           "ks_detected":dt>0.4,"psi_detected":dt>0.3,
           "ks_flagged_features":rj(ALL_FEATS, np.random.randint(1,6)),
           "psi_flagged_features":rj(ALL_FEATS, np.random.randint(2,8)),
           "dpd":round(dpd,4),"eod":round(eod,4),"n_samples":200}
    rows.append(row)
    if lv >= 2:
        alerts.append(make_alert(i, phase, lv, ACTIONS[lv], dt, rs, ea))
gt = {f:{"base_feature":f.split("_")[0],"phase":"phase2","component":"laboratory"} for f in LAB_FEATS}
save_scenario("equipment_calibration_drift", rows, alerts, gt, 0.80, "equipment", ["phase1","phase2","phase3"])

print("\nDone! Files saved to:", ROOT)
print("\nScenarios:")
print("  1. covid_respiratory_drift/   + .zip  (50 windows, COVID-era respiratory drift)")
print("  2. seasonal_baseline/         + .zip  (40 windows, normal seasonal variation)")
print("  3. equipment_calibration_drift/ + .zip (45 windows, lab equipment recalibration)")
print("\nTo test: Upload the .zip file in the Pipeline Analysis mode of the dashboard.")
