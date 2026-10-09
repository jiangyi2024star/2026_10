# -*- coding: utf-8 -*-
"""Reanalysis for manuscript revision: outcomes, VIF, multivariable logistic, bootstrap AUC."""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from imblearn.combine import SMOTEENN

root = Path(r"D:\任务2025\17-返修（过度护理）")
out = root / "_extracted"
out.mkdir(exist_ok=True)

df = pd.read_excel(root / "原始资料" / "data1（缺失值填补）.xlsx")
# rename for convenience
cols = {
    "年龄": "Age",
    "性别（男=1，女=0）": "Sex",
    "身高": "Height",
    "体重": "Weight",
    "吸烟史（0=无，1=有）": "Smoking",
    "饮酒史（0=无，1=有）": "Alcohol",
    "肝性脑病（0=无，1=有）": "HE",
    "肾功能不全（0=无，1=有）": "RF",
    "原发性腹膜炎（0=无，1=有）": "PP",
    "肺部感染（0=无，1=有）": "PNA",
    "消化道出血（0=无，1=有）": "GIB",
    "血白蛋白": "ALB",
    "总胆红素": "TBIL",
    "甲胎蛋白": "AFP",
    "HBV-DNA": "HBV_DNA",
    "ALT": "ALT",
    "PTA": "PTA",
    "AST": "AST",
    "INR": "INR",
    "过渡性护理指标1护理满意度": "TNI1",
    "过渡性护理指标2药物管理": "TNI2",
    "过渡性护理指标3护理计划的连续性": "TNI3",
    "过渡性护理指标4患者教育与自我管理": "TNI4",
    "过渡性护理指标5护理记录的完整性和准确性": "TNI5",
    "过渡性护理总分": "TNI_score",
    "出院30天内10再入院（是=1，否=0）": "Readmit30",
    "出院30天急诊访问次数": "ED30",
    "1年随访期间10死亡（是=1，否=0）": "Death1y",
    "因变量1短期预后（良好=0，差=1）": "Short",
    "因变量2长期预后（良好=0，差=1）": "Long",
}
df = df.rename(columns=cols)

lines = []
def log(s=""):
    lines.append(str(s))
    print(s)

log("=== Outcome definition check ===")
log(f"N={len(df)}")
log(f"Short poor={df['Short'].sum()} ({df['Short'].mean():.3f})")
log(f"Long poor={df['Long'].sum()} ({df['Long'].mean():.3f})")
log(f"Readmit30={df['Readmit30'].sum()}, ED30>0={(df['ED30']>0).sum()}, Death1y={df['Death1y'].sum()}")

# How is short defined?
short_or = ((df["Readmit30"] == 1) | (df["ED30"] >= 1)).astype(int)
log(f"Agreement Short vs (Readmit OR ED>=1): {(short_or==df['Short']).mean():.4f}")
log(pd.crosstab(short_or, df["Short"]))
log(f"Agreement Long vs Death1y: {(df['Death1y']==df['Long']).mean():.4f}")
log(pd.crosstab(df["Death1y"], df["Long"]))

# Missingness
log("\n=== Missingness (before fill file already filled) ===")
raw = pd.read_excel(root / "原始资料" / "data0.xlsx")
# try locate AFP INR columns in raw - messy
log("raw shape " + str(raw.shape))

# Univariate logistic for HBV-DNA
log("\n=== HBV-DNA distribution ===")
log(df["HBV_DNA"].describe())
log(f"unique count={df['HBV_DNA'].nunique()}, zeros={(df['HBV_DNA']==0).sum()}")

# VIF
log("\n=== VIF: subscales + total ===")
feat_all = ["Age","RF","PP","GIB","TBIL","ALB","PTA","TNI1","TNI2","TNI3","TNI4","TNI5","TNI_score"]
X = df[feat_all].astype(float)
Xv = sm.add_constant(X)
vif = pd.DataFrame({
    "var": X.columns,
    "VIF": [variance_inflation_factor(Xv.values, i+1) for i in range(X.shape[1])]
})
log(vif.to_string(index=False))

log("\n=== VIF: subscales only (no total) ===")
feat_sub = ["Age","RF","PP","GIB","TBIL","ALB","PTA","TNI1","TNI2","TNI3","TNI4","TNI5"]
X2 = df[feat_sub].astype(float)
Xv2 = sm.add_constant(X2)
vif2 = pd.DataFrame({
    "var": X2.columns,
    "VIF": [variance_inflation_factor(Xv2.values, i+1) for i in range(X2.shape[1])]
})
log(vif2.to_string(index=False))

log("\n=== VIF: total only (no subscales) ===")
feat_tot = ["Age","RF","PP","GIB","TBIL","ALB","PTA","TNI_score"]
X3 = df[feat_tot].astype(float)
Xv3 = sm.add_constant(X3)
vif3 = pd.DataFrame({
    "var": X3.columns,
    "VIF": [variance_inflation_factor(Xv3.values, i+1) for i in range(X3.shape[1])]
})
log(vif3.to_string(index=False))

def fit_logit(yname, predictors, title):
    log(f"\n=== Multivariable logistic: {title} -> {yname} ===")
    y = df[yname].astype(int)
    X = df[predictors].astype(float)
    Xc = sm.add_constant(X)
    try:
        model = sm.Logit(y, Xc).fit(disp=False, maxiter=200)
        res = pd.DataFrame({
            "OR": np.exp(model.params),
            "CI_low": np.exp(model.conf_int()[0]),
            "CI_high": np.exp(model.conf_int()[1]),
            "P": model.pvalues,
        })
        log(res.round(4).to_string())
        return model, res
    except Exception as e:
        log(f"FAILED: {e}")
        return None, None

# Candidates from univariate p<0.05 as in paper
short_uni = ["GIB","ALB","PTA","TNI1","TNI2","TNI3","TNI4","TNI5"]  # exclude TNI_score for multi
long_uni = ["Age","RF","PP","GIB","TBIL","ALB","PTA","TNI1","TNI3","TNI4"]  # exclude TNI_score

fit_logit("Short", short_uni, "short subscales (no total)")
fit_logit("Short", ["GIB","ALB","PTA","TNI_score"], "short total only")
fit_logit("Long", long_uni, "long subscales (no total)")
fit_logit("Long", ["Age","RF","PP","GIB","TBIL","ALB","PTA","TNI_score"], "long total only")

# ML with proper leakage-free pipeline + bootstrap AUC
def evaluate_models(yname, features, n_boot=200, seed=42):
    log(f"\n=== Leakage-free ML evaluation: {yname} features={features} ===")
    y = df[yname].astype(int).values
    X = df[features].astype(float).values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    log(f"Train {len(y_train)} ({y_train.sum()} poor), Test {len(y_test)} ({y_test.sum()} poor)")

    # SMOTE-ENN on training only
    sme = SMOTEENN(random_state=seed)
    X_res, y_res = sme.fit_resample(X_train, y_train)
    log(f"After SMOTE-ENN train: {len(y_res)} ({(y_res==1).sum()} poor / {(y_res==0).sum()} good)")

    scaler = StandardScaler()
    X_res_s = scaler.fit_transform(X_res)
    X_test_s = scaler.transform(X_test)

    models = {
        "Logistic": LogisticRegression(max_iter=2000, random_state=seed),
        "SVM": SVC(probability=True, random_state=seed),
        "RF": RandomForestClassifier(n_estimators=300, random_state=seed),
        "GBM": GradientBoostingClassifier(random_state=seed),
    }
    rows = []
    for name, clf in models.items():
        clf.fit(X_res_s, y_res)
        proba = clf.predict_proba(X_test_s)[:, 1]
        pred = (proba >= 0.5).astype(int)
        auc = roc_auc_score(y_test, proba)
        # bootstrap CI on test set scores
        rng = np.random.default_rng(seed)
        boot = []
        n = len(y_test)
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            if len(np.unique(y_test[idx])) < 2:
                continue
            boot.append(roc_auc_score(y_test[idx], proba[idx]))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({
            "Model": name,
            "ACC": accuracy_score(y_test, pred),
            "PRE": precision_score(y_test, pred, zero_division=0),
            "REC": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
            "AUC": auc,
            "AUC_CI_low": lo,
            "AUC_CI_high": hi,
        })
        log(f"{name}: AUC={auc:.4f} ({lo:.4f}-{hi:.4f}), ACC={accuracy_score(y_test, pred):.4f}")
    return pd.DataFrame(rows)

# Primary feature set WITHOUT TNI_score to address multicollinearity
feat_ml_short = ["Age","RF","PP","GIB","ALB","TBIL","PTA","TNI1","TNI2","TNI3","TNI4","TNI5"]
feat_ml_long = ["Age","RF","PP","GIB","ALB","TBIL","PTA","TNI1","TNI2","TNI3","TNI4","TNI5"]

res_s = evaluate_models("Short", feat_ml_short)
res_l = evaluate_models("Long", feat_ml_long)

# Also with original 13 including TNI_score for comparison note
feat_orig = ["Age","RF","PP","GIB","ALB","TBIL","PTA","TNI1","TNI2","TNI3","TNI4","TNI5","TNI_score"]
log("\n--- Sensitivity with TNI_score included ---")
evaluate_models("Short", feat_orig)

# Univariate re-check HBV-DNA with statsmodels
log("\n=== Univariate Short ~ HBV_DNA ===")
try:
    m = sm.Logit(df["Short"], sm.add_constant(df[["HBV_DNA"]].astype(float))).fit(disp=False)
    log(m.summary2().tables[1].to_string())
except Exception as e:
    log(e)

# Spearman among TNI
log("\n=== Spearman TNI ===")
log(df[["TNI1","TNI2","TNI3","TNI4","TNI5","TNI_score"]].corr(method="spearman").round(3).to_string())

report_path = out / "reanalysis_report.txt"
report_path.write_text("\n".join(lines), encoding="utf-8")
res_s.to_csv(out / "ml_short_leakage_free.csv", index=False)
res_l.to_csv(out / "ml_long_leakage_free.csv", index=False)
log(f"\nSaved {report_path}")
