import pandas as pd
import numpy as np
import matplotlib
import joblib
import os
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# Set plotting backend
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Sklearn & Statistics Libraries
# ✅ Critical fix: Import enable_iterative_imputer first
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer

from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_predict, StratifiedKFold, GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


# ==========================================
# 1. Statistical & Plotting Functions
# ==========================================

def calculate_net_benefit(y_true, y_prob, thresholds):
    """Calculate Net Benefit for DCA"""
    y_true = np.array(y_true).ravel()
    y_prob = np.array(y_prob).ravel()
    net_benefits = []
    for pt in thresholds:
        if pt >= 1.0:
            net_benefits.append(0)
            continue
        y_pred = y_prob >= pt
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        n = len(y_true)
        weight = pt / (1 - pt)
        nb = (tp / n) - (fp / n) * weight
        net_benefits.append(nb)
    return np.array(net_benefits)


def recalibrate_probability(y_prob, y_true):
    """Calibrate probabilities (Only for DCA visualization optimization)"""
    y_true = np.array(y_true).ravel()
    y_prob = np.array(y_prob).ravel()
    epsilon = 1e-9
    y_prob = np.clip(y_prob, epsilon, 1 - epsilon)
    log_odds = np.log(y_prob / (1 - y_prob))
    lr = LogisticRegression(C=1e9, solver='lbfgs')
    lr.fit(log_odds.reshape(-1, 1), y_true)
    new_log_odds = log_odds * lr.coef_[0][0] + lr.intercept_[0]
    return 1 / (1 + np.exp(-new_log_odds))


def calculate_nri_idi(y_true, y_prob_base, y_prob_new):
    """Calculate NRI and IDI"""
    y_true = np.array(y_true).ravel()
    y_prob_base = np.array(y_prob_base).ravel()
    y_prob_new = np.array(y_prob_new).ravel()

    mask = ~np.isnan(y_prob_base) & ~np.isnan(y_prob_new)
    y_true, y_prob_base, y_prob_new = y_true[mask], y_prob_base[mask], y_prob_new[mask]

    mean_new_event = np.mean(y_prob_new[y_true == 1])
    mean_old_event = np.mean(y_prob_base[y_true == 1])
    mean_new_nonevent = np.mean(y_prob_new[y_true == 0])
    mean_old_nonevent = np.mean(y_prob_base[y_true == 0])
    idi = (mean_new_event - mean_old_event) - (mean_new_nonevent - mean_old_nonevent)

    up_events = np.sum(y_prob_new[y_true == 1] > y_prob_base[y_true == 1])
    down_events = np.sum(y_prob_new[y_true == 1] < y_prob_base[y_true == 1])
    up_nonevents = np.sum(y_prob_new[y_true == 0] > y_prob_base[y_true == 0])
    down_nonevents = np.sum(y_prob_new[y_true == 0] < y_prob_base[y_true == 0])
    nri = ((up_events - down_events) / np.sum(y_true == 1)) + ((down_nonevents - up_nonevents) / np.sum(y_true == 0))
    return nri, idi


def calculate_ascvd_risk(df, sbp_col='SBP'):
    """Calculate ASCVD Score"""
    data = df.copy()
    col_map = {'age': 'age', 'gender': 'gender', 'tc': 'TC', 'hdl': 'HDL', 'sbp': sbp_col, 'hyp_rx': 'Hypertension',
               'smoke': 'smoking', 'diab': 'HbA1c'}
    for k, v in col_map.items():
        if v not in data.columns: return np.zeros(len(df))

    if data['TC'].mean() < 10: data['TC'] *= 38.67
    if data['HDL'].mean() < 5: data['HDL'] *= 38.67
    data['diab_flag'] = (data['HbA1c'] >= 6.5).astype(int)

    preds = []
    for i in range(len(data)):
        try:
            r = data.iloc[i]
            is_male = r['gender'] == 1
            ln_age, ln_tc, ln_hdl, ln_sbp = np.log(r['age']), np.log(r['TC']), np.log(r['HDL']), np.log(r[sbp_col])
            tr_sbp = ln_sbp if r['Hypertension'] == 1 else 0
            untr_sbp = ln_sbp if r['Hypertension'] == 0 else 0

            if not is_male:
                terms = -29.799 * ln_age + 4.884 * (
                            ln_age ** 2) + 13.540 * ln_tc - 3.114 * ln_age * ln_tc - 13.578 * ln_hdl + 3.149 * ln_age * ln_hdl + 2.019 * tr_sbp + 1.957 * untr_sbp + 7.574 * \
                        r['smoking'] - 1.665 * ln_age * r['smoking'] + 0.661 * r['diab_flag']
                score = 1 - 0.9665 ** np.exp(terms - (-29.18))
            else:
                terms = 12.344 * ln_age + 11.853 * ln_tc - 2.664 * ln_age * ln_tc - 7.990 * ln_hdl + 1.769 * ln_age * ln_hdl + 1.797 * tr_sbp + 1.764 * untr_sbp + 7.837 * \
                        r['smoking'] - 1.795 * ln_age * r['smoking'] + 0.658 * r['diab_flag']
                score = 1 - 0.9144 ** np.exp(terms - 61.18)
            preds.append(score)
        except:
            preds.append(np.nan)
    return np.array(preds)


def prepare_ascvd_data(df, sbp_col_name):
    """ASCVD Data Prep (Median Imputation)"""
    cols = [sbp_col_name, 'TC', 'HDL', 'HbA1c', 'age', 'gender', 'smoking', 'Hypertension']
    needed = [c for c in cols if c in df.columns]

    # ✅ Use SimpleImputer(median)
    imp = SimpleImputer(strategy='median')

    df_imp = df.copy()
    df_imp[needed] = imp.fit_transform(df[needed])
    return calculate_ascvd_risk(df_imp, sbp_col=sbp_col_name)


def plot_combined_metrics(y_true, prob_new, prob_base, title_prefix):
    """Main Plotting Function"""
    fpr_new, tpr_new, _ = roc_curve(y_true, prob_new)
    auc_new = auc(fpr_new, tpr_new)
    fpr_base, tpr_base, _ = roc_curve(y_true, prob_base)
    auc_base = auc(fpr_base, tpr_base)
    nri, idi = calculate_nri_idi(y_true, prob_base, prob_new)

    print(f"\n📊 [{title_prefix}] Results:")
    print(f"   - ASCVD AUC: {auc_base:.4f}")
    print(f"   - IMCI  AUC: {auc_new:.4f}")
    print(f"   - NRI:       {nri:.4f}")

    # DCA Settings
    prevalence = np.mean(y_true)
    x_limit = min(0.9, max(0.2, prevalence * 4))
    thresholds = np.linspace(0.01, x_limit, 100)

    # Auto-calibration (Only for DCA visualization)
    if np.mean(prob_new) > prevalence * 1.5:
        prob_new_dca = recalibrate_probability(prob_new, y_true)
        prob_base_dca = recalibrate_probability(prob_base, y_true)
    else:
        prob_new_dca = prob_new
        prob_base_dca = prob_base

    nb_new = calculate_net_benefit(y_true, prob_new_dca, thresholds)
    nb_base = calculate_net_benefit(y_true, prob_base_dca, thresholds)
    treat_all = [prevalence - (1 - prevalence) * (pt / (1 - pt)) for pt in thresholds]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.plot(fpr_base, tpr_base, 'b--', label=f'ASCVD (AUC={auc_base:.2f})')
    ax1.plot(fpr_new, tpr_new, 'r-', lw=2, label=f'IMCI (AUC={auc_new:.2f})')
    ax1.plot([0, 1], [0, 1], 'k:')
    ax1.set_title(f'{title_prefix}: ROC')
    ax1.set_xlabel('1 - Specificity');
    ax1.set_ylabel('Sensitivity')
    ax1.legend(loc='lower right');
    ax1.grid(alpha=0.3)

    ax2.plot(thresholds, nb_new, 'r-', lw=2, label='IMCI')
    ax2.plot(thresholds, nb_base, 'b--', label='ASCVD')
    ax2.plot(thresholds, treat_all, ':', color='gray', label='Treat All')
    ax2.plot(thresholds, np.zeros_like(thresholds), 'k-', label='Treat None')
    ax2.set_ylim(-0.02, max(max(nb_new), prevalence) + 0.05);
    ax2.set_xlim(0, x_limit)
    ax2.set_title(f'{title_prefix}: DCA')
    ax2.set_xlabel('Threshold');
    ax2.set_ylabel('Net Benefit')
    ax2.legend();
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def load_data_robust(filepath, target_col):
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except:
        print(f"❌ File not found: {filepath}"); return None, None
    df.columns = df.columns.str.strip()
    if target_col in df.columns:
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        df.dropna(subset=[target_col], inplace=True)
        df[target_col] = df[target_col].astype(int)
    else:
        print(f"❌ Target column missing: {target_col}"); return None, None
    return df, df[target_col]


# ==========================================
# 2. Main Program
# ==========================================
def main():
    train_file = '代谢免疫综合指标 - 对齐UKB_result（测试）.csv'
    ext_file = 'UKB数据-对齐NHANES(外部验证)_result（测试）.csv'
    target_col = 'all_cause_death_status'
    sbp_column_name = 'SBP'

    features_candidates = [
        'Lymphocyte %', 'Monocyte %', 'Neutrophil %', 'Basophil %', 'Neutrophil count',
        'Lymphocyte count', 'Albumin', 'CRP', 'BMI', 'RBC', 'Hb', 'MCV', 'MCH', 'RDW', 'MPV',
        'CVD', 'Hypertension', 'Hyperlipidemia', 'TC', 'HDL', 'HbA1c',
        'smoking', 'drinking', 'Urea nitrogen', 'Ca', 'GGT', 'uric acid', 'creatinine',
    ]

    # === STAGE 1: Training ===
    print("\n" + "=" * 50 + "\n   STAGE 1: Training\n" + "=" * 50)
    df_train, y_train = load_data_robust(train_file, target_col)
    if df_train is None: return

    # Fuzzy matching for column names
    existing_features = []
    for f in features_candidates:
        if f in df_train.columns:
            existing_features.append(f)
        elif f.replace(' ', '') in df_train.columns:
            df_train.rename(columns={f.replace(' ', ''): f}, inplace=True)
            existing_features.append(f)
        elif f.replace('%', ' %') in df_train.columns:
            df_train.rename(columns={f.replace('%', ' %'): f}, inplace=True)
            existing_features.append(f)

    # ✅ Critical: Record feature order used in training
    X_train = df_train[existing_features]
    train_feature_names = X_train.columns.tolist()  # Save this list

    print(f"✅ Matched Features: {len(existing_features)}")

    # Train Pipeline
    print("🚀 Starting GridSearch (SimpleImputer)...")
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('selector', SelectFromModel(LassoCV(cv=5, random_state=42), max_features=5)),
        ('smote', SMOTE(random_state=42, sampling_strategy=0.6, k_neighbors=3)),
        ('classifier', LogisticRegression(random_state=42, max_iter=10000, solver='liblinear', fit_intercept=False))
    ])

    param_grid = {
        'smote__sampling_strategy': [0.5, 0.6],
        'classifier__C': np.logspace(-3, 1, 15),
        'classifier__penalty': ['l1', 'l2']
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(pipeline, param_grid, scoring='roc_auc', cv=cv, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print(f"🎉 Best Parameters: {grid_search.best_params_}")

    # Extract selected features
    selector = best_model.named_steps['selector']
    scaler = best_model.named_steps['scaler']
    clf = best_model.named_steps['classifier']
    feature_mask = selector.get_support()
    selected_feats = X_train.columns[feature_mask].tolist()

    real_coefs = clf.coef_[0] / scaler.scale_[feature_mask]
    implicit_const = -np.sum((clf.coef_[0] * scaler.mean_[feature_mask]) / scaler.scale_[feature_mask])

    print(f"✅ LASSO Selected Features ({len(selected_feats)}): {selected_feats}")

    # === 🖨️ Print Formula (Added Block) ===
    print("\n" + "=" * 50)
    print("📋 IMCI Calculation Formula (For Raw Data)")
    print("=" * 50)
    print(f"IMCI Score = {implicit_const:.5f} (Intercept)")

    formula_terms = []
    for name, coef in zip(selected_feats, real_coefs):
        sign = "+" if coef >= 0 else "-"
        abs_coef = abs(coef)
        print(f"   {sign} ({abs_coef:.5f} * {name})")
        formula_terms.append(f"{sign} {abs_coef:.4f}*{name}")

    print("-" * 50)
    print(f"Risk Probability P = 1 / (1 + exp(-( {implicit_const:.4f} {' '.join(formula_terms)} )))")
    print("=" * 50 + "\n")
    # =======================================

    # Internal Validation Plot
    prob_imci_cv = cross_val_predict(best_model, X_train, y_train, cv=cv, method='predict_proba')[:, 1]
    train_ascvd = prepare_ascvd_data(df_train, sbp_column_name)
    lr_ascvd = LogisticRegression();
    lr_ascvd.fit(np.log(train_ascvd + 1e-9).reshape(-1, 1), y_train)
    prob_ascvd_cv = cross_val_predict(lr_ascvd, np.log(train_ascvd + 1e-9).reshape(-1, 1), y_train, cv=cv,
                                      method='predict_proba')[:, 1]

    plot_combined_metrics(y_train, prob_imci_cv, prob_ascvd_cv, "Internal Validation")

    # === STAGE 2: External Validation ===
    print("\n" + "=" * 50 + "\n   STAGE 2: External Validation\n" + "=" * 50)
    df_ext, y_ext = load_data_robust(ext_file, target_col)
    if df_ext is None: return

    # Align columns
    for f in existing_features:  # Iterate all candidate features
        if f not in df_ext.columns:
            if f.replace(' ', '') in df_ext.columns:
                df_ext.rename(columns={f.replace(' ', ''): f}, inplace=True)
            elif f.replace(' %', '%') in df_ext.columns:
                df_ext.rename(columns={f.replace(' %', '%'): f}, inplace=True)

    # ✅ Critical fix: Construct full feature matrix
    missing_cols = set(train_feature_names) - set(df_ext.columns)
    if missing_cols:
        print(f"⚠️ Warning: External validation set missing columns, filling with 0: {missing_cols}")
        for c in missing_cols: df_ext[c] = 0

    X_ext_full = df_ext[train_feature_names].copy()  # Ensure order matches training

    # Unit Conversion
    if 'Ca' in X_ext_full.columns and X_ext_full['Ca'].mean() > 5.0: X_ext_full['Ca'] *= 0.25
    if 'Albumin' in X_ext_full.columns and X_ext_full['Albumin'].mean() < 10.0: X_ext_full['Albumin'] *= 10

    # ✅ Use trained imputer on full matrix
    trained_imputer = best_model.named_steps['imputer']
    X_ext_full_val = trained_imputer.transform(X_ext_full)
    X_ext_full_imp = pd.DataFrame(X_ext_full_val, columns=train_feature_names)

    # ✅ Extract LASSO selected features from imputed matrix
    X_ext_selected = X_ext_full_imp[selected_feats]

    score_ext = implicit_const + X_ext_selected.dot(real_coefs)
    prob_imci_ext = 1 / (1 + np.exp(-score_ext))

    # Calculate External ASCVD
    ext_ascvd = prepare_ascvd_data(df_ext, sbp_column_name)

    plot_combined_metrics(y_ext, prob_imci_ext, ext_ascvd, "External Validation")


if __name__ == "__main__":
    main()