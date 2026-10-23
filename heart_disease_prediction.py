"""
============================================================
HEART DISEASE RISK PREDICTION
FODS (Foundations of Data Science) — Course Project
============================================================
Dataset : UCI Cleveland Heart Disease Dataset
          303 instances | 13 features | Binary target
Source  : UCI Machine Learning Repository
Models  : Logistic Regression, Decision Tree,
          Random Forest, K-Nearest Neighbors
============================================================
"""

# ============================================================
# SECTION 1 — IMPORTS & CONFIGURATION
# ============================================================

import os
import urllib.request
import warnings
import joblib

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')          # Non-interactive backend (saves to files)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report,
)

warnings.filterwarnings('ignore')

# -- Paths & constants
PLOTS_DIR   = 'plots'
DATA_DIR    = 'data'
DATA_FILE   = os.path.join(DATA_DIR, 'heart.csv')
REPORT_FILE = 'report.txt'

DATA_URL = (
    'https://archive.ics.uci.edu/ml/machine-learning-databases/'
    'heart-disease/processed.cleveland.data'
)

COLUMN_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol',
    'fbs', 'restecg', 'thalach', 'exang',
    'oldpeak', 'slope', 'ca', 'thal', 'target',
]

RANDOM_STATE = 42

# -- Plot style
sns.set_theme(style='whitegrid', palette='Set2')
plt.rcParams.update({
    'figure.dpi': 120,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
})


# ============================================================
# SECTION 2 — DATA ACQUISITION
# ============================================================

def load_data() -> pd.DataFrame:
    """
    Download (once) and load the UCI Cleveland Heart Disease dataset.
    Returns a raw DataFrame with proper column names and '?' as NaN.
    """
    print("\n" + "=" * 60)
    print("PHASE 1 — DATA ACQUISITION")
    print("=" * 60)

    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(DATA_FILE):
        print("[INFO] Downloading dataset from UCI ML Repository...")
        try:
            urllib.request.urlretrieve(DATA_URL, DATA_FILE)
            print(f"[INFO] Dataset saved to: {DATA_FILE}")
        except Exception as e:
            raise RuntimeError(
                f"[ERROR] Could not download dataset: {e}\n"
                "Please download manually from:\n"
                "  https://archive.ics.uci.edu/ml/machine-learning-databases/"
                "heart-disease/processed.cleveland.data\n"
                f"and save it to: {DATA_FILE}"
            )
    else:
        print(f"[INFO] Dataset already cached at: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE, names=COLUMN_NAMES, na_values='?')
    print(f"[INFO] Loaded dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nFirst 5 rows:")
    print(df.head().to_string())
    print("\nData types:")
    print(df.
    dtypes.to_string())
    return df


# ============================================================
# SECTION 3 — DATA CLEANING
# ============================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset:
      - Report and handle missing values (drop rows with ? in ca/thal)
      - Remove duplicate rows
      - Binarize the target column (0 = healthy, 1 = disease)
      - Detect and report outliers using the IQR method
    """
    print("\n" + "=" * 60)
    print("PHASE 2 — DATA CLEANING")
    print("=" * 60)

    # -- Basic info
    print("\nDataset Info:")
    print(f"  Shape : {df.shape}")
    print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")

    print("\nDescriptive Statistics:")
    print(df.describe().round(2).to_string())

    # -- Missing values
    print("\nMissing values per column:")
    missing = df.isnull().sum()
    print(missing[missing > 0].to_string())

    rows_before = len(df)
    df = df.dropna()
    rows_dropped = rows_before - len(df)
    print(f"\n[CLEAN] Dropped {rows_dropped} rows with missing values. "
          f"Remaining: {len(df)} rows")

    # -- Duplicate rows
    dup_count = df.duplicated().sum()
    print(f"\n[CLEAN] Duplicate rows found: {dup_count}")
    if dup_count > 0:
        df = df.drop_duplicates()
        print(f"[CLEAN] After removing duplicates: {len(df)} rows")

    # -- Binarize target
    # Original: 0 = no disease, 1/2/3/4 = disease present
    df['target'] = (df['target'] > 0).astype(int)
    print("\n[CLEAN] Target column binarized:")
    target_counts = df['target'].value_counts()
    print(f"  0 (Healthy) : {target_counts.get(0, 0)} patients "
          f"({target_counts.get(0, 0)/len(df)*100:.1f}%)")
    print(f"  1 (Disease) : {target_counts.get(1, 0)} patients "
          f"({target_counts.get(1, 0)/len(df)*100:.1f}%)")

    # -- Outlier detection (IQR) — report only, do not remove
    print("\n[CLEAN] Outlier detection using IQR method:")
    numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    outlier_report = []
    for col in numeric_cols:
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_out = ((df[col] < lower) | (df[col] > upper)).sum()
        outlier_report.append({
            'Feature'     : col,
            'Q1'          : round(Q1, 2),
            'Q3'          : round(Q3, 2),
            'IQR'         : round(IQR, 2),
            'Lower Fence' : round(lower, 2),
            'Upper Fence' : round(upper, 2),
            'Outliers'    : n_out,
        })
    outlier_df = pd.DataFrame(outlier_report)
    print(outlier_df.to_string(index=False))
    print("[NOTE] Outliers retained — values are clinically valid and dataset is small.")

    return df.reset_index(drop=True)


# ============================================================
# SECTION 4 — EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

def eda(df: pd.DataFrame) -> None:
    """
    Perform exploratory data analysis and save 5 visualizations:
      1. target_distribution.png   — Class balance bar chart
      2. age_distribution.png      — Age histogram by class
      3. feature_vs_target.png     — Boxplots of numeric features vs target
      4. categorical_vs_target.png — Count plots of categorical features
      5. correlation_heatmap.png   — Correlation matrix heatmap
    """
    print("\n" + "=" * 60)
    print("PHASE 3 — EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # -- Plot 1: Target Class Distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df['target'].value_counts().sort_index()
    bars = ax.bar(
        ['Healthy (0)', 'Disease (1)'],
        counts.values,
        color=['#2ecc71', '#e74c3c'],
        edgecolor='white', linewidth=1.5, width=0.5,
    )
    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f'{count}\n({count/len(df)*100:.1f}%)',
            ha='center', va='bottom', fontweight='bold',
        )
    ax.set_title('Target Class Distribution\n(Heart Disease Presence)', pad=12)
    ax.set_ylabel('Number of Patients')
    ax.set_ylim(0, max(counts.values) * 1.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    _save_plot('target_distribution.png')

    # -- Plot 2: Age Distribution by Class
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, color, name in [(0, '#2ecc71', 'Healthy'),
                                (1, '#e74c3c', 'Disease')]:
        subset = df[df['target'] == label]['age']
        ax.hist(subset, bins=15, alpha=0.6, color=color, label=name,
                edgecolor='white', linewidth=0.8)
    ax.set_title('Age Distribution by Heart Disease Status')
    ax.set_xlabel('Age (years)')
    ax.set_ylabel('Count')
    ax.legend(title='Status')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    _save_plot('age_distribution.png')

    # -- Plot 3: Numeric Features vs Target (Boxplots)
    numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, feat in enumerate(numeric_features):
        data_by_class = [
            df[df['target'] == 0][feat].values,
            df[df['target'] == 1][feat].values,
        ]
        bp = axes[i].boxplot(
            data_by_class,
            patch_artist=True,
            tick_labels=['Healthy', 'Disease'],
            medianprops=dict(color='black', linewidth=2),
        )
        bp['boxes'][0].set_facecolor('#2ecc7180')
        bp['boxes'][1].set_facecolor('#e74c3c80')
        axes[i].set_title(feat.replace('_', ' ').title())
        axes[i].set_ylabel('Value')
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)

    axes[-1].set_visible(False)   # Hide the empty 6th subplot
    fig.suptitle('Numeric Features vs Heart Disease Status',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    _save_plot('feature_vs_target.png', bbox_inches='tight')

    # -- Plot 4: Categorical Features vs Target
    cat_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope']
    cat_labels = {
        'sex'    : 'Sex (0=F, 1=M)',
        'cp'     : 'Chest Pain Type (1-4)',
        'fbs'    : 'Fasting Blood Sugar > 120',
        'restecg': 'Resting ECG (0/1/2)',
        'exang'  : 'Exercise-induced Angina',
        'slope'  : 'ST Slope (1-3)',
    }
    labels = {0: 'Healthy', 1: 'Disease'}
    palette = {0: '#2ecc71', 1: '#e74c3c'}

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, feat in enumerate(cat_features):
        ct = df.groupby([feat, 'target']).size().reset_index(name='count')
        for t_val, grp in ct.groupby('target'):
            axes[i].bar(
                grp[feat].astype(str),
                grp['count'],
                label=labels[t_val],
                color=palette[t_val],
                alpha=0.8,
                width=0.35,
                align='edge' if t_val == 1 else 'center',
            )
        axes[i].set_title(cat_labels[feat])
        axes[i].set_xlabel('Category')
        axes[i].set_ylabel('Count')
        axes[i].legend(title='Status', fontsize=9)
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)

    fig.suptitle('Categorical Features vs Heart Disease Status',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save_plot('categorical_vs_target.png')

    # -- Plot 5: Correlation Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr(numeric_only=False)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt='.2f',
        cmap='RdYlGn', center=0,
        linewidths=0.5, linecolor='white',
        square=True, cbar_kws={'shrink': 0.8},
        ax=ax,
    )
    ax.set_title('Feature Correlation Matrix', pad=14, fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save_plot('correlation_heatmap.png', bbox_inches='tight')

    # -- Print key EDA insights
    target_corr = df.corr(numeric_only=False)['target'].drop('target').abs().sort_values(ascending=False)
    print("\n[EDA] Top features correlated with target (absolute Pearson r):")
    for feat, val in target_corr.head(7).items():
        print(f"  {feat:<12} : {val:.3f}")

    print("\n[EDA] Key Insights:")
    avg_age_h = df[df['target'] == 0]['age'].mean()
    avg_age_d = df[df['target'] == 1]['age'].mean()
    print(f"  1. Avg age  — Healthy: {avg_age_h:.1f} yrs | Disease: {avg_age_d:.1f} yrs")
    pct_m = df[df['sex'] == 1]['target'].mean() * 100
    pct_f = df[df['sex'] == 0]['target'].mean() * 100
    print(f"  2. Disease  — Males: {pct_m:.1f}% | Females: {pct_f:.1f}%")
    class_ratio = df['target'].value_counts()
    print(f"  3. Balance  — Healthy: {class_ratio[0]} | Disease: {class_ratio[1]} "
          f"(ratio {class_ratio[0]/class_ratio[1]:.2f}:1)")
    print(f"  4. Top corr — '{target_corr.index[0]}' (r={target_corr.iloc[0]:.3f})")
    exang_pct = df[df['exang'] == 1]['target'].mean() * 100
    print(f"  5. Angina   — {exang_pct:.1f}% of exercise-angina patients have disease")


def _save_plot(filename: str, **kwargs) -> None:
    """Helper: save current figure to PLOTS_DIR and close it."""
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, **kwargs)
    plt.close()
    print(f"[EDA] Saved: {path}")


# ============================================================
# SECTION 5 — FEATURE ENGINEERING & SELECTION
# ============================================================

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer new features and encode categoricals:
      - Derive: age_band, risk_score
      - One-hot encode: cp, thal, slope, restecg
      - Binary categoricals (sex, fbs, exang) kept as-is (already 0/1)
    Returns the transformed DataFrame ready for modelling.
    """
    print("\n" + "=" * 60)
    print("PHASE 4 — FEATURE ENGINEERING & SELECTION")
    print("=" * 60)

    df = df.copy()

    # -- Derived features

    # Age band: <40=0, 40-55=1, 55-70=2, >70=3
    df['age_band'] = pd.cut(
        df['age'],
        bins=[0, 40, 55, 70, 120],
        labels=[0, 1, 2, 3],
        right=True,
    ).astype(int)

    # Composite risk score: sum of key binary risk indicators
    df['risk_score'] = (
        df['exang'].astype(int)               # exercise-induced angina
        + df['fbs'].astype(int)               # high fasting blood sugar
        + (df['oldpeak'] > 1.5).astype(int)   # significant ST depression
        + (df['ca'] > 0).astype(int)          # vessels with narrowing
    )

    print(f"[FEAT] 'age_band' distribution: {df['age_band'].value_counts().sort_index().to_dict()}")
    print(f"[FEAT] 'risk_score' distribution: {df['risk_score'].value_counts().sort_index().to_dict()}")

    # -- One-hot encoding of multi-category features
    df = pd.get_dummies(df, columns=['cp'],      prefix='cp',      drop_first=True)
    df = pd.get_dummies(df, columns=['thal'],    prefix='thal',    drop_first=True)
    df = pd.get_dummies(df, columns=['slope'],   prefix='slope',   drop_first=True)
    df = pd.get_dummies(df, columns=['restecg'], prefix='restecg', drop_first=True)

    # Convert bool columns produced by get_dummies to int
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    print(f"[FEAT] Dataset shape after encoding: {df.shape}")
    feat_cols = [c for c in df.columns if c != 'target']
    print(f"[FEAT] Final feature set ({len(feat_cols)} features):")
    for i, col in enumerate(feat_cols, 1):
        print(f"       {i:2d}. {col}")

    return df


# ============================================================
# SECTION 6 — MODEL BUILDING & EVALUATION
# ============================================================

def build_and_evaluate_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_names: list,
) -> dict:
    """
    Train 4 classifiers, compute all metrics, and run 5-fold CV on Recall.
    Returns a dict of results keyed by model name.
    """
    print("\n" + "=" * 60)
    print("PHASE 5 — MODEL BUILDING")
    print("=" * 60)
    print(f"  Training set : {X_train.shape[0]} samples")
    print(f"  Test set     : {X_test.shape[0]} samples")
    print(f"  Features     : {X_train.shape[1]}")

    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, C=1.0,
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=5, random_state=RANDOM_STATE,
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=None,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        'K-Nearest Neighbors': KNeighborsClassifier(
            n_neighbors=7, metric='minkowski',
        ),
    }

    results = {}

    for name, model in models.items():
        print(f"\n  [{name}]")

        # Train
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Metrics
        acc     = accuracy_score(y_test, y_pred)
        prec    = precision_score(y_test, y_pred)
        rec     = recall_score(y_test, y_pred)
        f1      = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        cm      = confusion_matrix(y_test, y_pred)

        # 5-fold Cross-Validation (Recall as scoring)
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=5, scoring='recall',
        )

        results[name] = {
            'model'          : model,
            'y_pred'         : y_pred,
            'y_proba'        : y_proba,
            'accuracy'       : acc,
            'precision'      : prec,
            'recall'         : rec,
            'f1'             : f1,
            'roc_auc'        : roc_auc,
            'confusion_matrix': cm,
            'cv_recall_mean' : cv_scores.mean(),
            'cv_recall_std'  : cv_scores.std(),
        }

        print(f"    Accuracy   : {acc:.4f}")
        print(f"    Precision  : {prec:.4f}")
        print(f"    Recall     : {rec:.4f}  <- key metric for healthcare")
        print(f"    F1-Score   : {f1:.4f}")
        print(f"    ROC-AUC    : {roc_auc:.4f}")
        print(f"    CV Recall  : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
        print()
        print(classification_report(y_test, y_pred,
                                    target_names=['Healthy', 'Disease'],
                                    digits=4))

    return results


# ============================================================
# SECTION 7 — EVALUATION VISUALIZATIONS
# ============================================================

def plot_evaluation(results: dict, y_test: pd.Series) -> None:
    """
    Generate 3 evaluation visualizations:
      1. model_comparison.png  — Grouped bar chart of all metrics
      2. confusion_matrix.png  — Heatmap for the best model (by Recall)
      3. roc_curves.png        — ROC curves for all 4 models
    """
    print("\n" + "=" * 60)
    print("PHASE 6 — EVALUATION VISUALIZATIONS")
    print("=" * 60)

    metric_keys   = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    model_names   = list(results.keys())
    short_names   = ['LR', 'DT', 'RF', 'KNN']

    # -- Plot 1: Model Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(13, 6))
    x      = np.arange(len(model_names))
    width  = 0.14
    colors = ['#3498db', '#e67e22', '#2ecc71', '#9b59b6', '#e74c3c']

    for i, (key, label) in enumerate(zip(metric_keys, metric_labels)):
        vals = [results[m][key] for m in model_names]
        bars = ax.bar(x + i * width, vals, width, label=label,
                      color=colors[i], edgecolor='white', linewidth=0.8)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f'{val:.2f}',
                ha='center', va='bottom', fontsize=7.5,
            )

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(short_names, fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score')
    ax.set_title(
        'Model Performance Comparison\n'
        '(LR=Logistic Regression | DT=Decision Tree | '
        'RF=Random Forest | KNN=K-Nearest Neighbors)',
        fontsize=12,
    )
    ax.legend(loc='upper right', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    _save_plot('model_comparison.png')

    # -- Select best model by Recall
    best_name = max(results, key=lambda m: results[m]['recall'])
    best      = results[best_name]
    print(f"[EVAL] Best model by Recall: {best_name} "
          f"(Recall = {best['recall']:.4f})")

    # -- Plot 2: Confusion Matrix (Best Model)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        best['confusion_matrix'],
        annot=True, fmt='d', cmap='Blues',
        xticklabels=['Predicted Healthy', 'Predicted Disease'],
        yticklabels=['Actual Healthy', 'Actual Disease'],
        linewidths=1, linecolor='white',
        ax=ax,
    )
    ax.set_title(
        f'Confusion Matrix — {best_name}\n'
        f'Recall = {best["recall"]:.4f}',
        fontsize=11,
    )
    plt.tight_layout()
    _save_plot('confusion_matrix.png')

    # -- Plot 3: ROC Curves
    fig, ax = plt.subplots(figsize=(7, 6))
    colors_roc = ['#3498db', '#e67e22', '#2ecc71', '#9b59b6']

    for (name, res), color, sname in zip(results.items(), colors_roc, short_names):
        fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{sname} (AUC = {res["roc_auc"]:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate (Recall)')
    ax.set_title('ROC Curves — All Models')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    _save_plot('roc_curves.png')


def plot_feature_importance(rf_model, feature_names) -> None:
    """Plot and save Random Forest feature importances (horizontal bar chart)."""
    importances = rf_model.feature_importances_
    indices     = np.argsort(importances)   # ascending for barh

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(
        range(len(importances)),
        importances[indices],
        color=plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(importances))),
        edgecolor='white', linewidth=0.5,
    )
    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=9)
    ax.set_xlabel('Gini Importance')
    ax.set_title('Feature Importances — Random Forest',
                 fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    _save_plot('feature_importance.png')


# ============================================================
# SECTION 8 — REPORT GENERATION
# ============================================================

def generate_report(
    results: dict,
    df_clean: pd.DataFrame,
    df_engineered: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """
    Generate a structured text report and save it to report.txt.
    Prints the report to the console as well.
    """
    print("\n" + "=" * 60)
    print("PHASE 7 — FINAL REPORT")
    print("=" * 60)

    best_name = max(results, key=lambda m: results[m]['recall'])
    best      = results[best_name]

    # Dynamic EDA stats
    avg_age_h      = df_clean[df_clean['target'] == 0]['age'].mean()
    avg_age_d      = df_clean[df_clean['target'] == 1]['age'].mean()
    male_disease   = df_clean[df_clean['sex'] == 1]['target'].mean() * 100
    female_disease = df_clean[df_clean['sex'] == 0]['target'].mean() * 100
    exang_disease  = df_clean[df_clean['exang'] == 1]['target'].mean() * 100
    target_corr    = (
        df_clean.corr(numeric_only=False)['target']
        .drop('target').abs().sort_values(ascending=False)
    )
    class_cnt = df_clean['target'].value_counts()

    lines = [
        "=" * 65,
        "  HEART DISEASE RISK PREDICTION — FINAL PROJECT REPORT",
        "  FODS (Foundations of Data Science) Course Project",
        "=" * 65,
        "",
        "1. PROBLEM STATEMENT",
        "   " + "-" * 50,
        "   Cardiovascular disease is the leading cause of death globally.",
        "   This project builds a machine-learning pipeline to predict the",
        "   PRESENCE of heart disease from clinical features, enabling",
        "   early identification of at-risk patients.",
        "   Binary Target: 0 = No Heart Disease | 1 = Heart Disease Present",
        "",
        "2. DATASET",
        "   " + "-" * 50,
        "   Source  : UCI Machine Learning Repository — Cleveland Heart Disease",
        "   URL     : https://archive.ics.uci.edu/dataset/45/heart+disease",
        f"   Raw rows: 303 | After cleaning: {len(df_clean)} rows",
        f"   Features: 13 raw -> {len(df_engineered.columns) - 1} after engineering",
        "",
        "   Columns:",
        "     age      — Age in years",
        "     sex      — Sex (1=Male, 0=Female)",
        "     cp       — Chest pain type (1=typical angina...4=asymptomatic)",
        "     trestbps — Resting blood pressure (mm Hg)",
        "     chol     — Serum cholesterol (mg/dl)",
        "     fbs      — Fasting blood sugar > 120 mg/dl",
        "     restecg  — Resting ECG results (0/1/2)",
        "     thalach  — Maximum heart rate achieved",
        "     exang    — Exercise-induced angina (1/0)",
        "     oldpeak  — ST depression induced by exercise",
        "     slope    — Slope of peak exercise ST segment (1/2/3)",
        "     ca       — No. of major vessels colored by fluoroscopy (0-3)",
        "     thal     — Thalassemia (3=normal, 6=fixed, 7=reversible defect)",
        "     target   — Diagnosis [BINARIZED: 0=healthy, 1=disease]",
        "",
        "3. DATA CLEANING SUMMARY",
        "   " + "-" * 50,
        "   Missing values : 6 rows dropped (ca and thal had '?' placeholder)",
        "   Duplicates     : 0 found",
        "   Target         : Binarized (1/2/3/4 -> 1 = disease present)",
        f"   Class balance  : {class_cnt[0]} healthy / {class_cnt[1]} disease",
        "   Outliers       : Detected via IQR but retained (clinically valid)",
        "",
        "4. EDA INSIGHTS",
        "   " + "-" * 50,
        f"   a) Average age: Healthy = {avg_age_h:.1f} yrs vs Disease = {avg_age_d:.1f} yrs",
        f"      Older patients are more likely to have heart disease.",
        f"   b) Disease prevalence: Males = {male_disease:.1f}% vs Females = {female_disease:.1f}%",
        f"      Males in this dataset are at higher risk.",
        f"   c) {exang_disease:.1f}% of patients with exercise-induced angina have disease",
        f"      Strong positive predictor.",
        f"   d) Top 3 features correlated with target (Pearson |r|):",
    ]
    for feat in target_corr.index[:3]:
        lines.append(f"      - {feat:<12} : {target_corr[feat]:.3f}")
    lines += [
        f"   e) 'thalach' (max heart rate) is NEGATIVELY correlated with disease",
        f"      Higher max heart rate -> less likely to have disease.",
        "",
        "5. FEATURE ENGINEERING",
        "   " + "-" * 50,
        "   - age_band   : Age binned into 4 bands (0=<40, 1=40-55, 2=55-70, 3=>70)",
        "   - risk_score : Sum of exang + fbs + (oldpeak>1.5) + (ca>0)",
        "   - One-hot encoding: cp (4->3), thal (3->2), slope (3->2), restecg (3->2)",
        "   - StandardScaler applied before model training (fit on train, transform test)",
        "",
        "6. MODEL COMPARISON",
        "   " + "-" * 50,
        f"   {'Model':<22} {'Acc':>6} {'Prec':>7} {'Recall':>8} "
        f"{'F1':>7} {'AUC':>7} {'CV-Rec':>8}",
        "   " + "-" * 68,
    ]

    for name, res in results.items():
        short = name.ljust(22)
        lines.append(
            f"   {short} {res['accuracy']:>6.4f} {res['precision']:>7.4f} "
            f"{res['recall']:>8.4f} {res['f1']:>7.4f} {res['roc_auc']:>7.4f} "
            f"{res['cv_recall_mean']:>6.4f}+/-{res['cv_recall_std']:.4f}"
        )

    lines += [
        "",
        "7. FINAL MODEL RECOMMENDATION & JUSTIFICATION",
        "   " + "-" * 50,
        "",
        f"   RECOMMENDED MODEL: {best_name}",
        f"     Recall     : {best['recall']:.4f}",
        f"     Precision  : {best['precision']:.4f}",
        f"     F1-Score   : {best['f1']:.4f}",
        f"     Accuracy   : {best['accuracy']:.4f}",
        f"     ROC-AUC    : {best['roc_auc']:.4f}",
        "",
        "   HEALTHCARE FRAMING:",
        "   In a clinical context, a FALSE NEGATIVE (missing a patient who",
        "   actually has heart disease) is far more dangerous than a false",
        "   positive (flagging a healthy patient for further tests).",
        "   - False positive -> unnecessary (but safe) follow-up",
        "   - False negative -> potentially life-threatening missed diagnosis",
        "",
        "   Therefore, RECALL is prioritised as the primary model selection metric.",
        f"   {best_name} achieves the highest recall, minimising false negatives.",
        "",
        "8. CONCLUSION & LIMITATIONS",
        "   " + "-" * 50,
        "   The pipeline demonstrates the full data-science workflow:",
        "   data acquisition -> cleaning -> EDA -> engineering -> modelling -> evaluation",
        "",
        "   Limitations:",
        "   - Small dataset (297 rows) limits generalisability",
        "   - Data collected in the 1980s; may not reflect modern populations",
        "   - No hyperparameter tuning (GridSearchCV could improve performance)",
        "   - Class imbalance is mild; SMOTE could be applied for robustness",
        "   - Explainability (SHAP) not included in this version",
        "",
        "9. OUTPUT FILES GENERATED",
        "   " + "-" * 50,
        "   plots/target_distribution.png   — Class balance bar chart",
        "   plots/age_distribution.png      — Age histogram by disease status",
        "   plots/feature_vs_target.png     — Boxplots of numeric features",
        "   plots/categorical_vs_target.png — Count plots for categorical features",
        "   plots/correlation_heatmap.png   — Full feature correlation heatmap",
        "   plots/model_comparison.png      — Multi-metric model comparison bars",
        "   plots/confusion_matrix.png      — Confusion matrix (best model)",
        "   plots/roc_curves.png            — ROC curves (all 4 models)",
        "   plots/feature_importance.png    — Random Forest feature importance",
        "   report.txt                      — This report",
        "=" * 65,
    ]

    report_text = '\n'.join(lines)
    print(report_text)

    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\n[DONE] Report saved to: {REPORT_FILE}")


# ============================================================
# SECTION 9 — MAIN EXECUTION
# ============================================================

def main() -> None:
    """
    Orchestrates all phases of the Heart Disease Risk Prediction pipeline.

    Usage:
        python heart_disease_prediction.py

    Outputs:
        data/heart.csv          — cached dataset (auto-downloaded on first run)
        plots/*.png             — 9 visualisation PNG files
        report.txt              — structured summary report
    """
    print("=" * 60)
    print("  HEART DISEASE RISK PREDICTION — FODS PROJECT")
    print("=" * 60)

    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # Phase 1: Data Acquisition
    df_raw = load_data()

    # Phase 2: Data Cleaning
    df_clean = clean_data(df_raw)

    # Phase 3: Exploratory Data Analysis
    eda(df_clean)

    # Phase 4: Feature Engineering
    df_engineered = feature_engineering(df_clean)

    # Phase 5: Train / Test Split & Feature Scaling
    print("\n" + "=" * 60)
    print("PHASE 5 — TRAIN / TEST SPLIT & SCALING")
    print("=" * 60)

    X = df_engineered.drop('target', axis=1)
    y = df_engineered['target']

    # 80/20 stratified split to preserve class proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Fit scaler on training data only (prevent data leakage)
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print(f"[SPLIT] Train: {X_train_sc.shape[0]} | Test: {X_test_sc.shape[0]} | "
          f"Features: {X_train_sc.shape[1]}")
    print("[SCALE] StandardScaler fitted on training set, applied to test set")

    # Phase 6: Model Building & Evaluation
    results = build_and_evaluate_models(
        X_train_sc, X_test_sc,
        y_train, y_test,
        feature_names=list(X.columns),
    )

    # Phase 7: Visualisations
    plot_evaluation(results, y_test)
    plot_feature_importance(
        results['Random Forest']['model'],
        feature_names=list(X.columns),
    )

    # Phase 8: Report
    generate_report(results, df_clean, df_engineered, y_test)

    # Save all models and scaler for the frontend
    print("\n" + "=" * 60)
    print("PHASE 9 — SAVING MODEL & SCALER")
    print("=" * 60)
    
    os.makedirs('models', exist_ok=True)
    
    # Save all models in a dictionary
    all_models = {name: res['model'] for name, res in results.items()}
    joblib.dump(all_models, 'models/all_models.pkl')
    
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(list(X.columns), 'models/feature_columns.pkl')
    print(f"[SAVE] All {len(all_models)} models saved to models/all_models.pkl")
    print(f"[SAVE] Scaler saved to models/scaler.pkl")
    print(f"[SAVE] Feature columns saved to models/feature_columns.pkl")

    # Final summary
    print("\n" + "=" * 60)
    print("  ALL DONE!")
    print("=" * 60)
    print(f"  Plots saved in : ./{PLOTS_DIR}/")
    print(f"  Data cached at : ./{DATA_FILE}")
    print(f"  Report saved   : ./{REPORT_FILE}")
    print("=" * 60)


if __name__ == '__main__':
    main()
