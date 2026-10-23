# FODS Course Project: Execution Plan
## Heart Disease Risk Prediction / Diabetes Risk Prediction

---

## 1. Project Overview

| Attribute | Detail |
|---|---|
| Category | Healthcare-Based Project |
| Difficulty | Easy–Medium (recommended for low-risk, high syllabus coverage) |
| Techniques Covered | Data preprocessing, EDA, feature engineering, classification, model evaluation |
| Deliverables | Jupyter notebook / Python script, written report, optional slides |

**Why this project:** Clean, well-documented, publicly available datasets (UCI / Kaggle) mean minimal time lost to data-wrangling problems, while still covering the core FODS syllabus: cleaning, EDA, encoding, feature engineering, classification, and evaluation metrics.

---

## 2. System Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌───────────────────┐
│   Raw Dataset    │ --> │  Data Cleaning &  │ --> │   Exploratory      │
│   (CSV/UCI)       │     │  Preprocessing    │     │   Data Analysis    │
└─────────────────┘     └──────────────────┘     └───────────────────┘
                                                             │
                                                             v
┌─────────────────┐     ┌──────────────────┐     ┌───────────────────┐
│   Model           │ <-- │  Feature          │ <-- │   Visualization &   │
│   Evaluation       │     │  Engineering &    │     │   Correlation       │
│   (metrics/report) │     │  Selection        │     │   Analysis          │
└─────────────────┘     └──────────────────┘     └───────────────────┘
        │
        v
┌─────────────────┐     ┌──────────────────┐
│   Model            │ --> │   Final Report /   │
│   Training          │     │   Presentation      │
│  (Classification)  │     │   (Notebook + Slides)│
└─────────────────┘     └──────────────────┘
```

**Data flow explanation:**
1. Raw data enters the pipeline as a CSV file.
2. It passes through cleaning and preprocessing (missing values, duplicates, outliers).
3. EDA and visualization surface patterns and relationships before modeling.
4. Feature engineering transforms raw columns into model-ready inputs.
5. Multiple classification models are trained and compared.
6. The best model is evaluated with appropriate healthcare-relevant metrics.
7. Results are compiled into a final notebook and report/presentation.

---

## 3. Step-by-Step Execution Plan

### Phase 1 — Setup & Data Acquisition
- Choose dataset: **UCI Heart Disease dataset** or **Pima Indians Diabetes dataset** (Kaggle/UCI).
- Confirm dataset has a minimum of ~300–800 rows and a clear binary target column (disease present/absent).
- Set up project environment: Python, pandas, numpy, matplotlib/seaborn, scikit-learn, Jupyter Notebook.
- Load dataset into pandas and do a first-pass sanity check.

### Phase 2 — Data Cleaning
- Check `df.info()`, `df.describe()`, `df.isnull().sum()`.
- Identify and handle missing values (imputation, removal, or flagging).
- Identify invalid/placeholder values (e.g., blood pressure = 0, glucose = 0) and treat them as missing.
- Remove duplicate rows.
- Detect and treat outliers (boxplots, IQR method, or domain-based thresholds).

### Phase 3 — Exploratory Data Analysis (EDA)
- Univariate analysis: distribution plots for age, cholesterol, glucose, BMI, etc.
- Bivariate analysis: feature vs. target comparisons (boxplots, violin plots).
- Correlation heatmap to identify strongly related features.
- Class balance check (is the target variable imbalanced? e.g., 70% healthy / 30% at-risk).
- Document 3–5 key insights from this phase — these become talking points in your report.

### Phase 4 — Feature Engineering & Selection
- Encode categorical variables (one-hot encoding or label encoding as appropriate).
- Scale/normalize numeric features (StandardScaler or MinMaxScaler).
- Create derived features if useful (e.g., BMI category, age bands, risk score combinations).
- Select top features using correlation analysis or feature importance from a tree-based model.

### Phase 5 — Model Building
- Split data into train/test sets (commonly 80/20 or 70/30).
- Train a baseline model: **Logistic Regression**.
- Train comparison models: **Decision Tree**, **Random Forest**, **K-Nearest Neighbors**.
- Use cross-validation if time permits, to reduce variance in performance estimates.

### Phase 6 — Model Evaluation
- Compute Accuracy, Precision, Recall, F1-score, and ROC-AUC for each model.
- Generate a confusion matrix for the best-performing model.
- **Important framing point for healthcare projects:** Recall matters more than accuracy — a false negative (missing an at-risk patient) is more costly than a false positive. Justify your final model choice with this reasoning.
- Compare models in a summary table.

### Phase 7 — Reporting
- Compile findings into a structured notebook or written report:
  - Problem statement
  - Dataset description
  - EDA insights (with visuals)
  - Feature engineering decisions
  - Model comparison table
  - Final model justification
  - Conclusion and limitations
- Optional: prepare a slide deck summarizing the above for presentation.

---

## 4. What You Need to Bring Back

Before implementation begins, gather and share the following:

1. **Project choice** — Confirm: Heart Disease or Diabetes (or a different project from the original list).
2. **Dataset** — Either upload the CSV directly, or confirm you'd like help sourcing one from UCI/Kaggle.
3. **Course requirements / rubric** — Any mandated techniques (e.g., "must include hypothesis testing," "must compare at least 3 models") from your FODS syllabus or assignment brief.
4. **Comfort level with Python/pandas/scikit-learn** — Determines how much explanation vs. direct code you need.
5. **Deliverable format** — Jupyter notebook, plain Python script, written report with embedded code, and/or presentation slides.
6. **Deadline / time available** — Helps scope the depth (e.g., 2 models vs. 5 models, minimal EDA vs. extensive EDA).

---

## 5. Suggested Tech Stack

| Purpose | Tool/Library |
|---|---|
| Data handling | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Modeling | scikit-learn |
| Environment | Jupyter Notebook / Google Colab |
| Optional | plotly (interactive visuals), SHAP (feature importance/explainability) |

---

## 6. Next Steps

Once the items in Section 4 are provided, the next step is to build the actual working notebook covering: data loading, cleaning, EDA visualizations, feature engineering, model training, and evaluation — followed by a written summary that can be adapted into the final report.
