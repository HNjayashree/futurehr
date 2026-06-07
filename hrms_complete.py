"""
╔══════════════════════════════════════════════════════════════════════╗
║         AI-POWERED HRMS — COMPLETE ML SYSTEM                       ║
║         Using Real Kaggle Datasets + Multiple Algorithms            ║
║         All Trained Models Saved as .pkl Files                      ║
╚══════════════════════════════════════════════════════════════════════╝

DATASETS USED:
  1. WA_Fn-UseC_-HR-Employee-Attrition.csv   → Attrition + Performance
  2. MFG10YearTerminationData.csv            → Leave / Termination
  3. HRDataset_v14.csv                       → Employee Management
  4. UpdatedResumeDataSet.csv                → Resume Screening
  5. Salary Data.csv                         → Salary Recommendation
  6. train.csv                               → Recruitment Fit

MODELS TRAINED (Multiple algorithms per feature):
  ✅ Feature 1: Attrition       → Logistic Regression + Random Forest + SVM
  ✅ Feature 2: Performance     → Decision Tree + Gradient Boosting + KNN
  ✅ Feature 3: Salary          → Linear Regression + Polynomial + Ridge
  ✅ Feature 4: Resume Screen   → TF-IDF + Logistic Reg + Naive Bayes
  ✅ Feature 5: Leave/Terminat  → Random Forest + Decision Tree
  ✅ Feature 6: Recruitment     → SVM + Extra Trees
  ✅ Feature 7: Anomaly Detect  → Isolation Forest (Unsupervised)
  ✅ Feature 8: Clustering      → K-Means Skill Gap

ALL MODELS SAVED AS .pkl FILES IN: ./saved_models/
"""

import os
import sys
import warnings
import json
import joblib
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

# ── Create output folder ────────────────────────────────────────────────────
os.makedirs('saved_models', exist_ok=True)

# ── ML Imports ──────────────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               ExtraTreesClassifier, IsolationForest,
                               GradientBoostingRegressor)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans
from sklearn.preprocessing import (StandardScaler, LabelEncoder,
                                    PolynomialFeatures)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report,
                              mean_squared_error, r2_score)

# ═══════════════════════════════════════════════════════════════════════════
#  DATASET PATHS — UPDATE THESE TO MATCH YOUR FILE LOCATIONS
# ═══════════════════════════════════════════════════════════════════════════

PATHS = {
    'attrition':   r"C:\Users\Jayashree\Downloads\archive (21)\WA_Fn-UseC_-HR-Employee-Attrition.csv",
    'termination': r"C:\Users\Jayashree\Downloads\archive (22)\MFG10YearTerminationData.csv",
    'hr_dataset':  r"C:\Users\Jayashree\Downloads\archive (23)\HRDataset_v14.csv",
    'resume':      r"C:\Users\Jayashree\Downloads\UpdatedResumeDataSet.csv",
    'salary':      r"C:\Users\Jayashree\Downloads\archive (26)\Salary Data.csv",
    'recruitment': r"C:\Users\Jayashree\Downloads\archive (27)\train.csv",
}

# ═══════════════════════════════════════════════════════════════════════════
#  HELPER: SAFE CSV LOADER
# ═══════════════════════════════════════════════════════════════════════════

def load_csv(key):
    path = PATHS[key]
    if not os.path.exists(path):
        print(f"  ⚠️  File not found: {path}")
        print(f"      Please check the path and try again.")
        return None
    try:
        df = pd.read_csv(path)
        print(f"  ✅  Loaded [{key}] → {df.shape[0]} rows × {df.shape[1]} cols")
        return df
    except Exception as e:
        print(f"  ❌  Error loading [{key}]: {e}")
        return None


def save_model(obj, filename):
    """Save any object (model/scaler/encoder) as .pkl"""
    path = os.path.join('saved_models', filename)
    joblib.dump(obj, path)
    size_kb = round(os.path.getsize(path) / 1024, 1)
    print(f"      💾  Saved: saved_models/{filename}  ({size_kb} KB)")
    return path


def print_section(title):
    print(f"\n{'═'*65}")
    print(f"  🔷  {title}")
    print(f"{'═'*65}")


def print_result(model_name, metric_name, value):
    print(f"      {model_name:<35} {metric_name}: {value}")


# ═══════════════════════════════════════════════════════════════════════════
#  FEATURE 1: ATTRITION PREDICTION
#  Dataset: WA_Fn-UseC_-HR-Employee-Attrition.csv
#  Algorithms: Logistic Regression + Random Forest + SVM
# ═══════════════════════════════════════════════════════════════════════════

def train_attrition_models():
    print_section("FEATURE 1: ATTRITION PREDICTION (3 Algorithms)")
    print("  📂  Dataset: WA_Fn-UseC_-HR-Employee-Attrition.csv\n")

    df = load_csv('attrition')
    if df is None:
        return

    # ── Preprocessing ──────────────────────────────────────────────────────
    # Convert Yes/No columns to 0/1
    binary_cols = ['Attrition', 'OverTime', 'Gender']
    for col in binary_cols:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Drop columns with no useful info for ML
    drop_cols = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Encode remaining categorical columns
    le_dict = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

    # Features and target
    target = 'Attrition'
    feature_cols = [c for c in df.columns if c != target]
    X = df[feature_cols].fillna(0)
    y = df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    results = {}

    # ── Model A: Logistic Regression ───────────────────────────────────────
    print("  🔸  Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced', random_state=42)
    lr.fit(X_train, y_train)
    acc_lr = accuracy_score(y_test, lr.predict(X_test))
    cv_lr  = cross_val_score(lr, X_scaled, y, cv=5, scoring='accuracy').mean()
    print_result("Logistic Regression", "Accuracy", f"{acc_lr*100:.2f}%")
    print_result("Logistic Regression", "CV Score ", f"{cv_lr*100:.2f}%")
    save_model(lr, 'attrition_logistic_regression.pkl')
    results['LogisticRegression'] = round(acc_lr * 100, 2)

    # ── Model B: Random Forest ─────────────────────────────────────────────
    print("\n  🔸  Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10,
                                 class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    acc_rf = accuracy_score(y_test, rf.predict(X_test))
    cv_rf  = cross_val_score(rf, X_scaled, y, cv=5, scoring='accuracy').mean()
    print_result("Random Forest", "Accuracy", f"{acc_rf*100:.2f}%")
    print_result("Random Forest", "CV Score ", f"{cv_rf*100:.2f}%")
    # Top features
    importances = sorted(zip(feature_cols, rf.feature_importances_),
                         key=lambda x: x[1], reverse=True)[:5]
    print(f"      Top Risk Factors: {[f[0] for f in importances]}")
    save_model(rf, 'attrition_random_forest.pkl')
    results['RandomForest'] = round(acc_rf * 100, 2)

    # ── Model C: SVM ───────────────────────────────────────────────────────
    print("\n  🔸  Training SVM (RBF Kernel)...")
    svm = SVC(kernel='rbf', C=1.0, probability=True,
              class_weight='balanced', random_state=42)
    svm.fit(X_train, y_train)
    acc_svm = accuracy_score(y_test, svm.predict(X_test))
    print_result("SVM RBF", "Accuracy", f"{acc_svm*100:.2f}%")
    save_model(svm, 'attrition_svm.pkl')
    results['SVM'] = round(acc_svm * 100, 2)

    # Save shared preprocessing artifacts
    save_model(scaler,   'attrition_scaler.pkl')
    save_model(feature_cols, 'attrition_feature_cols.pkl')

    best = max(results, key=results.get)
    print(f"\n  🏆  Best Model: {best} ({results[best]}%)")
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  FEATURE 2: PERFORMANCE PREDICTION
#  Dataset: HRDataset_v14.csv
#  Algorithms: Decision Tree + Gradient Boosting + KNN
# ═══════════════════════════════════════════════════════════════════════════

def train_performance_models():
    print_section("FEATURE 2: PERFORMANCE PREDICTION (3 Algorithms)")
    print("  📂  Dataset: HRDataset_v14.csv\n")

    df = load_csv('hr_dataset')
    if df is None:
        return

    # Find performance column (varies by dataset version)
    perf_col = None
    for c in ['PerformanceScore', 'PerfScoreID', 'Performance Score']:
        if c in df.columns:
            perf_col = c
            break
    if perf_col is None:
        print("  ⚠️  Performance column not found. Available:", df.columns.tolist())
        return

    # Encode categoricals
    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    df[perf_col] = LabelEncoder().fit_transform(df[perf_col].astype(str))

    feature_cols = [c for c in df.columns if c != perf_col and
                    df[c].dtype in [np.int64, np.float64]]
    feature_cols = [c for c in feature_cols if df[c].nunique() > 1]

    X = df[feature_cols].fillna(0)
    y = df[perf_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)

    results = {}

    # ── Model A: Decision Tree ─────────────────────────────────────────────
    print("  🔸  Training Decision Tree...")
    dt = DecisionTreeClassifier(max_depth=8, random_state=42)
    dt.fit(X_train, y_train)
    acc_dt = accuracy_score(y_test, dt.predict(X_test))
    print_result("Decision Tree", "Accuracy", f"{acc_dt*100:.2f}%")
    print_result("Decision Tree", "Depth   ", str(dt.get_depth()))
    save_model(dt, 'performance_decision_tree.pkl')
    results['DecisionTree'] = round(acc_dt * 100, 2)

    # ── Model B: Gradient Boosting ─────────────────────────────────────────
    print("\n  🔸  Training Gradient Boosting...")
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                     max_depth=4, random_state=42)
    gb.fit(X_train, y_train)
    acc_gb = accuracy_score(y_test, gb.predict(X_test))
    print_result("Gradient Boosting", "Accuracy", f"{acc_gb*100:.2f}%")
    save_model(gb, 'performance_gradient_boosting.pkl')
    results['GradientBoosting'] = round(acc_gb * 100, 2)

    # ── Model C: KNN ───────────────────────────────────────────────────────
    print("\n  🔸  Training K-Nearest Neighbors (k=5)...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    acc_knn = accuracy_score(y_test, knn.predict(X_test))
    print_result("KNN (k=5)", "Accuracy", f"{acc_knn*100:.2f}%")
    save_model(knn, 'performance_knn.pkl')
    results['KNN'] = round(acc_knn * 100, 2)

    save_model(scaler,       'performance_scaler.pkl')
    save_model(feature_cols, 'performance_feature_cols.pkl')

    best = max(results, key=results.get)
    print(f"\n  🏆  Best Model: {best} ({results[best]}%)")
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  FEATURE 3: SALARY RECOMMENDATION
#  Dataset: Salary Data.csv
#  Algorithms: Linear Regression + Polynomial Regression + Ridge Regression
# ═══════════════════════════════════════════════════════════════════════════

def train_salary_models():
    print_section("FEATURE 3: SALARY RECOMMENDATION (3 Algorithms)")
    print("  📂  Dataset: Salary Data.csv\n")

    df = load_csv('salary')
    if df is None:
        return

    print(f"      Columns found: {df.columns.tolist()}")

    # Find salary column
    salary_col = None
    for c in df.columns:
        if 'salary' in c.lower() or 'pay' in c.lower() or 'wage' in c.lower():
            salary_col = c
            break
    if salary_col is None:
        salary_col = df.select_dtypes(include=[np.number]).columns[-1]
    print(f"      Target column: '{salary_col}'")

    # Encode categoricals
    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    df.dropna(subset=[salary_col], inplace=True)
    y = df[salary_col]
    feature_cols = [c for c in df.columns if c != salary_col]
    X = df[feature_cols].fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    results = {}

    # ── Model A: Linear Regression ─────────────────────────────────────────
    print("  🔸  Training Linear Regression...")
    lr_scaler = StandardScaler()
    X_train_s = lr_scaler.fit_transform(X_train)
    X_test_s  = lr_scaler.transform(X_test)
    lr = LinearRegression()
    lr.fit(X_train_s, y_train)
    r2_lr   = r2_score(y_test, lr.predict(X_test_s))
    rmse_lr = np.sqrt(mean_squared_error(y_test, lr.predict(X_test_s)))
    print_result("Linear Regression", "R² Score", f"{r2_lr:.4f}")
    print_result("Linear Regression", "RMSE    ", f"{rmse_lr:,.2f}")
    save_model(lr,       'salary_linear_regression.pkl')
    save_model(lr_scaler,'salary_lr_scaler.pkl')
    results['LinearRegression'] = round(r2_lr, 4)

    # ── Model B: Polynomial Regression (degree=2) ──────────────────────────
    print("\n  🔸  Training Polynomial Regression (degree=2)...")
    poly_pipeline = Pipeline([
        ('poly',   PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler()),
        ('reg',    LinearRegression())
    ])
    poly_pipeline.fit(X_train, y_train)
    r2_poly   = r2_score(y_test, poly_pipeline.predict(X_test))
    rmse_poly = np.sqrt(mean_squared_error(y_test, poly_pipeline.predict(X_test)))
    print_result("Polynomial Regression", "R² Score", f"{r2_poly:.4f}")
    print_result("Polynomial Regression", "RMSE    ", f"{rmse_poly:,.2f}")
    save_model(poly_pipeline, 'salary_polynomial_regression.pkl')
    results['PolynomialRegression'] = round(r2_poly, 4)

    # ── Model C: Ridge Regression ──────────────────────────────────────────
    print("\n  🔸  Training Ridge Regression (alpha=1.0)...")
    ridge_scaler = StandardScaler()
    X_train_r = ridge_scaler.fit_transform(X_train)
    X_test_r  = ridge_scaler.transform(X_test)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_r, y_train)
    r2_ridge   = r2_score(y_test, ridge.predict(X_test_r))
    rmse_ridge = np.sqrt(mean_squared_error(y_test, ridge.predict(X_test_r)))
    print_result("Ridge Regression", "R² Score", f"{r2_ridge:.4f}")
    print_result("Ridge Regression", "RMSE    ", f"{rmse_ridge:,.2f}")
    save_model(ridge,        'salary_ridge_regression.pkl')
    save_model(ridge_scaler, 'salary_ridge_scaler.pkl')
    save_model(feature_cols, 'salary_feature_cols.pkl')
    results['Ridge'] = round(r2_ridge, 4)

    best = max(results, key=results.get)
    print(f"\n  🏆  Best Model (highest R²): {best} ({results[best]})")
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  FEATURE 4: RESUME SCREENING
#  Dataset: UpdatedResumeDataSet.csv
#  Algorithms: TF-IDF + Logistic Regression / Naive Bayes
# ═══════════════════════════════════════════════════════════════════════════

def train_resume_models():
    print_section("FEATURE 4: RESUME SCREENING (2 Algorithms)")
    print("  📂  Dataset: UpdatedResumeDataSet.csv\n")

    df = load_csv('resume')
    if df is None:
        return

    print(f"      Columns found: {df.columns.tolist()}")

    # Find text and label columns
    text_col  = None
    label_col = None
    for c in df.columns:
        if 'resume' in c.lower() or 'text' in c.lower() or 'content' in c.lower():
            text_col = c
        if 'category' in c.lower() or 'label' in c.lower() or 'class' in c.lower():
            label_col = c

    if text_col is None:
        text_col = df.columns[1]   # usually second column
    if label_col is None:
        label_col = df.columns[0]  # usually first column

    print(f"      Text column   : '{text_col}'")
    print(f"      Label column  : '{label_col}'")

    df.dropna(subset=[text_col, label_col], inplace=True)
    X = df[text_col].astype(str)
    le = LabelEncoder()
    y  = le.fit_transform(df[label_col].astype(str))

    print(f"      Categories    : {list(le.classes_)}")
    print(f"      Total resumes : {len(df)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    results = {}

    # ── Model A: TF-IDF + Logistic Regression ──────────────────────────────
    print("\n  🔸  Training TF-IDF + Logistic Regression...")
    lr_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=5000,
                                   stop_words='english')),
        ('clf',   LogisticRegression(max_iter=1000, C=1.0, random_state=42))
    ])
    lr_pipeline.fit(X_train, y_train)
    acc_lr = accuracy_score(y_test, lr_pipeline.predict(X_test))
    print_result("TF-IDF + LR", "Accuracy", f"{acc_lr*100:.2f}%")
    save_model(lr_pipeline, 'resume_tfidf_logistic.pkl')
    results['TFIDF_LogisticReg'] = round(acc_lr * 100, 2)

    # ── Model B: TF-IDF + Naive Bayes ──────────────────────────────────────
    print("\n  🔸  Training TF-IDF + Multinomial Naive Bayes...")
    nb_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=5000,
                                   stop_words='english')),
        ('clf',   MultinomialNB(alpha=0.1))
    ])
    nb_pipeline.fit(X_train, y_train)
    acc_nb = accuracy_score(y_test, nb_pipeline.predict(X_test))
    print_result("TF-IDF + Naive Bayes", "Accuracy", f"{acc_nb*100:.2f}%")
    save_model(nb_pipeline, 'resume_tfidf_naivebayes.pkl')
    results['TFIDF_NaiveBayes'] = round(acc_nb * 100, 2)

    save_model(le, 'resume_label_encoder.pkl')

    best = max(results, key=results.get)
    print(f"\n  🏆  Best Model: {best} ({results[best]}%)")
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  FEATURE 5: LEAVE / TERMINATION PREDICTION
#  Dataset: MFG10YearTerminationData.csv
#  Algorithms: Random Forest + Decision Tree
# ═══════════════════════════════════════════════════════════════════════════

def train_leave_models():
    print_section("FEATURE 5: LEAVE / TERMINATION PREDICTION (2 Algorithms)")
    print("  📂  Dataset: MFG10YearTerminationData.csv\n")

    df = load_csv('termination')
    if df is None:
        return

    print(f"      Columns found: {df.columns.tolist()}")

    # Find termination / status target column
    target = None
    for c in df.columns:
        if 'term' in c.lower() or 'status' in c.lower() or 'left' in c.lower():
            target = c
            break
    if target is None:
        target = df.columns[-1]
    print(f"      Target column : '{target}'")

    # Encode categoricals
    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    df.dropna(subset=[target], inplace=True)
    feature_cols = [c for c in df.columns if c != target and
                    df[c].dtype in [np.int64, np.float64]]
    X = df[feature_cols].fillna(0)
    y = df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)

    results = {}

    # ── Model A: Random Forest ─────────────────────────────────────────────
    print("  🔸  Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10,
                                 class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    acc_rf = accuracy_score(y_test, rf.predict(X_test))
    print_result("Random Forest", "Accuracy", f"{acc_rf*100:.2f}%")
    save_model(rf, 'leave_random_forest.pkl')
    results['RandomForest'] = round(acc_rf * 100, 2)

    # ── Model B: Decision Tree ─────────────────────────────────────────────
    print("\n  🔸  Training Decision Tree...")
    dt = DecisionTreeClassifier(max_depth=8, class_weight='balanced', random_state=42)
    dt.fit(X_train, y_train)
    acc_dt = accuracy_score(y_test, dt.predict(X_test))
    print_result("Decision Tree", "Accuracy", f"{acc_dt*100:.2f}%")
    save_model(dt, 'leave_decision_tree.pkl')
    results['DecisionTree'] = round(acc_dt * 100, 2)

    save_model(scaler,       'leave_scaler.pkl')
    save_model(feature_cols, 'leave_feature_cols.pkl')

    best = max(results, key=results.get)
    print(f"\n  🏆  Best Model: {best} ({results[best]}%)")
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  FEATURE 6: RECRUITMENT FIT SCORING
#  Dataset: train.csv
#  Algorithms: SVM + Extra Trees
# ═══════════════════════════════════════════════════════════════════════════

def train_recruitment_models():
    print_section("FEATURE 6: RECRUITMENT FIT SCORING (2 Algorithms)")
    print("  📂  Dataset: train.csv\n")

    df = load_csv('recruitment')
    if df is None:
        return

    print(f"      Columns found: {df.columns.tolist()}")

    # Find target column
    target = None
    for c in df.columns:
        if 'hired' in c.lower() or 'target' in c.lower() or \
           'label' in c.lower() or 'result' in c.lower():
            target = c
            break
    if target is None:
        target = df.columns[-1]
    print(f"      Target column : '{target}'")

    # Encode categoricals
    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    df.dropna(subset=[target], inplace=True)
    feature_cols = [c for c in df.columns if c != target]
    X = df[feature_cols].fillna(0)
    y = df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)

    results = {}

    # ── Model A: SVM ───────────────────────────────────────────────────────
    print("  🔸  Training SVM (RBF Kernel)...")
    svm = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
    svm.fit(X_train, y_train)
    acc_svm = accuracy_score(y_test, svm.predict(X_test))
    print_result("SVM RBF", "Accuracy", f"{acc_svm*100:.2f}%")
    save_model(svm, 'recruitment_svm.pkl')
    results['SVM'] = round(acc_svm * 100, 2)

    # ── Model B: Extra Trees ───────────────────────────────────────────────
    print("\n  🔸  Training Extra Trees Classifier...")
    et = ExtraTreesClassifier(n_estimators=100, random_state=42)
    et.fit(X_train, y_train)
    acc_et = accuracy_score(y_test, et.predict(X_test))
    print_result("Extra Trees", "Accuracy", f"{acc_et*100:.2f}%")
    save_model(et, 'recruitment_extra_trees.pkl')
    results['ExtraTrees'] = round(acc_et * 100, 2)

    save_model(scaler,       'recruitment_scaler.pkl')
    save_model(feature_cols, 'recruitment_feature_cols.pkl')

    best = max(results, key=results.get)
    print(f"\n  🏆  Best Model: {best} ({results[best]}%)")
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  FEATURE 7: ATTENDANCE ANOMALY DETECTION
#  Dataset: HRDataset_v14.csv  (using attendance/absence columns)
#  Algorithm: Isolation Forest (Unsupervised)
# ═══════════════════════════════════════════════════════════════════════════

def train_anomaly_model():
    print_section("FEATURE 7: ATTENDANCE ANOMALY DETECTION (Isolation Forest)")
    print("  📂  Dataset: HRDataset_v14.csv\n")

    df = load_csv('hr_dataset')
    if df is None:
        return

    # Find absence / attendance related columns
    absence_cols = [c for c in df.columns if
                    'absenc' in c.lower() or 'absent' in c.lower() or
                    'days' in c.lower() or 'late' in c.lower()]
    if not absence_cols:
        # Use all numeric columns as proxy
        absence_cols = df.select_dtypes(include=[np.number]).columns[:5].tolist()

    print(f"      Using columns: {absence_cols}")

    X = df[absence_cols].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(contamination=0.05, random_state=42)
    preds = iso.fit_predict(X_scaled)
    n_anomalies = int(np.sum(preds == -1))
    print_result("Isolation Forest", "Anomalies detected", str(n_anomalies))
    print_result("Isolation Forest", "Total records      ", str(len(df)))
    print_result("Isolation Forest", "Anomaly rate       ", f"{n_anomalies/len(df)*100:.1f}%")

    save_model(iso,          'anomaly_isolation_forest.pkl')
    save_model(scaler,       'anomaly_scaler.pkl')
    save_model(absence_cols, 'anomaly_feature_cols.pkl')

    return {'IsolationForest': n_anomalies}


# ═══════════════════════════════════════════════════════════════════════════
#  FEATURE 8: SKILL GAP CLUSTERING
#  Dataset: HRDataset_v14.csv  (performance + tenure + department)
#  Algorithm: K-Means
# ═══════════════════════════════════════════════════════════════════════════

def train_clustering_model():
    print_section("FEATURE 8: SKILL GAP ANALYSIS — K-Means Clustering")
    print("  📂  Dataset: HRDataset_v14.csv\n")

    df = load_csv('hr_dataset')
    if df is None:
        return

    # Encode categoricals
    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Use numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:8].tolist()
    print(f"      Using columns: {numeric_cols}")

    X = df[numeric_cols].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    dist = dict(zip(
        ['High Performer', 'Growth Potential', 'Needs Mentoring', 'Specialist'],
        [int(x) for x in np.bincount(labels)]
    ))
    print(f"      Cluster distribution: {dist}")
    print_result("K-Means (k=4)", "Inertia", f"{kmeans.inertia_:.2f}")

    save_model(kmeans,      'clustering_kmeans.pkl')
    save_model(scaler,      'clustering_scaler.pkl')
    save_model(numeric_cols,'clustering_feature_cols.pkl')

    return {'KMeans': dist}


# ═══════════════════════════════════════════════════════════════════════════
#  PREDICTION FUNCTIONS — Load saved .pkl and predict on new data
# ═══════════════════════════════════════════════════════════════════════════

def predict_attrition(employee_dict):
    """Load saved Random Forest and predict attrition for one employee"""
    try:
        model   = joblib.load('saved_models/attrition_random_forest.pkl')
        scaler  = joblib.load('saved_models/attrition_scaler.pkl')
        f_cols  = joblib.load('saved_models/attrition_feature_cols.pkl')
        X = np.array([[employee_dict.get(c, 0) for c in f_cols]])
        X_s = scaler.transform(X)
        proba = model.predict_proba(X_s)[0]
        risk  = round(float(proba[1]) * 100, 1)
        level = 'HIGH' if risk > 65 else 'MEDIUM' if risk > 35 else 'LOW'
        return {'attrition_risk_%': risk, 'risk_level': level}
    except Exception as e:
        return {'error': str(e)}


def predict_salary(employee_dict):
    """Load saved Polynomial Regression and predict salary"""
    try:
        model  = joblib.load('saved_models/salary_polynomial_regression.pkl')
        f_cols = joblib.load('saved_models/salary_feature_cols.pkl')
        X = pd.DataFrame([{c: employee_dict.get(c, 0) for c in f_cols}])
        salary = float(model.predict(X)[0])
        return {
            'recommended_salary': round(max(20000, salary), 2),
            'band_min': round(max(20000, salary) * 0.90, 2),
            'band_max': round(max(20000, salary) * 1.10, 2),
        }
    except Exception as e:
        return {'error': str(e)}


def predict_resume(resume_text):
    """Load saved TF-IDF + LR pipeline and classify resume"""
    try:
        model = joblib.load('saved_models/resume_tfidf_logistic.pkl')
        le    = joblib.load('saved_models/resume_label_encoder.pkl')
        pred  = model.predict([resume_text])[0]
        proba = model.predict_proba([resume_text])[0]
        return {
            'category':   le.inverse_transform([pred])[0],
            'confidence': round(float(max(proba)) * 100, 1),
        }
    except Exception as e:
        return {'error': str(e)}


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN — TRAIN ALL MODELS + DEMO PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "█"*65)
    print("█" + " "*20 + "AI-POWERED HRMS" + " "*28 + "█")
    print("█" + " "*15 + "Training All ML Models" + " "*26 + "█")
    print("█"*65)

    all_results = {}

    print("\n📂  Loading datasets and training models...\n")
    print("  ℹ️   Note: Each feature uses a DIFFERENT algorithm combination.")
    print("  ℹ️   All trained models are saved as .pkl files.\n")

    # Train all features
    all_results['attrition']    = train_attrition_models()
    all_results['performance']  = train_performance_models()
    all_results['salary']       = train_salary_models()
    all_results['resume']       = train_resume_models()
    all_results['leave']        = train_leave_models()
    all_results['recruitment']  = train_recruitment_models()
    all_results['anomaly']      = train_anomaly_model()
    all_results['clustering']   = train_clustering_model()

    # ── Summary ─────────────────────────────────────────────────────────────
    print_section("SUMMARY — ALL SAVED .pkl FILES")
    pkl_files = sorted(os.listdir('saved_models'))
    total_size = 0
    for f in pkl_files:
        fp   = os.path.join('saved_models', f)
        size = os.path.getsize(fp) / 1024
        total_size += size
        print(f"  ✅  {f:<50} {size:>7.1f} KB")
    print(f"\n  📦  Total models saved : {len(pkl_files)} files")
    print(f"  💾  Total size         : {total_size/1024:.2f} MB")
    print(f"  📁  Location           : ./saved_models/\n")

    # ── Demo Predictions ─────────────────────────────────────────────────────
    print_section("DEMO: LIVE PREDICTIONS FROM SAVED .pkl MODELS")

    sample = {
        'Age': 35, 'MonthlyIncome': 5000, 'OverTime': 1,
        'YearsAtCompany': 3, 'JobSatisfaction': 2,
        'WorkLifeBalance': 1, 'JobLevel': 2,
        'TotalWorkingYears': 8, 'YearsWithCurrManager': 1,
        'experience': 8, 'age': 35,
    }

    print("\n  👤  Sample Employee:", json.dumps(
        {k: v for k, v in list(sample.items())[:6]}, indent=6))

    print("\n  🔴  Attrition Risk    →", predict_attrition(sample))
    print("  💰  Salary Band       →", predict_salary(sample))
    print("  📄  Resume Screen     →", predict_resume(
        "python machine learning tensorflow data science neural networks"))

    # Save results JSON
    safe_results = {}
    for k, v in all_results.items():
        if v is not None:
            safe_results[k] = {str(kk): str(vv) for kk, vv in v.items()} if isinstance(v, dict) else str(v)

    with open('saved_models/training_results.json', 'w') as f:
        json.dump(safe_results, f, indent=2)

    print("\n" + "█"*65)
    print("█" + " "*10 + "✅  ALL MODELS TRAINED & SAVED SUCCESSFULLY" + " "*10 + "█")
    print("█" + " "*12 + "📁  Check ./saved_models/ folder" + " "*19 + "█")
    print("█" + " "*8  + "⚠️   DO NOT upload saved_models/ to GitHub" + " "*12 + "█")
    print("█"*65)


if __name__ == '__main__':
    main()
