"""
train.py
Website Visitor Lead Conversion Scorer - Model Training & Evaluation Pipeline
"""

import os
import sys

# Ensure UTF-8 output encoding on Windows console
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Configuration & Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")

DATA_FILE = os.path.join(DATA_DIR, "website_visitors.csv")
MODEL_FILE = os.path.join(MODELS_DIR, "lead_conversion_model.joblib")
CONF_MATRIX_FILE = os.path.join(REPORTS_DIR, "confusion_matrix.png")
METRICS_FILE = os.path.join(REPORTS_DIR, "metrics.json")

def ensure_directories():
    """Ensure data, models, and reports folders exist."""
    for folder in [DATA_DIR, MODELS_DIR, REPORTS_DIR]:
        os.makedirs(folder, exist_ok=True)

def generate_synthetic_dataset(num_records=1000, random_seed=42):
    """
    Generate exactly num_records realistic synthetic website visitor records.
    Ensures:
      - Exactly 1,000 records
      - Reproducible random seed
      - No missing values
      - No duplicate records
      - Realistic correlation between behaviors and lead conversion
    """
    np.random.seed(random_seed)
    
    referral_categories = ['Google', 'Social Media', 'Direct', 'Email', 'Referral']
    referral_probs = [0.36, 0.22, 0.22, 0.12, 0.08]
    
    # We will generate a slightly larger pool and deduplicate to guarantee exactly num_records unique rows
    pool_size = int(num_records * 1.5)
    
    # Session duration: lognormal distribution clipped between 20 and 900 seconds
    mu, sigma = 5.2, 0.75
    raw_duration = np.random.lognormal(mu, sigma, pool_size)
    session_duration = np.clip(raw_duration, 25, 900).astype(int)
    
    # Pages visited: Poisson + base, correlated with duration
    base_pages = np.random.poisson(lam=3.5, size=pool_size) + 1
    duration_bonus = (session_duration / 120).astype(int)
    pages_visited = np.clip(base_pages + duration_bonus, 1, 25)
    
    # Referral source
    referral_source = np.random.choice(referral_categories, size=pool_size, p=referral_probs)
    
    # Content downloads: 0 to 4
    # Higher chance if more pages visited and higher duration
    download_logits = -1.8 + 0.12 * pages_visited + (session_duration / 250.0)
    download_probs = 1 / (1 + np.exp(-download_logits))
    content_downloads = np.random.binomial(n=4, p=np.clip(download_probs * 0.45, 0.05, 0.85), size=pool_size)
    
    df_pool = pd.DataFrame({
        'session_duration': session_duration,
        'pages_visited': pages_visited,
        'referral_source': referral_source,
        'content_downloads': content_downloads
    })
    
    # Drop duplicates across features to ensure 100% unique records
    df_unique = df_pool.drop_duplicates().copy()
    
    # If less than required, add jitter to create unique records
    while len(df_unique) < num_records:
        extra_needed = num_records - len(df_unique)
        extra_duration = np.random.randint(30, 850, extra_needed)
        extra_pages = np.random.randint(1, 20, extra_needed)
        extra_source = np.random.choice(referral_categories, size=extra_needed, p=referral_probs)
        extra_downloads = np.random.choice([0, 1, 2, 3, 4], size=extra_needed, p=[0.5, 0.25, 0.15, 0.07, 0.03])
        df_extra = pd.DataFrame({
            'session_duration': extra_duration,
            'pages_visited': extra_pages,
            'referral_source': extra_source,
            'content_downloads': extra_downloads
        })
        df_unique = pd.concat([df_unique, df_extra]).drop_duplicates()
        
    df = df_unique.iloc[:num_records].copy().reset_index(drop=True)
    
    # Generate target 'converted' based on realistic logistic model
    # Source intent weighting
    source_weights = {
        'Referral': 1.1,
        'Email': 0.8,
        'Google': 0.2,
        'Direct': -0.3,
        'Social Media': -0.7
    }
    source_bonus = df['referral_source'].map(source_weights)
    
    # Latent conversion score
    # High session duration + high pages visited + content downloads strongly predict conversion
    z = (
        -3.6
        + 0.0055 * df['session_duration']
        + 0.21 * df['pages_visited']
        + 0.95 * df['content_downloads']
        + source_bonus
        + np.random.normal(0, 0.45, size=num_records)
    )
    
    prob = 1 / (1 + np.exp(-z))
    # Sample conversion based on probability
    converted = (prob >= 0.50).astype(int)
    df['converted'] = converted
    
    # Verify exact count, no nulls, no duplicates
    assert len(df) == num_records, f"Expected {num_records} rows, got {len(df)}"
    assert df.isnull().sum().sum() == 0, "Dataset contains null values"
    assert df.drop(columns=['converted']).duplicated().sum() == 0, "Dataset contains duplicate feature rows"
    
    # Save to CSV
    ensure_directories()
    df.to_csv(DATA_FILE, index=False)
    print(f"[OK] Generated synthetic dataset with exactly {len(df)} records at: {DATA_FILE}")
    print(f"    - Converted: {df['converted'].sum()} ({(df['converted'].mean()*100):.1f}%)")
    print(f"    - Non-converted: {(df['converted'] == 0).sum()} ({((1-df['converted'].mean())*100):.1f}%)")
    
    return df

def train_and_evaluate_model():
    """
    Train Random Forest Pipeline on the website visitors dataset,
    evaluate performance, and save model artifacts and reports.
    """
    ensure_directories()
    
    # Load dataset or generate if missing
    if not os.path.exists(DATA_FILE):
        print("[!] Dataset not found. Generating fresh dataset...")
        df = generate_synthetic_dataset()
    else:
        df = pd.read_csv(DATA_FILE)
        print(f"[OK] Loaded existing dataset from {DATA_FILE} ({len(df)} rows)")
        
    # Feature columns and target
    categorical_features = ['referral_source']
    numerical_features = ['session_duration', 'pages_visited', 'content_downloads']
    all_features = numerical_features + categorical_features
    target = 'converted'
    
    X = df[all_features]
    y = df[target]
    
    # 80/20 train/test split with stratification for class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"[OK] Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
    
    # Build Scikit-learn Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='drop'
    )
    
    # Random Forest Classifier
    rf_classifier = RandomForestClassifier(
        n_estimators=120,
        max_depth=7,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )
    
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', rf_classifier)
    ])
    
    # Train model
    print("[*] Training Random Forest Pipeline...")
    model_pipeline.fit(X_train, y_train)
    print("[OK] Model training completed successfully.")
    
    # Predictions & probabilities on test set
    y_pred = model_pipeline.predict(X_test)
    y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]
    
    # Evaluation Metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred))
    rec = float(recall_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    roc_auc = float(roc_auc_score(y_test, y_pred_proba))
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    
    # Feature Importances extraction
    cat_encoder = model_pipeline.named_steps['preprocessor'].named_transformers_['cat']
    encoded_cat_names = list(cat_encoder.get_feature_names_out(categorical_features))
    feature_names = numerical_features + encoded_cat_names
    importances = model_pipeline.named_steps['classifier'].feature_importances_
    
    feat_imp_list = [
        {"feature": name, "importance": round(float(imp), 4)}
        for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    ]
    
    metrics = {
        "model_name": "Random Forest Classifier",
        "total_records": len(df),
        "train_records": len(X_train),
        "test_records": len(X_test),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp)
        },
        "classification_report": report_dict,
        "feature_importances": feat_imp_list
    }
    
    # Save Metrics to JSON
    with open(METRICS_FILE, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)
    print(f"[OK] Saved model evaluation metrics to {METRICS_FILE}")
    
    # Generate and Save Confusion Matrix Heatmap
    plt.figure(figsize=(6.5, 5.2), dpi=200)
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    sns.set_theme(style="white")
    
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Purples',
        cbar=False,
        annot_kws={"size": 14, "weight": "bold"},
        xticklabels=['Not Converted (0)', 'Converted (1)'],
        yticklabels=['Not Converted (0)', 'Converted (1)']
    )
    plt.title("Lead Conversion Confusion Matrix", fontsize=14, pad=15, weight='bold', color="#1E293B")
    plt.xlabel("Predicted Class", fontsize=11, labelpad=10, weight='semibold', color="#334155")
    plt.ylabel("Actual Class", fontsize=11, labelpad=10, weight='semibold', color="#334155")
    plt.tight_layout()
    plt.savefig(CONF_MATRIX_FILE, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved confusion matrix heatmap to {CONF_MATRIX_FILE}")
    
    # Save trained model pipeline
    joblib.dump(model_pipeline, MODEL_FILE)
    print(f"[OK] Saved trained model pipeline to {MODEL_FILE}")
    
    print("\n" + "="*50)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*50)
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall:    {rec*100:.2f}%")
    print(f"F1-Score:  {f1*100:.2f}%")
    print(f"ROC-AUC:   {roc_auc*100:.2f}%")
    print("="*50)
    
    return metrics

if __name__ == "__main__":
    ensure_directories()
    # Always generate exactly 1000 records if not present, or refresh
    generate_synthetic_dataset(num_records=1000, random_seed=42)
    train_and_evaluate_model()
