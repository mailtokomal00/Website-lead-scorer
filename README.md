# Website Visitor Lead Conversion Scorer 🎯

An enterprise-grade, modern Machine Learning web application designed to predict whether anonymous website visitors are likely to convert into qualified sales leads using behavioral session telemetry.

Built with **Python**, **Scikit-learn**, **Pandas**, **Plotly**, and **Streamlit**.

---

## 📌 Executive Summary & Objective

In digital marketing and B2B SaaS, sales teams frequently waste valuable time manually chasing low-intent visitors. This system utilizes a trained **Random Forest Machine Learning Pipeline** to score each visitor in real time, calculating a **0–100 Lead Score** and categorizing them into actionable business tiers (High, Medium, or Low Lead).

### The Four Core Visitor Signals
1. **Session Duration:** Time spent actively browsing the website (in seconds).
2. **Number of Pages Visited:** Total distinct URLs browsed.
3. **Referral Source:** Channel acquisition (`Google`, `Social Media`, `Direct`, `Email`, `Referral`).
4. **Number of Content Downloads:** High-intent gated resource interactions (eBooks, whitepapers, case studies).

---

## 🏆 Lead Scoring System & Business Classification

The lead score is directly calibrated from the Machine Learning model's predicted conversion probability:

| Tier | Lead Score Range | Business Category | Qualification Status | Recommended Action |
| :--- | :--- | :--- | :--- | :--- |
| 🟢 **High Lead** | **70 – 100** | High Conversion Intent | Sales Qualified Lead (SQL) | Immediate direct outreach by Senior Sales Rep within 1 hour. |
| 🟡 **Medium Lead** | **40 – 69** | Moderate Consideration | Marketing Qualified Lead (MQL) | Automated email nurture workflow with product case studies & webinar invites. |
| 🔴 **Low Lead** | **0 – 39** | Exploratory / Top of Funnel | Prospect | Passive display retargeting, blog recommendations, and newsletter signup. |

---

## 📊 Dataset Specification

- **File Name:** `data/website_visitors.csv`
- **Total Records:** Exactly **1,000 unique records**.
- **Data Quality:** Zero missing values, zero duplicate records across features.
- **Reproducibility:** Generated using a fixed random seed (`numpy.random.seed(42)`).
- **Target Variable:** `converted` (1 = Converted Lead, 0 = Non-Converted Visitor).
- **Class Balance:** ~37.4% Converted (374 records), ~62.6% Non-Converted (626 records).

> ⚠️ **Academic / Demonstration Notice:** The dataset used in this project consists of synthetic records generated for educational and demonstration purposes. It does not contain proprietary or confidential customer records.

---

## 🤖 Machine Learning Pipeline & Performance

The training workflow is implemented using an end-to-end **Scikit-learn Pipeline**:

- **Numerical Preprocessing:** `StandardScaler` applied to `session_duration`, `pages_visited`, and `content_downloads`.
- **Categorical Preprocessing:** `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` applied to `referral_source`.
- **Model Classifier:** `RandomForestClassifier(n_estimators=120, max_depth=7, class_weight='balanced', random_state=42)`.
- **Validation Split:** 80% Training (800 records), 20% Testing (200 records) stratified by class.

### Evaluation Metrics on Test Set

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **91.50%** |
| **Precision** | **84.52%** |
| **Recall** | **94.67%** |
| **F1-Score** | **89.31%** |
| **ROC-AUC** | **98.03%** |

Model evaluation reports, feature importance rankings, and confusion matrix visualizations are automatically exported to `reports/metrics.json` and `reports/confusion_matrix.png`.

---

## 📁 Project Structure

```text
website-lead-scorer/
│
├── app.py                     # Streamlit SaaS analytics web application
├── train.py                   # Data generation, pipeline training & evaluation
├── predict.py                 # Inference engine & scoring classifier (CLI & API)
├── requirements.txt           # Project dependencies
├── README.md                  # Comprehensive project documentation
│
├── data/
│   └── website_visitors.csv   # 1,000 synthetic visitor dataset
│
├── models/
│   └── lead_conversion_model.joblib # Persisted Scikit-learn Pipeline
│
└── reports/
    ├── confusion_matrix.png   # High-resolution confusion matrix heatmap
    └── metrics.json           # Detailed evaluation metrics & feature importances
```

---

## 🚀 Installation & Quick Start

### 1. Clone or Open the Project
```bash
cd Website-lead-scorer
```

### 2. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Data and Train the Model
Run the automated pipeline to generate the 1,000-record dataset, train the Random Forest model, and export evaluation metrics:
```bash
python train.py
```

### 4. (Optional) Run CLI Predictions
Test individual visitor profiles directly from your command line:
```bash
python predict.py --duration 420 --pages 7 --source Referral --downloads 3
```

### 5. Launch the Streamlit Web Application
Start the interactive SaaS dashboard:
```bash
streamlit run app.py
```

The application will launch in your browser at `http://localhost:8501`.

---

## 🎨 Dashboard Features

- **Modern SaaS Theme:** Clean slate/light-grey styling, vibrant indigo/purple accents, and responsive KPI scorecards.
- **Interactive Scoring Simulator:** Custom sliders for session duration, page views, channel dropdown, and content downloads.
- **One-Click Persona Presets:** Test archetypes instantly (*High-Intent Enterprise*, *Email Campaign Lead*, *Organic Search Skimmer*, *Casual Social Visitor*).
- **Interactive Gauge & Visual Indicators:** Visual half-donut Plotly gauge highlighting conversion probability and lead score.
- **Session History & CSV Export:** Download recent visitor scores generated during your session.
- **Exploratory Data Analysis:** Donut breakdown, traffic distribution, duration box plots, and download conversion curves.
- **Model Performance Hub:** Confusion matrix heatmap, feature importance rankings, and classification report.
- **Actionable Lead Insights:** Strategic channel ROI analysis and behavioral inflection points.

---

## 🛠️ Technology Stack

- **Frontend:** Streamlit, HTML5, CSS3
- **Data & Math:** Pandas, NumPy
- **Machine Learning:** Scikit-learn (RandomForestClassifier, ColumnTransformer, Pipeline)
- **Model Persistence:** Joblib
- **Data Visualization:** Plotly Express, Plotly Graph Objects, Seaborn, Matplotlib

---

## 📜 License & Acknowledgments

Developed as a demonstration Machine Learning and Full-Stack analytics project. Free to use for educational, research, and portfolio demonstrations.
