# 🏥 Healthcare Clinical Analytics & 30-Day Patient Readmission Risk Prediction with AI
### AICTE | IBM SkillsBuild Academic Internship Capstone Project

---

## 📌 Project Overview
Unplanned 30-day hospital readmissions represent a critical clinical challenge impacting patient safety, bed capacity, and hospital reimbursement penalties. This capstone project builds an end-to-end clinical data analytics and predictive modeling pipeline to analyze 3,000 inpatient admission encounters across chronic condition categories and predict 30-day readmission risk using supervised machine learning.

* **Student Name:** Peddiboyina Bhavitha
* **AICTE Student ID:** STU6a7c95e3ea4cd1786549731
* **College Enrolment No:** 25501A05J3
* **Institution:** Prasad V. Potluri Siddhartha Institute of Technology (PIN: 520007)
* **Internship Track:** Data Analytics with AI (Foundation to Implementation)
* **Host Organization:** BharatCares & CSRBOX in collaboration with IBM SkillsBuild & AICTE
* **Lead Mentor:** Mr. Kartik Hooda
* **Target Audience:** Hospital networks, clinical quality committees, healthcare operations

---

## 🚀 Key Features & Highlights
* **Independent Dataset:** 3,000 inpatient clinical encounters tracking patient demographics, admission acuity, primary diagnoses, length of hospital stay, lab testing volume, medication counts, and glycemic biomarkers (`HbA1c`).
* **Exploratory Clinical Data Analysis (EDA):**
  * 30-day readmission rate by primary diagnosis category (Diabetes/Endocrine, Cardiovascular, etc.).
  * Age cohort readmission comparison (Geriatric 65+ vs. younger cohorts).
  * Length of hospital stay (days) distribution and relapse correlation.
  * Prior emergency department visits as a predictive recurrence signal.
  * Impact of glycemic control (`HbA1c` testing) on readmission risk.
* **Predictive AI / Machine Learning:** Supervised **Logistic Regression** classifier trained on an 80/20 stratified split.
* **Performance Evaluation:**
  * **Accuracy:** 65.83%
  * **Precision:** 64.11%
  * **Recall:** 50.76%
  * **F1-Score:** 56.66%
  * **ROC-AUC Score:** 0.6858
  * **Confusion Matrix:** Heatmap visualization with true/false clinical diagnostics.
* **Clinical Risk Stratification & Discharge Protocols:**
  * 3-tier Clinical Risk System (High Risk: Intensive Post-Care, Medium Risk: Nurse Follow-Up, Low Risk: Standard Discharge).
  * Top 20 High-Risk Inpatients prioritized for transitional care intervention.
  * Discharge guidelines for medication reconciliation and outpatient transition clinics.

---

## 📂 Repository Structure
```
├── healthcare_readmission_analysis.ipynb # Executed Google Colab notebook with all charts & tables
├── patient_readmission_dataset.csv       # Raw inpatient clinical dataset (3,000 rows)
├── requirements.txt                     # Python package dependencies
├── project_report.pdf                   # Formal capstone project report (PDF export)
├── project_report.md                    # Complete formal capstone project report (Markdown)
├── project_report.html                  # Print-ready HTML report
├── SUBMISSION_METADATA.md               # Ready-to-copy fields for Google Form submission
└── README.md                            # Project documentation and summary
```

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python 3.10+
* **Data Manipulation:** `pandas`, `numpy`
* **Data Visualization:** `matplotlib`, `seaborn`
* **Machine Learning:** `scikit-learn` (`LogisticRegression`, `StandardScaler`, `train_test_split`)
* **Evaluation Metrics:** Confusion Matrix, ROC-AUC, Classification Report

---

## ⚡ How to Run
1. Open [Google Colab](https://colab.research.google.com/).
2. Click **File -> Upload notebook** and upload `healthcare_readmission_analysis.ipynb`.
3. Click **Runtime -> Run all**. All outputs and visualizations will render automatically.
