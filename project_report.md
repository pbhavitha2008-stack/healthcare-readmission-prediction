# AICTE | IBM SkillsBuild Academic Internship
## Capstone Project Report: Healthcare Inpatient Clinical Analytics & 30-Day Patient Readmission Risk Prediction with AI

---

### Project & Candidate Information
* **Student Name:** Peddiboyina Bhavitha
* **AICTE Student ID:** STU6a7c95e3ea4cd1786549731
* **College Enrolment / Roll No:** 25501A05J3
* **Institution / College:** Prasad V. Potluri Siddhartha Institute of Technology (PIN: 520007)
* **Internship Program:** AICTE | IBM SkillsBuild Academic Internship – Data Analytics with AI
* **Organization:** BharatCares & CSRBOX in collaboration with IBM SkillsBuild & AICTE
* **Lead Mentor / Industry Guide:** Mr. Kartik Hooda
* **Project Track:** Data Analytics with AI (Foundation to Implementation)
* **Submission Date:** 21 September 2026

---

## 1. Executive Summary

Unplanned 30-day hospital readmissions represent one of the most pressing challenges facing modern healthcare systems worldwide. Frequent readmissions indicate potential gaps in post-discharge transitional care, lower patient recovery quality, and impose severe financial penalties under hospital quality incentive programs.

This capstone project develops an end-to-end clinical data analytics and machine learning solution for a regional hospital network, evaluating **3,000 inpatient admission records** across chronic disease categories (Cardiovascular, Endocrine/Diabetes, Respiratory, Digestive, and Musculoskeletal):
1. Cleans and audits multi-dimensional clinical data, tracking length of stay, laboratory test intensity, prior acute care utilization, and glycemic control biomarkers (`HbA1c`).
2. Conducts **Exploratory Data Analysis (EDA)** to establish evidence-based correlations between clinical variables and readmission probability.
3. Formulates a supervised **Logistic Regression classification model** delivering **65.83% Accuracy**, **64.11% Precision**, **50.76% Recall**, and an **ROC-AUC of 0.6858**.
4. Implements a 3-tier **Clinical Risk Stratification Framework** (High Risk: Intensive Post-Care, Medium Risk: Nurse Follow-Up, Low Risk: Standard Discharge) to optimize clinical staffing and prevent acute care recidivism.

---

## 2. Business & Clinical Problem Statement

### 2.1 The Clinical Challenge
Hospitals frequently discharge complex chronic disease patients who experience medical relapses within 30 days. When clinical teams lack predictive foresight, all discharged patients receive the same generic instructions, resulting in preventable complications, overcrowded emergency departments, and avoidable bed occupancy.

### 2.2 Core Project Objectives
* **Clinical Pattern Recognition:** Identify which disease categories, age demographics, and inpatient length-of-stay durations associate with the highest relapse risk.
* **Risk Stratification via AI:** Build a predictive machine learning model to assign an objective readmission risk score to patients prior to discharge.
* **Transitional Care Optimization:** Enable hospital care coordinators to allocate high-touch follow-up resources (home visits, telehealth check-ins, medication reconciliation) to the patients who need them most.

---

## 3. Dataset Architecture & Clinical Variables

The project utilizes an independent clinical dataset comprising **3,000 unique inpatient encounters**:

### 3.1 Data Dictionary
| Clinical Attribute | Data Type | Description |
| :--- | :--- | :--- |
| `Patient_ID` | String | Unique patient identifier (`PID-10001` to `PID-13000`) |
| `Age_Group` | Categorical | Age bracket: `18-30`, `31-50`, `51-65`, `65+` |
| `Gender` | Categorical | `Male`, `Female` |
| `Admission_Type` | Categorical | Acuity at admission: `Emergency`, `Urgent`, `Elective` |
| `Diagnosis_Category` | Categorical | Primary diagnosis: `Cardiovascular`, `Diabetes/Endocrine`, `Respiratory`, `Digestive`, `Musculoskeletal` |
| `Time_in_Hospital` | Integer | Total inpatient length of stay in days (1 to 14) |
| `Num_Lab_Procedures` | Integer | Count of diagnostic lab tests conducted during stay (10 to 95) |
| `Num_Medications` | Integer | Count of unique pharmaceuticals administered (1 to 35) |
| `Num_Outpatient_Visits`| Integer | Outpatient clinic encounters in the preceding 12 months (0 to 6) |
| `Num_Emergency_Visits` | Integer | Emergency department presentations in the preceding 12 months (0 to 4) |
| `HbA1c_Test_Result` | Categorical | Glycemic biomarker: `Normal`, `High`, `None` |
| `Glucose_Test_Result`| Categorical | Serum glucose test: `Normal`, `High`, `None` |
| `Insulin_Prescribed` | Binary | Whether insulin therapy was initiated or adjusted (`Yes`/`No`) |
| `Readmitted_30Days` | Binary Target | 1 = Readmitted within 30 days, 0 = No readmission |

### 3.2 Dataset Distribution
* **Total Encounters:** 3,000
* **Non-Readmitted Patients (0):** 1,682 (56.1%)
* **Readmitted Patients (1):** 1,318 (43.9%)
* **Data Hygiene:** Zero missing values across all clinical variables.

---

## 4. Exploratory Data Analysis & Clinical Insights

### Visualization 1: Readmission Rate by Primary Diagnosis Category
* **Observation:** **Diabetes / Endocrine** conditions show the highest readmission rate (**52.4%**), followed closely by **Cardiovascular diseases (48.6%)**. Musculoskeletal disorders show the lowest readmission rate (28.3%).
* **Clinical Insight:** Metabolic and cardiac disorders require complex chronic disease management and polypharmacy, increasing vulnerability to post-discharge instability.
* **Recommendation:** Establish specialized outpatient diabetic and cardiac transition clinics to evaluate patients within 7 days of discharge.

---

### Visualization 2: Readmission Rate Across Age Cohorts
* **Observation:** Patients aged **65+** have a readmission rate of **54.2%**, nearly double the rate of the 18–30 age bracket (27.8%).
* **Clinical Insight:** Geriatric patients frequently suffer from multi-morbidity, cognitive decline, and reduced social support, amplifying post-discharge relapse risk.
* **Recommendation:** Deploy dedicated geriatric nurse navigators to coordinate home healthcare and caregiver training for all discharged patients aged 65+.

---

### Visualization 3: Inpatient Length of Stay vs. Readmission
* **Observation:** Patients readmitted within 30 days had a higher median hospital stay (**5.4 days**) compared to non-readmitted patients (**3.8 days**).
* **Clinical Insight:** Longer hospital stays are a proxy for higher underlying clinical acuity, nosocomial infection risks, and slower physiological recovery.
* **Recommendation:** Flag any patient hospitalized for $\ge 6$ days for a multidisciplinary discharge audit before authorizing release.

---

### Visualization 4: Prior Emergency Department Utilization
* **Observation:** Readmission rates escalate linearly with prior emergency room visits: patients with **0 prior ER visits had a 34.1% readmission rate**, whereas patients with **3+ visits had a 71.8% readmission rate**.
* **Clinical Insight:** Frequent ER visits reflect chronic disease instability, poor primary care access, or financial barriers to outpatient medication adherence.
* **Recommendation:** Assign medical social workers to frequent ER visitors to address socioeconomic barriers, transportation, and prescription affordability.

---

### Visualization 5: Glycemic Control (HbA1c Test) vs. Readmission
* **Observation:** Patients with an **elevated HbA1c result** experienced a **58.9% readmission rate**, compared to **36.2% for normal HbA1c** and 41.5% where no test was ordered.
* **Clinical Insight:** Uncontrolled hyperglycemia impairs immune response, slows wound healing, and destabilizes vascular function.
* **Recommendation:** Mandate bedside HbA1c screening for all adult inpatients with diabetes and implement automated endocrinology consult triggers for high readings.

---

## 5. Machine Learning Model Training & Evaluation

### 5.1 Preprocessing Pipeline
* **Encoding:** Categorical predictors (`Age_Group`, `Admission_Type`, `Diagnosis_Category`, `HbA1c_Test_Result`, `Insulin_Prescribed`) were one-hot encoded with reference levels dropped.
* **Stratified Partition:** 80% Training Set (2,400 encounters) and 20% Holdout Test Set (600 encounters).
* **Feature Scaling:** Continuous clinical metrics (`Time_in_Hospital`, `Num_Lab_Procedures`, `Num_Medications`, `Num_Outpatient_Visits`, `Num_Emergency_Visits`) were standardized with `StandardScaler` fitted strictly on training data.

### 5.2 Model Performance Metrics (Test Set)
* **Accuracy:** **65.83%**
* **Precision:** **64.11%**
* **Recall:** **50.76%**
* **F1-Score:** **56.66%**
* **ROC-AUC Score:** **0.6858**

### 5.3 Clinical Confusion Matrix Heatmap Breakdown
```
                   PREDICTED SAFE (0)   PREDICTED READMIT (1)
ACTUAL SAFE (0)           261                    75
ACTUAL READMIT (1)        130                   134
```
* **True Negatives (261):** Low-risk patients correctly identified for standard discharge.
* **False Positives (75):** Patients flagged for extra transitional care who did not actually readmit (precautionary care).
* **True Positives (134):** High-risk patients successfully identified for intensive post-discharge interventions.

---

## 6. Clinical Risk Stratification & Discharge Protocols

Based on individual predicted readmission probabilities, inpatients are stratified into three actionable clinical tiers:
1. **High Risk ($\ge 70\%$ Readmission Probability):**
   * *Protocol:* 48-hour post-discharge home health visit, pharmacist-led medication reconciliation, and mandatory 7-day clinic follow-up.
2. **Medium Risk ($40\% - 69\%$ Probability):**
   * *Protocol:* Automated telephone check-ins at Day 3 and Day 10 post-discharge, with symptom check surveys.
3. **Low Risk ($< 40\%$ Probability):**
   * *Protocol:* Standard discharge summary, outpatient primary care appointment within 14–21 days.

---

## 7. Conclusion & Quality Improvement Impact
By deploying this predictive healthcare analytics framework, hospital administrators can shift from retrospective readmission reporting to prospective clinical intervention. Targeting high-risk chronic disease patients significantly improves patient outcomes, reduces hospital bed overcrowding, and protects hospital reimbursement under national healthcare quality standards.
