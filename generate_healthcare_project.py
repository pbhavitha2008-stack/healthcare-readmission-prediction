import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Build the self-contained, turnkey Healthcare Jupyter Notebook
notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# AICTE | IBM SkillsBuild Academic Internship\n",
    "## Capstone Project: Healthcare Clinical Analytics & 30-Day Patient Readmission Risk Prediction with AI\n",
    "---\n",
    "**Internship Track:** Data Analytics with AI  \n",
    "**Host Organization:** BharatCares & CSRBOX in collaboration with IBM SkillsBuild & AICTE  \n",
    "**Lead Mentor:** Mr. Kartik Hooda  \n",
    "**Date:** September 2026  \n",
    "---\n",
    "### Abstract:\n",
    "Unplanned hospital readmissions within 30 days of discharge represent a critical indicator of clinical quality, patient safety, and healthcare expenditure. Inpatient readmissions impose substantial financial penalties on hospital networks and strain healthcare capacity. This project builds an end-to-end clinical data analytics and predictive modeling pipeline that:\n",
    "1. Ingests and cleans an inpatient clinical dataset of 3,000 patient admission records across chronic conditions.\n",
    "2. Conducts Exploratory Data Analysis (EDA) investigating the clinical correlation between length of stay, laboratory testing intensity, glycemic control, and readmission.\n",
    "3. Formulates a supervised machine learning model using **Logistic Regression** to stratify patients by readmission probability.\n",
    "4. Evaluates model performance using Accuracy, Precision, Recall, F1-Score, ROC-AUC, and a Confusion Matrix.\n",
    "5. Categorizes discharged patients into actionable **Clinical Risk Tiers** (High, Medium, Low Risk) to guide post-discharge transitional care and nursing follow-ups."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["### 1. Environment Setup & Library Imports"]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import (\n",
    "    accuracy_score, precision_score, recall_score, f1_score,\n",
    "    confusion_matrix, classification_report, roc_curve, roc_auc_score\n",
    ")\n",
    "\n",
    "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n",
    "sns.set_palette(\"Blues_r\")\n",
    "plt.rcParams['font.size'] = 11\n",
    "print(\"Clinical Analytics libraries imported successfully!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2. Dataset Generation & Ingestion (Turnkey Self-Contained)\n",
    "*Note: In adherence to the guidelines set by AICTE/IBM mentors, this project uses an independent clinical dataset of 3,000 patient encounters generated directly in memory for 100% reproducible and turnkey execution in Google Colab.*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Set seed for absolute reproducibility\n",
    "np.random.seed(42)\n",
    "n_patients = 3000\n",
    "\n",
    "patient_ids = [f\"PID-{i:05d}\" for i in range(10001, 10001 + n_patients)]\n",
    "age_groups = np.random.choice(['18-30', '31-50', '51-65', '65+'], size=n_patients, p=[0.10, 0.25, 0.35, 0.30])\n",
    "genders = np.random.choice(['Male', 'Female'], size=n_patients, p=[0.49, 0.51])\n",
    "admission_types = np.random.choice(['Emergency', 'Urgent', 'Elective'], size=n_patients, p=[0.55, 0.25, 0.20])\n",
    "\n",
    "diagnoses = ['Cardiovascular', 'Diabetes/Endocrine', 'Respiratory', 'Digestive', 'Musculoskeletal']\n",
    "diagnosis_probs = [0.32, 0.26, 0.18, 0.14, 0.10]\n",
    "diagnosis_category = np.random.choice(diagnoses, size=n_patients, p=diagnosis_probs)\n",
    "\n",
    "# Time in hospital: 1 to 14 days\n",
    "time_in_hospital = np.random.choice(range(1, 15), size=n_patients, p=[0.15, 0.20, 0.18, 0.14, 0.10, 0.07, 0.05, 0.03, 0.03, 0.02, 0.01, 0.01, 0.005, 0.005])\n",
    "\n",
    "num_lab_procedures = np.random.randint(10, 95, size=n_patients)\n",
    "num_medications = np.clip(np.random.poisson(lam=12, size=n_patients) + (time_in_hospital // 2), 1, 35)\n",
    "\n",
    "num_outpatient = np.random.choice(range(0, 7), size=n_patients, p=[0.60, 0.20, 0.10, 0.05, 0.03, 0.01, 0.01])\n",
    "num_emergency = np.random.choice(range(0, 5), size=n_patients, p=[0.72, 0.16, 0.07, 0.03, 0.02])\n",
    "\n",
    "hba1c_test = np.random.choice(['Normal', 'High', 'None'], size=n_patients, p=[0.20, 0.18, 0.62])\n",
    "glucose_test = np.random.choice(['Normal', 'High', 'None'], size=n_patients, p=[0.15, 0.12, 0.73])\n",
    "insulin_prescribed = np.where(diagnosis_category == 'Diabetes/Endocrine', np.random.choice(['Yes', 'No'], size=n_patients, p=[0.65, 0.35]), np.random.choice(['Yes', 'No'], size=n_patients, p=[0.15, 0.85]))\n",
    "\n",
    "risk_logits = (\n",
    "    -2.1\n",
    "    + 0.8 * (age_groups == '65+')\n",
    "    + 0.6 * (admission_types == 'Emergency')\n",
    "    + 0.5 * (diagnosis_category == 'Diabetes/Endocrine')\n",
    "    + 0.4 * (diagnosis_category == 'Cardiovascular')\n",
    "    + 0.5 * num_emergency\n",
    "    + 0.6 * (hba1c_test == 'High')\n",
    "    + 0.06 * time_in_hospital\n",
    "    + 0.03 * num_medications\n",
    ")\n",
    "readmit_prob = 1 / (1 + np.exp(-risk_logits))\n",
    "readmitted = (np.random.rand(n_patients) < readmit_prob).astype(int)\n",
    "\n",
    "df = pd.DataFrame({\n",
    "    'Patient_ID': patient_ids,\n",
    "    'Age_Group': age_groups,\n",
    "    'Gender': genders,\n",
    "    'Admission_Type': admission_types,\n",
    "    'Diagnosis_Category': diagnosis_category,\n",
    "    'Time_in_Hospital': time_in_hospital,\n",
    "    'Num_Lab_Procedures': num_lab_procedures,\n",
    "    'Num_Medications': num_medications,\n",
    "    'Num_Outpatient_Visits': num_outpatient,\n",
    "    'Num_Emergency_Visits': num_emergency,\n",
    "    'HbA1c_Test_Result': hba1c_test,\n",
    "    'Glucose_Test_Result': glucose_test,\n",
    "    'Insulin_Prescribed': insulin_prescribed,\n",
    "    'Readmitted_30Days': readmitted\n",
    "})\n",
    "\n",
    "# Save local copy in Colab session\n",
    "df.to_csv('patient_readmission_dataset.csv', index=False)\n",
    "\n",
    "print(f\"Inpatient Records: {df.shape[0]}, Clinical Features: {df.shape[1]}\")\n",
    "display(df.head())\n",
    "print(\"\\n--- Clinical Data Types & Missing Value Audit ---\")\n",
    "print(df.info())\n",
    "print(\"\\nMissing Values Check:\")\n",
    "print(df.isnull().sum())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["### 3. Exploratory Data Analysis (EDA) & Visualizations"]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["#### Visualization 1: Readmission Rate by Diagnosis Category"]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "diag_readmit = df.groupby('Diagnosis_Category')['Readmitted_30Days'].mean().reset_index()\n",
    "diag_readmit['Readmission_Rate_%'] = diag_readmit['Readmitted_30Days'] * 100\n",
    "diag_readmit = diag_readmit.sort_values(by='Readmission_Rate_%', ascending=False)\n",
    "\n",
    "plt.figure(figsize=(9, 5))\n",
    "sns.barplot(data=diag_readmit, x='Readmission_Rate_%', y='Diagnosis_Category', palette='magma')\n",
    "plt.title('30-Day Readmission Rate by Primary Diagnosis Category (%)', fontsize=13, fontweight='bold')\n",
    "plt.xlabel('Readmission Rate (%)', fontweight='bold')\n",
    "plt.ylabel('Primary Diagnosis', fontweight='bold')\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "display(diag_readmit)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["#### Visualization 2: Readmission Rate by Age Group"]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "age_readmit = df.groupby('Age_Group')['Readmitted_30Days'].mean().reset_index()\n",
    "age_readmit['Rate_%'] = age_readmit['Readmitted_30Days'] * 100\n",
    "\n",
    "plt.figure(figsize=(8, 5))\n",
    "bars = plt.bar(age_readmit['Age_Group'], age_readmit['Rate_%'], color='#e6550d', width=0.5)\n",
    "plt.title('Patient Readmission Rate Across Age Cohorts (%)', fontsize=13, fontweight='bold')\n",
    "plt.xlabel('Age Cohort', fontweight='bold')\n",
    "plt.ylabel('Readmission Rate (%)', fontweight='bold')\n",
    "for bar in bars:\n",
    "    yval = bar.get_height()\n",
    "    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', fontweight='bold')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["#### Visualization 3: Length of Hospital Stay vs. Readmission"]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(8, 5))\n",
    "sns.boxplot(data=df, x='Readmitted_30Days', y='Time_in_Hospital', palette=['#3182bd', '#de2d26'])\n",
    "plt.title('Length of Hospital Stay (Days) by Readmission Status', fontsize=13, fontweight='bold')\n",
    "plt.xticks([0, 1], ['Not Readmitted (0)', 'Readmitted within 30 Days (1)'])\n",
    "plt.xlabel('Readmission Status', fontweight='bold')\n",
    "plt.ylabel('Days in Hospital', fontweight='bold')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["#### Visualization 4: Impact of Prior Emergency Department Visits"]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "er_readmit = df.groupby('Num_Emergency_Visits')['Readmitted_30Days'].mean().reset_index()\n",
    "er_readmit['Rate_%'] = er_readmit['Readmitted_30Days'] * 100\n",
    "\n",
    "plt.figure(figsize=(8, 5))\n",
    "plt.plot(er_readmit['Num_Emergency_Visits'], er_readmit['Rate_%'], marker='o', color='#756bb1', linewidth=2.5, markersize=8)\n",
    "plt.title('Readmission Probability vs. Prior Emergency Room Visits', fontsize=13, fontweight='bold')\n",
    "plt.xlabel('Number of Prior Emergency Visits in Past Year', fontweight='bold')\n",
    "plt.ylabel('Readmission Rate (%)', fontweight='bold')\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "display(er_readmit)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["#### Visualization 5: Glycemic Control (HbA1c Test) vs. Readmission Rate"]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "hba1c_readmit = df.groupby('HbA1c_Test_Result')['Readmitted_30Days'].mean().reset_index()\n",
    "hba1c_readmit['Rate_%'] = hba1c_readmit['Readmitted_30Days'] * 100\n",
    "\n",
    "plt.figure(figsize=(7, 5))\n",
    "sns.barplot(data=hba1c_readmit, x='HbA1c_Test_Result', y='Rate_%', palette='viridis')\n",
    "plt.title('30-Day Readmission by Glycemic (HbA1c) Test Outcome', fontsize=13, fontweight='bold')\n",
    "plt.xlabel('HbA1c Test Result', fontweight='bold')\n",
    "plt.ylabel('Readmission Rate (%)', fontweight='bold')\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "display(hba1c_readmit)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["### 4. Clinical Feature Engineering & Data Preparation"]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "feature_cols = ['Time_in_Hospital', 'Num_Lab_Procedures', 'Num_Medications', 'Num_Outpatient_Visits', 'Num_Emergency_Visits']\n",
    "df_encoded = pd.get_dummies(df[['Age_Group', 'Admission_Type', 'Diagnosis_Category', 'HbA1c_Test_Result', 'Insulin_Prescribed']], drop_first=True)\n",
    "X = pd.concat([df[feature_cols], df_encoded], axis=1)\n",
    "y = df['Readmitted_30Days']\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)\n",
    "scaler = StandardScaler()\n",
    "X_train_scaled = scaler.fit_transform(X_train)\n",
    "X_test_scaled = scaler.transform(X_test)\n",
    "\n",
    "print(f\"Training set: {X_train.shape}, Test set: {X_test.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["### 5. Machine Learning Model Training (Logistic Regression)"]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = LogisticRegression(max_iter=1000, random_state=42)\n",
    "model.fit(X_train_scaled, y_train)\n",
    "\n",
    "y_pred = model.predict(X_test_scaled)\n",
    "y_prob = model.predict_proba(X_test_scaled)[:, 1]\n",
    "print(\"Clinical Readmission Model trained successfully!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["### 6. Model Evaluation & Performance Metrics"]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"=\"*45)\n",
    "print(\"      HEALTHCARE READMISSION METRICS REPORT    \")\n",
    "print(\"=\"*45)\n",
    "print(f\"Accuracy : {accuracy_score(y_test, y_pred)*100:.2f}%\")\n",
    "print(f\"Precision: {precision_score(y_test, y_pred)*100:.2f}%\")\n",
    "print(f\"Recall   : {recall_score(y_test, y_pred)*100:.2f}%\")\n",
    "print(f\"F1-Score : {f1_score(y_test, y_pred)*100:.2f}%\")\n",
    "print(f\"ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}\")\n",
    "print(\"=\"*45)\n",
    "\n",
    "cm = confusion_matrix(y_test, y_pred)\n",
    "plt.figure(figsize=(6, 5))\n",
    "sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',\n",
    "            xticklabels=['Predicted Safe (0)', 'Predicted Readmit (1)'],\n",
    "            yticklabels=['Actual Safe (0)', 'Actual Readmit (1)'])\n",
    "plt.title('Clinical Confusion Matrix', fontweight='bold')\n",
    "plt.ylabel('Actual Readmission Status', fontweight='bold')\n",
    "plt.xlabel('Predicted Readmission Status', fontweight='bold')\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "print(\"\\n--- Classification Report ---\")\n",
    "print(classification_report(y_test, y_pred, target_names=['Safe (0)', 'Readmitted (1)']))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["### 7. Patient Risk Stratification & Discharge Protocols"]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_all_scaled = scaler.transform(X)\n",
    "df['Readmit_Probability_%'] = np.round(model.predict_proba(X_all_scaled)[:, 1] * 100, 2)\n",
    "\n",
    "def assign_clinical_risk(prob):\n",
    "    if prob >= 70.0: return 'High Risk (Intensive Post-Care)'\n",
    "    elif prob >= 40.0: return 'Medium Risk (Nurse Follow-Up)'\n",
    "    else: return 'Low Risk (Standard Discharge)'\n",
    "\n",
    "df['Clinical_Risk_Tier'] = df['Readmit_Probability_%'].apply(assign_clinical_risk)\n",
    "print(\"--- Inpatient Clinical Risk Stratification ---\")\n",
    "display(df['Clinical_Risk_Tier'].value_counts())\n",
    "\n",
    "top_risk_patients = df.sort_values(by='Readmit_Probability_%', ascending=False).head(20)\n",
    "print(\"\\n--- Top 20 High-Risk Inpatients Requiring Post-Discharge Protocols ---\")\n",
    "display(top_risk_patients[['Patient_ID', 'Age_Group', 'Diagnosis_Category', 'Time_in_Hospital', 'Num_Emergency_Visits', 'HbA1c_Test_Result', 'Readmit_Probability_%', 'Clinical_Risk_Tier']])\n",
    "\n",
    "df.to_csv('patient_readmission_risk_scores.csv', index=False)\n",
    "print(\"Exported clinical risk scores to 'patient_readmission_risk_scores.csv'.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 8. Clinical Recommendations & Quality Improvement Decisions\n",
    "1. **Transitional Care for High-Risk Cohort:** Patients with $\\ge 70\\%$ readmission risk must receive a 48-hour post-discharge nurse home visit and automated medication reconciliation.\n",
    "2. **Standardized Glycemic Protocols:** Patients with High HbA1c exhibit elevated readmission rates. Mandate in-hospital diabetes education and a 7-day post-discharge endocrine clinic appointment.\n",
    "3. **Emergency Recidivism Management:** Patients with $>2$ emergency visits in the preceding year should be assigned a dedicated medical social worker to resolve outpatient transport and medication adherence barriers."
   ]
  }
 ],
 "metadata": {"language_info": {"name": "python"}},
 "nbformat": 4,
 "nbformat_minor": 2
}

nb_path = r"C:\Users\raahe\.gemini\antigravity\scratch\healthcare_readmission_project\healthcare_readmission_analysis.ipynb"
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)
print(f"Updated Healthcare Notebook generated at: {nb_path}")
