# IoT Malware Detection – Classical Baseline (Random Forest)

## 📌 Project Overview

This project is part of a Master’s thesis (PFE) focused on:

> **Malware detection in IoT systems using Machine Learning and Deep Reinforcement Learning (DRL).**

This repository contains the **classical Machine Learning baseline**, which will later be compared with a DRL-based approach.

---

## 🎯 Objectives

- Build a **clean and reproducible ML pipeline**
- Train multiple baseline models:
  - Dummy Classifier
  - Logistic Regression
  - Decision Tree
  - Random Forest
- Select the best model using validation performance
- Evaluate the final model on an **unseen test set**
- Perform **diagnostic analysis** (robustness, feature importance, calibration)
- Provide a **strong reference baseline** for DRL comparison

---

## 📊 Dataset

We use a cleaned version of the **TON_IoT network dataset**.

### Features used (after preprocessing):
- Traffic statistics:
  - `duration`, `src_bytes`, `dst_bytes`
  - `src_pkts`, `dst_pkts`
  - `src_ip_bytes`, `dst_ip_bytes`
- Protocol-related:
  - `proto`, `dns_qclass`, `dns_qtype`, `dns_rejected`
- SSL:
  - `ssl_version`, `ssl_cipher`, `ssl_resumed`
- HTTP:
  - `http_trans_depth`, `http_request_body_len`, `http_response_body_len`

### Target:
- `label` → Binary classification (benign / malicious)

### Notes:
- `src_ip` was removed (risk of leakage)
- `type` is kept only for analysis (not used in training)

---

## ⚙️ Pipeline Architecture

The pipeline follows best practices:

1. **Data Loading**
2. **Feature Preparation**
   - Drop non-useful columns
3. **Train / Validation / Test Split**
   - Stratified splitting
4. **Model Training**
   - Hyperparameter tuning using `RandomizedSearchCV`
5. **Model Selection**
   - Based on **validation F1-score**
6. **Final Training**
   - Retrain best model on train + validation
7. **Final Evaluation**
   - Test set used only once
8. **Diagnostics**
   - Feature importance
   - Ablation study
   - Robustness tests
   - Calibration analysis

---

## 🧠 Models Compared

| Model | Description |
|------|------|
| Dummy | Baseline (majority class) |
| Logistic Regression | Linear model |
| Decision Tree | Non-linear model |
| Random Forest | Ensemble model |

---

## 🏆 Best Model

The best model selected:

> **Random Forest**

### Final Performance (Test Set):

- Accuracy: **0.9954**
- Precision: **0.9971**
- Recall: **0.9968**
- F1-score: **0.9969**
- ROC AUC: **0.99985**
- PR AUC: **0.99994**

---

## 📈 Key Observations

### 1. Dataset is highly separable
- Even Decision Tree achieves very high performance
- Indicates strong patterns in the data

### 2. Strong dependence on traffic volume features
From feature ablation:

- Removing `src_pkts` or `dst_pkts` significantly reduces performance
- Indicates model relies heavily on:
  - packet counts
  - traffic volume

👉 The model is detecting **traffic anomalies**, not necessarily deep behavioral patterns

---

### 3. Robustness

- Noise injection → small performance drop
- Missing values → small performance drop

👉 Model is **stable and reliable**

---

### 4. Calibration

- Model probabilities are usable
- Slight deviation from perfect calibration

---

## ⚠️ Limitations

- Dataset is relatively **easy**
- Model relies heavily on **traffic volume features**
- Does not capture:
  - temporal behavior
  - sequential dependencies
  - complex attack patterns

---

## 🚀 Next Step: DRL Model

This baseline will be used to compare against a:

> **Deep Reinforcement Learning (DRL) agent**

Future work includes:

- Designing environment:
  - state representation
  - actions
  - reward function
- Comparing DRL vs Random Forest:
  - performance
  - generalization
  - robustness

---

## 📂 Project Structure
