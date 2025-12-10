<!-- markdownlint-disable -->

# 🧠 Employee Attrition Prediction  
![status](https://img.shields.io/badge/Status-Active-brightgreen)  
![python](https://img.shields.io/badge/Python-3.10-blue)  
![sklearn](https://img.shields.io/badge/Scikit--Learn-ML%20Pipeline-orange)  
![streamlit](https://img.shields.io/badge/Streamlit-App-red)  
![license](https://img.shields.io/badge/License-MIT-purple)

Predicting Employee Turnover Using Machine Learning.

---

## 📌 Executive Summary

Employee attrition is one of the most expensive challenges organizations face.  
This project delivers a complete **end-to-end machine learning workflow** to identify employees at risk of leaving.

---

## 🚀 Key Features

- Automated scikit-learn preprocessing pipeline  
- ROC-AUC–based model selection  
- Confusion Matrix, ROC Curve, and threshold metrics  
- Streamlit batch scoring interface  
- Reproducible, modular codebase  
- HR-ready insights and outputs  

---

## 🧱 Architecture Overview


### ASCII Diagram

             ┌────────────────┐
             │   HR Dataset   │
             └───────┬────────┘
                     │
                     ▼
      ┌────────────────────────────────┐
      │     Preprocessing Pipeline     │
      │ (Scaling + OneHotEncoding etc.)│
      └───────┬────────────────────────┘
              │
              ▼
     ┌─────────────────────────────────┐
     │  Model Training (LR, RF etc.)   │
     └───────┬─────────────────────────┘
              │
              ▼
 ┌────────────────────────┐
 │   Evaluation Metrics    │
 │ ROC, F1, Precision etc. │
 └────────┬───────────────┘
          │
          ▼
 ┌────────────────────────┐
 │ Streamlit Deployment   │
 │  Batch Attrition Risk  │
 └────────────────────────┘

---

## 📂 Folder Structure

```text
employee-attrition-prediction/
├── data/
│   └── raw/
│       └── hr_data.csv
├── models/
│   └── best_attrition_model.joblib
├── notebooks/
│   └── 01_eda.ipynb
├── outputs/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── threshold_metrics.png
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── evaluate.py
│   └── app.py
├── requirements.txt
└── README.md
📊 Dataset Overview

Dataset: IBM HR Analytics Employee Attrition Dataset
Rows: ~1470
Target: Attrition (Yes/No)

Feature Categories
Category	Examples
Personal	Age, Gender, MaritalStatus
Work Environment	JobRole, Department
Performance	JobSatisfaction, JobInvolvement
Compensation	MonthlyIncome, StockOptionLevel
Behavioral	Overtime, DistanceFromHome

Imbalanced dataset: ~16% attrition.

⚙️ Machine Learning Pipeline
Preprocessing

Drop ID/non-informative columns

StandardScaler for numeric features

OneHotEncoder for categorical features

Combined using ColumnTransformer

Models Trained

Logistic Regression

Random Forest

Best model selected using ROC AUC.

Evaluation Artifacts

Confusion Matrix

ROC Curve

Threshold vs Precision/Recall/F1

📈 Results Summary
Metric	Score
Accuracy	~0.84
Precision	~0.43–0.48
Recall	~0.37–0.55
ROC AUC	~0.85

Interpretation:
Random Forest provides the strongest predictive performance. Threshold tuning helps HR prioritize retention actions.

🖥️ Streamlit Application

Run the app:

streamlit run src/app.py


Features:

Upload CSV

Predict attrition probability

Download scored file

🧪 Reproduce the Full Pipeline
Install dependencies
pip install -r requirements.txt

Train model
python src/model_training.py

Generate evaluation reports
python src/evaluate.py

Launch Streamlit
streamlit run src/app.py

🔮 Future Enhancements

Hyperparameter tuning

SHAP explainability dashboard

FastAPI microservice

Power BI retention risk dashboard

GitHub Actions CI/CD pipeline

🤝 Contributions

Pull requests welcome — create an issue before major changes.

📜 License

MIT License.