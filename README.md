# Employee Attrition Prediction

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML%20Pipeline-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=flat-square&logo=streamlit)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-~0.85-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

Identifies employees at risk of leaving using the IBM HR Analytics dataset. Includes a Streamlit app for HR teams to score employees and download risk reports.

---

## Business Problem

Replacing an employee costs 50–200% of their annual salary. Identifying flight risks early lets HR act on retention — before a resignation letter arrives.

---

## Dataset

| Property | Value |
|---|---|
| Source | IBM HR Analytics (Kaggle) |
| Rows | ~1,470 employees |
| Target | Attrition (Yes / No) |
| Class imbalance | ~16% attrition |

---

## Results

| Metric | Score |
|---|---|
| Accuracy | ~84% |
| ROC-AUC | ~0.85 |
| Precision | ~0.43–0.48 |
| Recall | ~0.37–0.55 |

---

## ML Pipeline

1. Drop non-informative columns
2. StandardScaler for numeric features
3. OneHotEncoder for categoricals via ColumnTransformer
4. Train Logistic Regression + Random Forest
5. Select best by ROC-AUC
6. Export confusion matrix, ROC curve, threshold metrics

---

## Quickstart

```bash
git clone https://github.com/deepanshu0110/employee-attrition-prediction.git
cd employee-attrition-prediction
pip install -r requirements.txt
python src/model_training.py
python src/evaluate.py
streamlit run src/app.py
```

---

## Tech Stack

Python · Pandas · Scikit-learn · Joblib · Streamlit · Matplotlib · Seaborn

---

## Roadmap

- [ ] SHAP explainability dashboard
- [ ] FastAPI microservice
- [ ] Hyperparameter tuning with Optuna

---

## Author

**Deepanshu Garg** — Freelance Data Scientist
- GitHub: [@deepanshu0110](https://github.com/deepanshu0110)
- Hire: [freelancer.com/u/deepanshu0110](https://www.freelancer.com/u/deepanshu0110)

MIT License