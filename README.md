# Employee Attrition Prediction
[![CI](https://github.com/deepanshu0110/employee-attrition-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/deepanshu0110/employee-attrition-prediction/actions)


![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML%20Pipeline-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=flat-square&logo=streamlit)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.81-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

Identifies employees at risk of leaving using the IBM HR Analytics dataset. Includes a Streamlit app for HR teams to score employees and download risk reports.

---

## Business Problem

Replacing an employee is expensive. Identifying flight risks early lets HR act on retention before a resignation letter arrives.

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

Best model: **Logistic Regression** (ROC-AUC 0.812, vs 0.804 for Random Forest). Test set: 294 employees, 47 of whom left.

| Metric | Score |
|---|---|
| Accuracy | 86.1% |
| ROC-AUC | 0.812 |
| Precision (attrition class) | 0.62 |
| Recall (attrition class) | 0.34 |
| F1 (attrition class) | 0.44 |

**What this means:** at the default 0.5 threshold the model catches only 16 of the 47 leavers. Accuracy looks high because 84% of employees stay. Use `outputs/threshold_metrics.png` to pick a lower threshold if missing a leaver costs more than a false alarm.

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

## Tests

```bash
pip install pytest
pytest tests/ -v
```

The tests trains and evaluates the model, checks ROC-AUC, and starts the Streamlit app. GitHub Actions runs them on every push.

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