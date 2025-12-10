# src/app.py

"""
Streamlit app for batch employee attrition prediction.

Usage:
- Run: `streamlit run src/app.py`
- Upload a CSV with the same structure as hr_data.csv.
  - If the CSV contains Attrition, it will be ignored for prediction.
- The app will output predicted probabilities of attrition.
"""

from pathlib import Path

import pandas as pd
import streamlit as st
import joblib

from data_preprocessing import clean_and_split_features, load_raw_data, TARGET_COL


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@st.cache_resource
def load_model():
    root = get_project_root()
    model_path = root / "models" / "best_attrition_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train model first with model_training.py."
        )
    model = joblib.load(model_path)
    return model


def main():
    st.title("Employee Attrition Prediction - Batch Scoring")

    st.markdown(
        """
    **Instructions**

    1. Prepare a CSV file with the same columns as the training dataset (`hr_data.csv`).
    2. It's okay if the file still contains the `Attrition` column; it will be ignored for scoring.
    3. Upload the file below and click **Run Prediction**.
    """
    )

    model = load_model()

    uploaded_file = st.file_uploader("Upload employee data CSV", type=["csv"])

    if uploaded_file is not None:
        df_input = pd.read_csv(uploaded_file)
        st.write("Uploaded data preview:")
        st.dataframe(df_input.head())

        # If Attrition column exists, drop it for prediction
        if TARGET_COL in df_input.columns:
            df_features = df_input.drop(columns=[TARGET_COL])
        else:
            df_features = df_input.copy()

        if st.button("Run Prediction"):
            # The model pipeline includes preprocessing, so we can call predict_proba directly
            proba = model.predict_proba(df_features)[:, 1]
            df_output = df_input.copy()
            df_output["Attrition_Probability"] = proba

            st.subheader("Scored Data (sample):")
            st.dataframe(df_output.head())

            # Allow download as CSV
            root = get_project_root()
            outputs_dir = root / "outputs"
            outputs_dir.mkdir(parents=True, exist_ok=True)
            out_path = outputs_dir / "scored_employees.csv"
            df_output.to_csv(out_path, index=False)

            st.success("Predictions completed.")
            st.write(f"Scored data saved to: `{out_path.resolve()}`")
            st.download_button(
                label="Download scored CSV",
                data=df_output.to_csv(index=False),
                file_name="scored_employees.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()

    # Key takeaways:
    # - Deploy the SAME trained pipeline for scoring to avoid data leakage issues.
    # - Batch scoring (CSV-based) is often the first practical deployment step for HR.
    # - Streamlit offers a quick way to expose models to non-technical stakeholders.
