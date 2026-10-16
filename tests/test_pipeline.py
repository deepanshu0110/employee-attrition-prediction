import re

from tests.helpers import ROOT, run_script


def test_training_and_evaluation_run():
    run_script("src/model_training.py")
    output = run_script("src/evaluate.py")
    auc = float(re.search(r"ROC AUC\s*:\s*([0-9.]+)", output).group(1))
    assert 0.75 < auc < 0.9
    assert (ROOT / "models" / "best_attrition_model.joblib").exists()


def test_streamlit_app_starts():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(ROOT / "src" / "app.py"), default_timeout=120).run()
    assert not app.exception
