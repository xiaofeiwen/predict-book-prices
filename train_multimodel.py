from autogluon.multimodal import AutoMMPredictor
import pandas as pd
from utils import feature_engineering_v1

FEATURE_ENGINEER = feature_engineering_v1
MODEL_PATH = "AutogluonModels/autoMMv1/"

if __name__ == "__main__":
    df_train = pd.read_csv("data/training.csv")
    df_train = FEATURE_ENGINEER(df_train)
    predictor = AutoMMPredictor(
        label="Price",
        problem_type="regression",
        path=MODEL_PATH,
        enable_progress_bar=True,
        verbosity=2,
    ).fit(df_train, presets="best_quality", time_limit=3600 * 10, seed=42,)
