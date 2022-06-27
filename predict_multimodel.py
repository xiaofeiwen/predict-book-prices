from autogluon.multimodal import AutoMMPredictor
import pandas as pd
from utils import predict

OUTPUT_NAME = "autoMMv1.csv"
MODEL_PATH = "AutogluonModels/autoMMv1/"
FEATURE_ENGINEER = feature_engineering_v1

if __name__ == "__main__":
    df_test = pd.read_csv("data/mlu-leaderboard-test.csv")
    model = AutoMMPredictor.load(MODEL_PATH)
    predict(df_test, FEATURE_ENGINEER, model, OUTPUT_NAME)
