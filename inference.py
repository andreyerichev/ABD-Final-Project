import joblib
import pandas as pd
import numpy as np

from acquiring_xsell.data import load_data
from acquiring_xsell.preprocessing import process_dtypes, fill_missing_values
from acquiring_xsell.features import build_features, make_product_df

class BusinessRuleModel:
    def __init__(self):
        self.score_col = "turnover"
        self.flag_col = "rule_flag_p3"
        self.min_ = None
        self.max_ = None

    def fit(self, X, y=None):
        scores = X[self.score_col].values
        self.min_ = scores.min()
        self.max_ = scores.max()
        return self

    def predict_score(self, X):
        scores = X[self.score_col].values
        flags = X[self.flag_col].values

        norm_scores = (scores - self.min_) / (self.max_ - self.min_ + 1e-9)
        norm_scores = np.where(flags == 1, 0.0, norm_scores)

        return norm_scores

    def predict_proba(self, X):
        s = self.predict_score(X)
        return np.vstack([1 - s, s]).T


PRODUCTS = ["p1p2", "p3", "altpay5"]

PRODUCT_VALUES = {
    "p1p2": 40_000,
    "p3": 15_000,
    "altpay5": 25_000
}


MODEL_PATHS = {
    "p1p2": "models/p1p2.joblib",
    "p3": "models/p3.joblib",
    "altpay5": "models/altpay5.joblib",
}

CALIBRATORS = {
    "p1p2": "models/p1p2_platt.joblib",
    "p3": "models/p3_platt.joblib",
    "altpay5": "models/altpay5_platt.joblib"
}


models = {k: joblib.load(v) for k, v in MODEL_PATHS.items()}
calibrators = {k: joblib.load(v) for k, v in CALIBRATORS.items()}


def get_scores(product: str, df: pd.DataFrame) -> np.ndarray:
    model = models[product]

    if product == "p3":
        X = df
        return model.predict_score(X)

    X = df[model.feature_names_in_]
    return model.predict_proba(X)[:, 1]


def get_proba(product: str, scores: np.ndarray) -> np.ndarray:
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)

    calibrator = calibrators[product]
    return calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]


def compute_expected_value(product: str, proba: np.ndarray) -> np.ndarray:
    return proba * PRODUCT_VALUES[product]


def main():

    df = load_data()
    df = process_dtypes(df)
    df = fill_missing_values(df)
    df = build_features(df, drop_dttm_columns=False)

    max_month = df["month"].max()
    df = df[df["month"] == max_month]

    

    results = []

    for product in PRODUCTS:

        df_p = make_product_df(df, product)

        scores = get_scores(product, df_p)

        proba = get_proba(product, scores)

        ev = compute_expected_value(product, proba)
        ev = np.nan_to_num(ev, nan=0.0)

        out = pd.DataFrame({
            "inn": df_p["inn"].values,
            "inn_status": df_p["inn_status"],
            "first_trx_month_inn": df_p["first_trx_month_inn"],
            "current_segment_inn": df_p["current_segment_inn"],
            "top_mcc_group_inn": df_p["top_mcc_group_inn"],
            "who_is_this_first_inn": df_p["who_is_this_first_inn"],
            "real_kam_on_inn": df_p["real_kam_on_inn"],
            "turnover_ma_3m": df_p["turnover_ma_3m"],
            "cnt_trx_ma_3m": df_p["cnt_trx_ma_3m"],
            "avg_check_wma_3m": df_p["avg_check_wma_3m"],
            "relevant_product": product,
            "expected_value": ev
        })

        results.append(out)

    df_all = pd.concat(results, ignore_index=True)

    df_all = df_all.sort_values("expected_value", ascending=False)

    df_all.to_csv("outputs/predictions.csv", index=False)


if __name__ == "__main__":
    main()