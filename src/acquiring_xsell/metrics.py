"""
Модуль для расчёта метрик качества моделей.

Все функции принимают на вход `df_eval` и возвращают pd.Series, где индекс - это 'month', а значения - метрика для этого месяца.

Так сделано, чтобы можно было легко агрегировать метрики по месяцам (например, брать среднее) и отслеживать динамику метрик во времени.
"""

import pandas as pd
import numpy as np
from typing import List
from sklearn.metrics import average_precision_score

K_VALUES = [500, 1000, 1500, 2000]

METRICS_STYLE = {
    "precision_k": "{:.2%}",
    "recall_k": "{:.2%}",
    "expected_conversions_k": "{:.2f}",
}


# ============================================================ Average Precision ============================================================ #


def monthly_average_precision_score(
    df_eval: pd.DataFrame,
    y_true_col: str,
    y_score_col: str,
    k: int = None,
) -> pd.Series:
    """
    Вычисляет Average Precision для каждого месяца `month`.
    Если k задан, вычисляет AP@K.
    """

    monthly_ap = {}
    for month, grp in df_eval.groupby("month"):
        k_eff = len(grp) if k is None else k

        top_k = grp.sort_values(by=y_score_col, ascending=False).head(k_eff)

        if top_k[y_true_col].nunique() < 2:
            monthly_ap[month] = np.nan

        else:
            monthly_ap[month] = average_precision_score(
                top_k[y_true_col], top_k[y_score_col]
            )

    return pd.Series(monthly_ap).sort_index()


# ============================================================== Top-K metrics ============================================================== #


def monthly_precision_k_score(
    df_eval: pd.DataFrame,
    y_true_col: str,
    y_score_col: str,
    k: int,
    ascending: bool = False,
) -> pd.Series:

    monthly_precision_k = {}
    for month, grp in df_eval.groupby("month"):
        top_k = grp.sort_values(by=y_score_col, ascending=ascending).head(k)

        precision_k = top_k[y_true_col].mean()

        monthly_precision_k[month] = precision_k

    return pd.Series(monthly_precision_k).sort_index()


def monthly_recall_k_score(
    df_eval: pd.DataFrame,
    y_true_col: str,
    y_score_col: str,
    k: int,
    ascending: bool = False,
) -> pd.Series:

    monthly_recall_k = {}
    for month, grp in df_eval.groupby("month"):
        top_k = grp.sort_values(by=y_score_col, ascending=ascending).head(k)

        tp = top_k[y_true_col].sum()
        total_pos = grp[y_true_col].sum()

        recall_k = tp / total_pos if total_pos > 0 else 0

        monthly_recall_k[month] = recall_k

    return pd.Series(monthly_recall_k).sort_index()


def monthly_expected_conv_k_score(
    df_eval: pd.DataFrame,
    y_true_col: str,
    y_score_col: str,
    k: int,
    ascending: bool = False,
) -> pd.Series:

    monthly_expected_conv_k = {}
    for month, grp in df_eval.groupby("month"):
        top_k = grp.sort_values(by=y_score_col, ascending=ascending).head(k)

        expected_conversions_k = top_k[y_true_col].sum()

        monthly_expected_conv_k[month] = expected_conversions_k

    return pd.Series(monthly_expected_conv_k).sort_index()


# ============================================================== Top-K AGG ============================================================== #


def all_top_k_scores(
    df_eval: pd.DataFrame,
    y_true_col: str,
    y_score_col: str,
    k_values: List[int] = K_VALUES,
    ascending: bool = False,
) -> pd.DataFrame:

    df = pd.DataFrame(
        columns=["precision_k", "recall_k", "expected_conversions_k"],
        index=k_values,
        dtype=float,
    )

    for k in k_values:
        df.loc[k] = {
            "precision_k": monthly_precision_k_score(
                df_eval, y_true_col, y_score_col, k, ascending
            ).mean(),
            "recall_k": monthly_recall_k_score(
                df_eval, y_true_col, y_score_col, k, ascending
            ).mean(),
            "expected_conversions_k": monthly_expected_conv_k_score(
                df_eval, y_true_col, y_score_col, k, ascending
            ).mean(),
        }

    return df.sort_index()


def format_metrics(df):
    return df.style.format(METRICS_STYLE)