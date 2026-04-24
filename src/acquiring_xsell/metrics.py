"""
Модуль для расчёта метрик качества моделей.
"""

import pandas as pd
from typing import List


def calculate_metrics(
    data: pd.DataFrame,
    y_score: str,
    k_values: List[int] = [500, 1000, 1500, 2000, 2500],
    business_rule_mode: bool = False,
    ascending: bool = False,
) -> None:
    """
    Выводит print() с рассчитанными Precision@K, Recall@K и Expected Conversions@K для заданного y_score и k_values.
    Args:
        data: pd.DataFrame
            Данные для расчёта метрик
        y_score: str
            Имя столбца ранжирования
        k_values: List[int] = [500, 1000, 1500, 2000, 2500]
            Список значений K для расчёта метрик
        business_rule_mode: bool = False
            Флаг, указывающий, что нужно использовать бизнес-правило для фильтрации наблюдений
        ascending: bool = False
            Флаг, указывающий, что нужно ранжировать по возрастанию y_score
    Требования к `data`:
    - Столбец `'target'` содержит целевую переменную
    - Столбец `'month'` содержит месяц, к которому относится строка
    - Если `business_rule_mode=True`, то в data должен присутствовать столбец `'rule_flag'`
    """
    for k in k_values:
        if business_rule_mode:
            df_k = data[data['rule_flag'] == 1].copy()
        else:
            df_k = data.copy()

        df_k = (
            df_k
            .sort_values(by=y_score, ascending=ascending)
            .groupby('month')
            .head(k)
        )

        tp = df_k[df_k['target'] == 1].groupby('month').size()
        fp = df_k[df_k['target'] == 0].groupby('month').size()

        total_positives = data.groupby('month').target.sum()
        k_eff = df_k.groupby('month').size()

        tp, fp = tp.align(fp, fill_value=0)
        tp, total_positives = tp.align(total_positives, fill_value=0)

        precision_k = tp / (tp + fp)
        recall_k = tp / total_positives

        expected_conversions = precision_k * k_eff

        print(
            f"K={k:<4} | "
            f"Precision: {precision_k.mean():.2%} | "
            f"Recall: {recall_k.mean():.2%} | "
            f"Expected Conversions: {expected_conversions.mean():.2f}"
        )


def calculate_random_baseline(
    df: pd.DataFrame,
    k_values: List[int] = [500, 1000, 1500, 2000, 2500],
) -> None:
    """
    Выводит print() с рассчитанными Precision@K, Coverage@K и Expected Conversions@K для случайного отбора.
    Метрики усредняются по месяцам (macro average).
    Args:
        df: pd.DataFrame
            Данные для расчёта метрик
        k_values: List[int] = [500, 1000, 1500, 2000, 2500]
            Список значений K для расчёта метрик
    Требования к `df`:
    - Столбец `'target'` содержит целевую переменную
    - Столбец `'month'` содержит месяц, к которому относится строка
    """
    positives = df.groupby('month').target.sum()

    all_observations = df.groupby('month').size()

    positives, all_observations = positives.align(all_observations, fill_value=0)

    precision = positives / all_observations

    for k in k_values:
        coverage = (k / all_observations).mean()
        expected_conversions = precision * k

        print(
            f"K={k:<4} | "
            f"Precision: {precision.mean():.2%} | "
            f"Coverage: {coverage.mean():.2%} | "
            f"Expected Conversions: {expected_conversions.mean():.2f}"
        )