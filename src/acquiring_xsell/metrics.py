"""
Модуль для расчёта метрик качества моделей.
"""

import pandas as pd
from typing import List

K_VALUES = [500, 1000, 1500, 2000]


def precision_k(data: pd.DataFrame, y_score_col: str, k: int, ascending: bool = False) -> float:
    """
    Рассчитывает Precision@K, macro average by 'month'.
    Args:
        data: pd.DataFrame
            Данные для расчёта метрик
        y_score_col: str
            Имя столбца, содержащего y_score
        k: int
            Значение K
        ascending: bool = False
            Флаг, указывающий, что нужно ранжировать по возрастанию y_score
    """
    precision_k = (
        data
        .sort_values(by=y_score_col, ascending=ascending)
        .groupby('month')
        .head(k)
        .groupby('month')['target']
        .mean()
    ).fillna(0).mean()
    
    return precision_k


def recall_k(data: pd.DataFrame, y_score_col: str, k: int, ascending: bool = False) -> float:
    """
    Рассчитывает Recall@K, macro average by 'month'.
    Args:
        data: pd.DataFrame
            Данные для расчёта метрик
        y_score_col: str
            Имя столбца, содержащего y_score
        k: int
            Значение K
        ascending: bool = False
            Флаг, указывающий, что нужно ранжировать по возрастанию y_score
    """
    tp = (
        data
        .sort_values(by=y_score_col, ascending=ascending)
        .groupby('month')
        .head(k)
        .groupby('month')['target']
        .sum()
    )
    total_pos = (
        data
        .groupby('month')['target']
        .sum()
    )
    
    tp, total_pos = tp.align(total_pos, fill_value=0)
    
    recall_k = (tp / total_pos).fillna(0).mean()
    
    return recall_k


def expected_conversions_k(data: pd.DataFrame, y_score_col: str, k: int, ascending: bool = False) -> float:
    """
    Рассчитывает Expected Conversions@K, macro average by 'month'.
    Args:
        data: pd.DataFrame
            Данные для расчёта метрик
        y_score_col: str
            Имя столбца, содержащего y_score
        k: int
        ascending: bool = False
            Флаг, указывающий, что нужно ранжировать по возрастанию y_score
    """
    expected_conversions_k = (
        data
        .sort_values(by=y_score_col, ascending=ascending)
        .groupby('month')
        .head(k)
        .groupby('month')['target']
        .sum()
    ).fillna(0).mean()
    
    return expected_conversions_k


def calculate_all_metrics_k(data: pd.DataFrame, y_score_col: str = 'y_proba', k_values: List[int] = K_VALUES, ascending: bool = False) -> pd.DataFrame:
    """
    Рассчитывает все метрики@K, macro average by 'month'.
    Args:
        data: pd.DataFrame
            Данные для расчёта метрик
        y_score_col: str
            Имя столбца, содержащего y_score
        k_values: List[int] = K_VALUES
            Список значений K для расчёта метрик
        ascending: bool = False
            Флаг, указывающий, что нужно ранжировать по возрастанию y_score
    """
    df = pd.DataFrame(columns=['Precision', 'Recall', 'Expected Conversions'], index=k_values)
    for k in k_values:
        df.loc[k] = {
            'Precision': precision_k(data, y_score_col, k, ascending),
            'Recall': recall_k(data, y_score_col, k, ascending),
            'Expected Conversions': expected_conversions_k(data, y_score_col, k, ascending)
        }
    return df








###############################################################
################### DEPRECATED FUNCTIONS ######################
###############################################################



def calculate_top_k_base(
    data: pd.DataFrame,
    y_score: str,
    k: int,
    rule_flag_col: str = None,
    ascending: bool = False,
):
    """
    Рассчитывает агрегированные по месяцам метрики для top-K клиентов.
    Args:
        data: pd.DataFrame
            Данные для расчёта метрик
        y_score: str
            Имя столбца ранжирования
        k: int
            Значение K
        rule_flag_col: str = None
            Имя столбца, содержащего бизнес-правило для фильтрации наблюдений
        ascending: bool = False
            Флаг, указывающий, что нужно ранжировать по возрастанию y_score
    Требования к `data`:
    - Столбец `'target'` содержит целевую переменную
    - Столбец `'month'` содержит месяц, к которому относится строка
    - Если `rule_flag_col` не None, то в data должен присутствовать столбец `rule_flag_col`
    """
    if rule_flag_col is not None:
        if rule_flag_col not in data.columns:
            raise ValueError(f"'{rule_flag_col}' not found in data.columns")
        df = data[data[rule_flag_col] == 1].copy()
    else:
        df = data.copy()

    df_k = (
        df.sort_values(by=y_score, ascending=ascending)
        .groupby('month').head(k)
    )

    tp = df_k[df_k['target'] == 1].groupby('month').size()
    fp = df_k[df_k['target'] == 0].groupby('month').size()

    total_positives = data.groupby('month').target.sum()
    k_eff = df_k.groupby('month').size()

    tp, fp = tp.align(fp, fill_value=0)
    tp, total_positives = tp.align(total_positives, fill_value=0)
    tp, k_eff = tp.align(k_eff, fill_value=0)

    return tp, fp, total_positives, k_eff


def precision_at_k(tp: pd.Series, fp: pd.Series) -> pd.Series:
    denom = tp + fp
    return (tp / denom).fillna(0)


def recall_at_k(tp: pd.Series, total_pos: pd.Series) -> pd.Series:
    return (tp / total_pos).fillna(0)


def expected_conversions_at_k(tp: pd.Series) -> pd.Series:
    return tp.fillna(0)


def calculate_metrics(
    data: pd.DataFrame,
    y_score: str,
    k_values: List[int] = K_VALUES,
    rule_flag_col: str = None,
    ascending: bool = False,
) -> None:
    """
    Выводит print() с рассчитанными средними по месяцам значениями (macro average)
    Precision@K, Recall@K и Expected Conversions@K для заданного y_score и k_values.
    Args:
        data: pd.DataFrame
            Данные для расчёта метрик
        y_score: str
            Имя столбца ранжирования
        k_values: List[int] = K_VALUES
        rule_flag_col: str = None
        ascending: bool = False
    """
    for k in k_values:

        tp, fp, total_pos, k_eff = calculate_top_k_base(
            data=data,
            y_score=y_score,
            k=k,
            rule_flag_col=rule_flag_col,
            ascending=ascending
        )

        precision = precision_at_k(tp, fp)
        recall = recall_at_k(tp, total_pos)
        expected_conversions = expected_conversions_at_k(tp)

        print(
            f"K={k:<4} | "
            f"Precision: {precision.mean():.2%} | "
            f"Recall: {recall.mean():.2%} | "
            f"Expected Conversions: {expected_conversions.mean():.2f}"
        )


def calculate_random_baseline(
    data: pd.DataFrame, 
    k_values: List[int] = K_VALUES
) -> None:
    """
    Выводит print() с рассчитанными ожидаемыми значениями Precision@K, Selection Rate@K и Expected Conversions@K для случайного отбора.
    Args:
        data: pd.DataFrame
            Данные для расчёта метрик
        k_values: List[int] = K_VALUES
            Список значений K для расчёта метрик
    """
    p = data.groupby('month').target.sum()
    n = data.groupby('month').size()

    p, n = p.align(n, fill_value=0)

    precision = p / n

    for k in k_values:
        selection_rate = k / n
        expected_conversions = precision * k

        print(
            f"K={k:<4} | "
            f"Expected Precision: {precision.mean():.2%} | "
            f"Selection Rate: {selection_rate.mean():.2%} | "
            f"Expected Conversions: {expected_conversions.mean():.2f}"
        )