"""
Модуль хранит функции для генерации признаков.
"""

import pandas as pd


def add_rule_flag_p1_p2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет столбец `rule_flag` в DataFrame, который указывает,
    релевантен ли клиент для подключения продукта P1P2.
    Релевантными считаются клиенты:
    - с активным статусом
    - с группой MCC, не являющейся Charity

    Args:
        df: pd.DataFrame
            Исходный DataFrame
    """
    active_statuses = ["Active", "Reborn"]
    banned_mcc_groups = ["Charity"]

    df = df.copy()
    
    df["rule_flag"] = (
        df["inn_status"].isin(active_statuses)
        & ~df["top_mcc_group_inn"].isin(banned_mcc_groups)
    ).astype("int8")

    return df


def add_rule_flag_p3(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет столбец `rule_flag` в DataFrame, который указывает,
    релевантен ли клиент для подключения продукта P3.
    Релевантными считаются клиенты:
    - с активным статусом
    - с группой MCC, не являющейся Charity
    - с группой MCC, являющейся Charity
    """
    active_statuses = ["Active", "Reborn"]
    banned_mcc_groups = ["Charity"]

    df = df.copy()
    
    df["rule_flag"] = (
        df["inn_status"].isin(active_statuses)
        & ~df["top_mcc_group_inn"].isin(banned_mcc_groups)
        & df["is_relevant_mcc_p3"]
    ).astype("int8")
    return df


def add_months_since_last_active_month(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет столбец `months_since_last_active_month` в DataFrame.
    Значением является количество месяцев с момента последней активности клиента.
    Args:
        df: pd.DataFrame
            Исходный DataFrame
    """
    df = df.copy()
    df["months_since_last_active_month"] = (
        (df["month"].dt.year - df["last_active_month_inn"].dt.year) * 12 +
        (df["month"].dt.month - df["last_active_month_inn"].dt.month)
    )
    return df


def add_months_since_first_trx(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет столбец `months_since_first_trx` в DataFrame.
    Значением является количество месяцев с момента первой транзакции клиента.

    Args:
        df: pd.DataFrame
            Исходный DataFrame
    """
    df = df.copy()
    df["months_since_first_trx"] = (
        (df["month"].dt.year - df["first_trx_month_inn"].dt.year) * 12 +
        (df["month"].dt.month - df["first_trx_month_inn"].dt.month)
    )
    return df


def add_has_p1_p2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет столбец `has_p1_p2` в DataFrame.
    Значением являетется True, если клиент подключил продукт P1P2.
    Args:
        df: pd.DataFrame
            Исходный DataFrame
    """
    df = df.copy()
    df["has_p1_p2"] = df["first_month_p1_p2"].notna()
    return df


def add_has_p3(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет столбец `has_p3` в DataFrame.
    Значением являетется True, если клиент подключил продукт P3.
    Args:
        df: pd.DataFrame
            Исходный DataFrame
    """
    df = df.copy()
    df["has_p3"] = df["first_month_p3"].notna()
    return df


def add_has_altpay5(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет столбец `has_altpay5` в DataFrame.
    Значением являетется True, если клиент подключил продукт Altpay5.
    Args:
        df: pd.DataFrame
            Исходный DataFrame
    """
    df = df.copy()
    df["has_altpay5"] = df["first_month_altpay5"].notna()
    return df