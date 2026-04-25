"""
Модуль хранит функции для генерации признаков.
"""

import pandas as pd


def add_is_first_trx_month(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет столбец `is_first_trx_month` в DataFrame.
    Значением являетется 1, если месяц соответствует месяцу первой транзакции.
    """
    df = df.copy()
    df['is_first_trx_month'] = (df["month"] == df["first_trx_month_inn"]).astype('int8')
    return df


def add_is_left_time_border(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет столбец `is_left_time_border` в DataFrame.
    Значением являетется 1, если месяц соответствует минимальному месяцу в датасете.
    """
    df = df.copy()
    df['is_left_time_border'] = (df["month"] == df["month"].min()).astype('int8')
    return df


def add_rule_flag_p1p2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет столбец `rule_flag` в DataFrame, который указывает,
    релевантен ли клиент для подключения продукта P1P2.
    Релевантными считаются клиенты:
    - с активным статусом
    - с группой MCC, не являющейся Charity
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


def add_rule_flag_altpay5(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет столбец `rule_flag` в DataFrame, который указывает,
    релевантен ли клиент для подключения продукта Altpay5.
    Релевантными считаются клиенты:
    - с активным статусом
    - с MCC, отвечающим бизнес-правилу
    - с оборотом, отвечающим бизнес-правилу
    """
    active_statuses = ["Active", "Reborn"]

    df = df.copy()
    
    df["rule_flag"] = (
        df["inn_status"].isin(active_statuses)
        & df["is_relevant_mcc_altpay5"]
        & df["is_relevant_turnover_altpay5"]
    ).astype("int8")

    return df


def add_months_since_last_active_month(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет столбец `months_since_last_active_month` в DataFrame.
    Значением является количество месяцев с момента последней активности клиента.
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

    """
    df = df.copy()
    df["months_since_first_trx"] = (
        (df["month"].dt.year - df["first_trx_month_inn"].dt.year) * 12 +
        (df["month"].dt.month - df["first_trx_month_inn"].dt.month)
    )
    return df


def add_has_p1p2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет столбец `has_p1p2` в DataFrame.
    Значением являетется True, если клиент подключил продукт P1P2.
    """
    df = df.copy()
    df["has_p1p2"] = df["first_month_p1p2"].notna()
    return df


def add_has_p3(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет столбец `has_p3` в DataFrame.
    Значением являетется True, если клиент подключил продукт P3.
    """
    df = df.copy()
    df["has_p3"] = df["first_month_p3"].notna()
    return df


def add_has_altpay5(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет столбец `has_altpay5` в DataFrame.
    Значением являетется True, если клиент подключил продукт Altpay5.
    """
    df = df.copy()
    df["has_altpay5"] = df["first_month_altpay5"].notna()
    return df


def add_first_month_p1p2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет столбец `first_month_p1p2` в DataFrame, который указывает,
    когда клиент подключил продукт P1P2. В качестве значения берется минимальная дата из дат подключения продуктов P1 и P2.
    После чего удаляется столбец `first_month_p1` и `first_month_p2`.
    """
    df = df.copy()

    df['first_month_p1p2'] = df[['first_month_p1', 'first_month_p2']].min(axis=1)
    
    df = df.drop(columns=['first_month_p1', 'first_month_p2'])

    return df


# TODO: Добавить функцию для создания датасета для переданного продукта
def make_product_df(base_df: pd.DataFrame, product: str) -> pd.DataFrame:
    """
    Собирает датафрейм для переданного продукта
    Args:
        base_df: pd.DataFrame
            Исходный DataFrame
        product: str
            Название продукта
    """