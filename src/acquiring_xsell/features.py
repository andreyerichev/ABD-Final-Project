"""
Модуль хранит функции для генерации признаков.
"""

import pandas as pd

PRODUCT_CONFIG = {
    "p1p2": {
        "first_month_col": "first_month_p1p2",
        "banned_mcc_groups": ["Charity"],
        "is_relevant_mcc_col": None,
        "is_relevant_turnover_col": None,
    },
    "p3": {
        "first_month_col": "first_month_p3",
        "banned_mcc_groups": ["Charity"],
        "is_relevant_mcc_col": "is_relevant_mcc_p3",
        "is_relevant_turnover_col": None,
    },
    "altpay5": {
        "first_month_col": "first_month_altpay5",
        "banned_mcc_groups": None,
        "is_relevant_mcc_col": "is_relevant_mcc_altpay5",
        "is_relevant_turnover_col": "is_relevant_turnover_altpay5",
    },
}

ACTIVE_STATUSES = ["Active", "Reborn"]


def add_is_first_trx_month(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет столбец `is_first_trx_month` в DataFrame.
    Значением являетется 1, если месяц соответствует месяцу первой транзакции.
    """
    df = df.copy()
    df['is_first_trx_month'] = (df["month"] == df["first_trx_month_inn"]).astype('int8')
    return df


def add_is_first_history_month(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет столбец `is_first_history_month` в DataFrame.
    Значением являетется 1, если месяц соответствует минимальному месяцу в датасете.
    """
    df = df.copy()
    df['is_first_history_month'] = (df["month"] == df["month"].min()).astype('int8')
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


def add_first_month_p1p2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Заменяет столбцы `first_month_p1` и `first_month_p2` на столбец `first_month_p1p2` в DataFrame,
    который указывает когда клиент подключил продукт P1P2.
    В качестве значения берется минимальная дата из дат подключения продуктов P1 и P2.
    """
    df = df.copy()
    df['first_month_p1p2'] = df[['first_month_p1', 'first_month_p2']].min(axis=1)
    df = df.drop(columns=['first_month_p1', 'first_month_p2']).rename(columns={'first_month_p1p2': 'first_month_p1p2'})
    return df


# ВАЖНО: запускать СТРОГО перед add_rule_flag_product
def add_has_product(df: pd.DataFrame, product: str) -> pd.DataFrame:
    """
    Добавляет столбец `has_{product}` в DataFrame.
    Значением являетется 1, если на момент текущего месяца `month` клиент уже имеет подключённый {product}.
    """
    if f"first_month_{product}" not in df.columns:
        raise ValueError(f"Столбец `first_month_{product}` не найден в DataFrame")

    df = df.copy()
    config = PRODUCT_CONFIG[product]
    df[f"has_{product}"] = (df['month'] >= df[config["first_month_col"]]).astype('int8')
    return df


# ВАЖНО: запускать СТРОГО после add_has_product
def add_rule_flag_product(df: pd.DataFrame, product: str) -> pd.DataFrame:
    """
    Добавляет столбец `rule_flag_{product}` в DataFrame, который указывает,
    релевантен ли клиент для подключения {product} по бизнес-правилам.
    """
    if f"has_{product}" not in df.columns:
        raise ValueError(f"Столбец `has_{product}` не найден в DataFrame")

    df = df.copy()
    config = PRODUCT_CONFIG[product]

    mask = pd.Series(True, index=df.index)

    mask &= df[f"has_{product}"] == 0  # Клиенты у которых уже есть продукт являются нерелевантными

    mask &= df["inn_status"].isin(ACTIVE_STATUSES)

    if config["banned_mcc_groups"]:
        mask &= ~df["top_mcc_group_inn"].isin(config["banned_mcc_groups"])

    col = config["is_relevant_mcc_col"]
    if col:
        mask &= df[col].fillna(False)

    col = config["is_relevant_turnover_col"]
    if col:
        mask &= df[col].fillna(False)

    df[f"rule_flag_{product}"] = mask.astype("int8")

    return df


def add_became_active_recently(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет столбец `became_active_recently` в DataFrame.
    Значением являетется 1, если lifetime_month_streak_inn > 0 и lifetime_month_streak_inn <= 3.
    """
    df = df.copy()
    df['became_active_recently'] = (
        (df['lifetime_month_streak_inn'] > 0) &
        (df['lifetime_month_streak_inn'] <= 3)
    ).astype('int8')
    return df


def drop_dttm_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Удаляет все столбцы с датами из DataFrame
    (кроме 'month' — он нужен для агрегации метрик качества по месяцам).
    Args:
        df: pd.DataFrame
            DataFrame с признаками
    Returns:
        pd.DataFrame
        DataFrame без столбцов с датами
    """
    df = df.copy()
    cols_to_drop = [
        'inn',
        "last_active_month_inn",
        "first_trx_month_inn",
    ]
    df = df.drop(columns=cols_to_drop)
    return df


def build_features(df: pd.DataFrame, drop_dttm_columns: bool = True) -> pd.DataFrame:
    """
    Собирает все признаки в DataFrame.
    Args:
        df: pd.DataFrame
            DataFrame с предобработанными столбцами (process_dtypes, fill_missing_values)
    Returns:
        pd.DataFrame
        DataFrame с признаками
    """
    df = df.copy()
    df = add_is_first_trx_month(df)
    df = add_is_first_history_month(df)
    df = add_months_since_last_active_month(df)
    df = add_months_since_first_trx(df)
    df = add_first_month_p1p2(df)
    df = add_has_product(df, "p1p2")
    df = add_has_product(df, "p3")
    df = add_has_product(df, "altpay5")
    df = add_rule_flag_product(df, "p1p2")
    df = add_rule_flag_product(df, "p3")
    df = add_rule_flag_product(df, "altpay5")
    df = add_became_active_recently(df)
    if drop_dttm_columns:
        df = drop_dttm_cols(df)
    return df


def make_product_df(df: pd.DataFrame, product: str) -> pd.DataFrame:
    """
    Собирает датафрейм для переданного продукта и определяет таргет как последний месяц перед подключением
    Args:
        df: pd.DataFrame
            DataFrame с признаками (build_features)
        product: str
            Название продукта
    Returns:
        pd.DataFrame
        DataFrame с данными для переданного продукта
    """
    df = df.copy()
    first_month_col = PRODUCT_CONFIG[product]["first_month_col"]

    # Оставляем в df только строки до подключения {product} (first_month < month)
    # Важно: столбец first_month_col должен быть КОНСТАНТНЫМ для всех строк в рамках одного ИНН
    df = df[
        (df['month'] < df[first_month_col]) | (df[first_month_col].isna())
    ].reset_index(drop=True)

    df['target'] = (
        df['month'] + pd.DateOffset(months=1) == df[first_month_col]
    ).astype('int8')

    df = df.drop(columns=[first_month_col, f"has_{product}"])

    # Устраняем Data Leakage по другим продуктам
    # Логика: если продукт будет подключён в будущем (first_month > current month),
    # то мы не должны "видеть" эту информацию → зануляем (NaN)
    other_first_month_product_cols = [
        PRODUCT_CONFIG[p]["first_month_col"] 
        for p in PRODUCT_CONFIG if p != product
    ]

    df = df.drop(columns=other_first_month_product_cols)

    return df