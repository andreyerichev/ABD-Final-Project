"""
Модуль хранит функции для предобработки исходных данных.
- Заполнение пропусков
- Приведение типов
- Преобразование дат
"""

import pandas as pd
from typing import List, Tuple


def process_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Приводит типы столбцов к необходимому типу.
    - Даты преобразуются в datetime
    - Логические столбцы преобразуются в int8
    Args:
        df: pd.DataFrame
            Исходный DataFrame
    Returns:
        pd.DataFrame
            Подготовленный DataFrame
    """
    df = df.copy()

    dttm_cols = [
        'month',
        'first_trx_month_inn',
        'last_active_month_inn',
        'first_month_p1',
        'first_month_p2',
        'first_month_p3',
        'first_month_altpay5',
    ]

    cat_cols = [
        'inn_status',
        'top_mcc_group_inn',
        'who_is_this_first_inn',
        'current_segment_inn',
    ]

    bool_cols = df.select_dtypes(include='bool').columns

    for col in dttm_cols:
        df[col] = pd.to_datetime(df[col])

    for col in cat_cols:
        df[col] = df[col].astype('category')

    for col in bool_cols:
        df[col] = df[col].astype('int8')

    return df


def fill_top_mcc_group_inn(df: pd.DataFrame) -> pd.DataFrame:
    """
    Заполняет пропуски в столбце 'top_mcc_group_inn'.
    Логика: если top_mcc_group_inn is NaN, то:
    - если inn_status == 'Churn' или 'New Churn', то заполняем 'Churn'
    - если inn_status == 'Active' или 'Reborn', то заполняем 'No MCC Group'
    - иначе оставляем NaN
    
    Args:
        df: pd.DataFrame
            Исходный DataFrame
    Returns:
        pd.DataFrame
        DataFrame с заполненными пропусками в столбце 'top_mcc_group_inn'
    """
    df = df.copy()
    df['top_mcc_group_inn'] = df['top_mcc_group_inn'].cat.add_categories(['No MCC Group', 'Churn'])
    mask = df['top_mcc_group_inn'].isna()
    df.loc[mask & df['inn_status'].isin(['Churn', 'New Churn']), 'top_mcc_group_inn'] = 'Churn'
    df.loc[mask & df['inn_status'].isin(['Active', 'Reborn']), 'top_mcc_group_inn'] = 'No MCC Group'
    return df


def fill_last_active_month_inn(df: pd.DataFrame) -> pd.DataFrame:
    """
    Заполняет пропуски в столбце 'last_active_month_inn'.
    Логика: если last_active_month_inn is NaN, то заполняем минимальный 'last_active_month_inn' по всему датасету минус 2 месяца
    поскольку если компания была привлечена давно, то её последний активный месяц не может быть определён из-за отсутствия данных о транзакциях,
    поэтому заполняем пропуски константным значением: минимальный 'last_active_month_inn' по всему датасету минус 2 месяца
    
    Args:
        df: pd.DataFrame
            Исходный DataFrame
    Returns:
        pd.DataFrame
        DataFrame с заполненными пропусками в столбце 'last_active_month_inn'
    """
    df = df.copy()
    min_last_active_month = df['last_active_month_inn'].min()
    mask = df['last_active_month_inn'].isna()
    df.loc[mask, 'last_active_month_inn'] = min_last_active_month - pd.DateOffset(months=2)
    return df


def fill_other_active_product_cnt(df: pd.DataFrame) -> pd.DataFrame:
    """
    Заполняет пропуски в столбце 'other_active_product_cnt'.
    Логика: если other_active_product_cnt is NaN, то заполняем 0
    поскольку если нет данных об активных продуктах на какой-либо месяц, считаем что их не было
    
    Args:
        df: pd.DataFrame
            Исходный DataFrame
    Returns:
        pd.DataFrame
        DataFrame с заполненными пропусками в столбце 'other_active_product_cnt'
    """
    df = df.copy()
    df['other_active_product_cnt'] = df['other_active_product_cnt'].fillna(0)
    return df


def fill_change_to_prev_month(df: pd.DataFrame) -> pd.DataFrame:
    """
    Заполняет пропуски в столбцах вида 'change_to_prev_month'.
    Логика: если нет данных о изменении метрики отношения к предыдущему месяцу,
    заполняем пропуски нейтральным значением 0
    поскольку если нет данных о изменении метрики отношения к предыдущему месяцу, считаем что она не изменилась
    
    Args:
        df: pd.DataFrame
            Исходный DataFrame
    Returns:
        pd.DataFrame
        DataFrame с заполненными пропусками в столбцах с окончанием '_change_to_prev_month'
    """
    df = df.copy()
    change_to_prev_month_cols = [col for col in df.columns if col.endswith("_change_to_prev_month")]
    df[change_to_prev_month_cols] = df[change_to_prev_month_cols].fillna(0)
    return df


def fill_cv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Заполняет пропуски в столбцах вида 'cv'.
    Логика: если нет данных о коэффициенте вариации,
    заполняем пропуски нейтральным значением 0
    поскольку если нет данных о коэффициенте вариации, считаем что он не изменился
    
    Args:
        df: pd.DataFrame
            Исходный DataFrame
    Returns:
        pd.DataFrame
        DataFrame с заполненными пропусками в столбцах, содержащих '_cv_' в названии
    """
    df = df.copy()
    cv_cols = [col for col in df.columns if "_cv_" in col]
    df[cv_cols] = df[cv_cols].fillna(0)
    return df


def fill_who_is_this_first_inn(df: pd.DataFrame) -> pd.DataFrame:
    """
    Заполняет пропуски в столбце 'who_is_this_first_inn'.
    Логика: если who_is_this_first_inn is NaN, то удаляем строку (изначально была всего одна ИНН с NaN в этом столбце)
    
    Args:
        df: pd.DataFrame
            Исходный DataFrame
    Returns:
        pd.DataFrame
        DataFrame с заполненными пропусками в столбце 'who_is_this_first_inn'
    """
    df = df.copy()
    assert df[df['who_is_this_first_inn'].isna()]['inn'].nunique() == 1
    df = df[df['who_is_this_first_inn'].notna()]
    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Заполняет пропуски в DataFrame.
    
    Args:
        df: pd.DataFrame
            Исходный DataFrame
    Returns:
        pd.DataFrame
        DataFrame с заполненными пропусками
    """
    df = df.copy()

    # Категориальные признаки
    df = fill_top_mcc_group_inn(df)
    df = fill_last_active_month_inn(df)
    df = fill_who_is_this_first_inn(df)
    
    # Числовые признаки
    df = fill_other_active_product_cnt(df)
    df = fill_change_to_prev_month(df)
    df = fill_cv(df)

    return df


def train_test_month_split(df: pd.DataFrame, border_month: str = '2025-12-01', get_valid: bool = False):
    """
    Разбивает DataFrame на train и test по месяцам.
    Логика: делим на train и test 80% и 20% соответственно, если get_valid = False, иначе делим на train, valid и test 60% 20% и 20% соответственно.
    Args:
        df: pd.DataFrame
            Исходный DataFrame
        get_valid: bool = False
            Флаг, указывающий, что нужно вернуть valid DataFrame
    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series] или tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        Tuple с train и test DataFrames и месяцами для train и test, если get_valid = False, иначе Tuple с train, valid и test DataFrames и месяцами для train, valid и test
    """
    df = df.copy()
    border_month = pd.to_datetime(border_month)

    if get_valid:
        train = df[df['month'] < border_month - pd.DateOffset(months=3)]
        valid = df[(df['month'] >= border_month - pd.DateOffset(months=3)) & (df['month'] < border_month)]
        test = df[df['month'] >= border_month]
        return train, valid, test
        
    else:
        train = df[df['month'] < border_month]
        test = df[df['month'] >= border_month]
        return train, test