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

    bool_cols = df.select_dtypes(include='bool').columns

    for col in dttm_cols:
        df[col] = pd.to_datetime(df[col])   

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
    Логика: если who_is_this_first_inn is NaN, то удаляем строку
    
    Args:
        df: pd.DataFrame
            Исходный DataFrame
    Returns:
        pd.DataFrame
        DataFrame с заполненными пропусками в столбце 'who_is_this_first_inn'
    """
    df = df.copy()
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


def simple_month_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Разбивает DataFrame на train и test по месяцам.
    Логика: делим на train и test 80% и 20% соответственно.
    Args:
        df: pd.DataFrame
            Исходный DataFrame
    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        Tuple с train и test DataFrames и месяцами для train и test
    """
    df = df.copy()
    train = df[df['month'] < '2025-12-01']
    test = df[df['month'] >= '2025-12-01']
    month_train = train['month']
    month_test = test['month']
    return train, test, month_train, month_test


class MonthTimeSeriesSplit:
    """
    Разбивает DataFrame на train и test по месяцам.
    Логика: используется expanding window time-series split.
    Args:
        n_splits: int
            Количество разбиений
        min_train_months: int
            Минимальное количество месяцев для train
        test_size_months: int
            Количество месяцев для test
        gap_months: int
           Пропуск между train и test
        month_col: str
            Название столбца с месяцами
    """
    def __init__(
        self,
        n_splits: int = 3,
        min_train_months: int = 6,
        test_size_months: int = 3,
        gap_months: int = 0,
        month_col: str = "month",
    ):
        self.month_col = month_col
        self.n_splits = n_splits
        self.min_train_months = min_train_months
        self.test_size_months = test_size_months
        self.gap_months = gap_months

    def split(self, df: pd.DataFrame):
        df = df.copy()

        months = sorted(df[self.month_col].dropna().unique())

        if len(months) < self.min_train_months + self.test_size_months:
            raise ValueError("Недостаточно месяцев для запрошенной конфигурации")

        splits = []

        for i in range(self.n_splits):

            train_end = self.min_train_months + i * self.test_size_months
            test_start = train_end + self.gap_months
            test_end = test_start + self.test_size_months

            if test_end > len(months):
                break

            train_months = months[:train_end]
            test_months = months[test_start:test_end]

            train_df = df[df[self.month_col].isin(train_months)].copy()
            test_df = df[df[self.month_col].isin(test_months)].copy()

            splits.append((train_df, test_df))

        return splits