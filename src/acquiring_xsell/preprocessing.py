"""
Модуль хранит функции для предобработки исходных данных.
- Заполнение пропусков
- Приведение типов
- Преобразование дат
"""

import pandas as pd


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

    # 'top_mcc_group_inn' для оттёкших и новых оттёкших клиентов
    # Логика: если компания на момент текущего месяца находится в оттоке, 
    # то она не может иметь группу MCC, так как нет транзакций
    df.loc[
        (df['inn_status'].isin(['Churn', 'New Churn'])) & (df['top_mcc_group_inn'].isna()),
        'top_mcc_group_inn'
    ] = 'Churn'

    # 'top_mcc_group_inn' для активных и возрождённых клиентов
    # Логика: если компания на момент текущего месяца находится в активном или возрождённом статусе, 
    # то она не может не иметь группу MCC
    df.loc[
        df['inn_status'].isin(['Active', 'Reborn']) & (df['top_mcc_group_inn'].isna()),
        'top_mcc_group_inn'
    ] = 'No MCC Group'

    # 'last_active_month_inn' для старого привлечения
    # Логика: если компания была привлечена давно, то её последний активный месяц не может быть определён, 
    # поэтому заполняем пропуски константным значением: минимальный 'last_active_month_inn' по всему датасету минус 2 месяца
    df.loc[
        df['last_active_month_inn'].isna(),
        'last_active_month_inn'
    ] = df['last_active_month_inn'].min() - pd.DateOffset(months=2)

    # 'other_active_product_cnt'
    # Логика: если нет данных об активных продуктах на какой-либо месяц, считаем что их не было
    df['other_active_product_cnt'] = df['other_active_product_cnt'].fillna(0)

    # 'change_to_prev_month' столбцы
    # Логика: если нет данных о изменении метрики отношения к предыдущему месяцу,
    # заполняем пропуски нейтральным значением 0 и добавляем флаги 'is_first_trx_month' и 'is_left_time_border'
    change_to_prev_month_cols = [col for col in df.columns if col.endswith("_change_to_prev_month")]
    df[change_to_prev_month_cols] = df[change_to_prev_month_cols].fillna(0)

    # С коэффициентами вариации логика та же
    cv_cols = [col for col in df.columns if "_cv_" in col]
    df[cv_cols] = df[cv_cols].fillna(0)

    # 'who_is_this_first_inn'
    # Было обнаружено всего 15 наблюдений с пропущенным значением в этом столбце,
    # поэтому удаляем их из датасета
    df = df[df['who_is_this_first_inn'].notna()]

    return df