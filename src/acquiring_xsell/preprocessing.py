"""
Модуль хранит функции для предобработки исходных данных.
- Заполнение пропусков
- Преобразование дат
- Разделение на обучающую и тестовую выборки
- Разделение исходного датасета на датасеты для каждого продукта
"""

import pandas as pd


def add_first_month_p1p2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет столбец `first_month_p1p2` в DataFrame, который указывает,
    когда клиент подключил продукт P1P2. В качестве значения берется минимальная дата из дат подключения продуктов P1 и P2.
    После чего удаляется столбец `first_month_p1` и `first_month_p2`.
    Args:
        df: pd.DataFrame
            Исходный DataFrame
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