"""
Модуль хранит функции для загрузки исходных данных.
"""

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_PATH = BASE_DIR / "data" / "raw" / "vkr_data.csv"


def load_data(path: Path = DEFAULT_PATH) -> pd.DataFrame:
    """
    Загружает данные из файла и возвращает DataFrame.
    
    Args:
        path: Path = DEFAULT_PATH
            Путь к файлу. По умолчанию используется DEFAULT_PATH.
    
    Returns:
        pd.DataFrame
            DataFrame с данными
    """
    return pd.read_csv(path, sep=';')