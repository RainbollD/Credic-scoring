"""
core.splitter

Модуль для различных разделения данных для обучения.
"""

from typing import Iterator, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


class KFoldSplitter:
    """
    Класс для разделения данных с помощью sklearn.model_selection.KFold
    """

    def __init__(self, **kwargs):
        """
        Инициализация KFoldSplitter.

        :param kwargs: Параметры для sklearn.model_selection.KFold.
        """
        self.kf = KFold(**kwargs)

    def split(self, data: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Возвращает индексы обучающей и тестовой выборки.

        :param data: Данные для разбиения.
        :return: Возвращает индексы выборок: (train_idx, test_idx)
        """
        return self.kf.split(data)


class StratifiedKFoldSplitter:
    """
    Класс для разделения данных с помощью sklearn.model_selection.StratifiedKFold
    """

    def __init__(self, **kwargs):
        """
        Инициализация StratifiedKFold.

        :param kwargs: Параметры для sklearn.model_selection.StratifiedKFold.
        """
        self.skf = StratifiedKFold(**kwargs)

    def split(self, X: pd.DataFrame, y: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Возвращает индексы обучающей и тестовой выборки.

        :param X: Данные обучающей выборки для разбиения.
        :param X: Данные валидационной выборки для разбиения.
        :return: Возвращает индексы выборок: (train_idx, test_idx)
        """
        return self.skf.split(X, y)


class TrainTestSplitter:
    """
        Класс для разделения данных с помощью sklearn.model_selection.train_test_split
    """

    def __init__(self, **kwargs):
        """
        Инициализация train_test_split.

        :param kwargs: Параметры для sklearn.model_selection.train_test_split.
        """
        self.feature_splitter = kwargs

    def split(self, X: pd.DataFrame, y: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Возвращает индексы обучающей и тестовой выборки.

        :param X: Данные обучающей выборки для разбиения.
        :param X: Данные валидационной выборки для разбиения.
        :return: Возвращает индексы выборок: (train_idx, test_idx)
        """
        return train_test_split(X, y, **self.feature_splitter)