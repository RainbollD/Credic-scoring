"""
classifiers.base

Модуль содержит абстрактный класс для классификаторов
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseClassifier(ABC):
    """
    Абстрактный базовый класс для всех классификаторов
    """

    def __init__(self, **kwargs):
        """
        Инициализация BaseClassifier.

        :param kwargs: Дополнительные параметры для дочерних классов
        """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Обучает классификатор на обучающей выборке

        :param X: Признаки.
        :param y: Метки.
        :param kwargs: Дополнительные параметры.
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказывает метки для входных данных

        :param X: Признаки.
        :return: Массив предсказнных меток.
        """