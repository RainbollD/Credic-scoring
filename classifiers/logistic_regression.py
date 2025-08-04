"""
classifiers.logistic_regression

Модуль содержит реализацию логистической регрессии.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from .base import BaseClassifier

class LogisticRegressionClassifier(BaseClassifier):
    """
    Классификатор на основе логистической регрессии из sklearn.
    """

    def __init__(self, **kwargs):
        """
        Инициализация LogisticRegressionClassifier

        :param kwargs: Параметры для sklearn.linear_model.LogisticRegression.
        """

        super().__init__(**kwargs)
        self.model = LogisticRegression(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Обучает классификатор на обучающей выборке.

        :param X: Признаки.
        :param y: Метки.
        :param kwargs: Дополнительные параметры обучения.
        """

        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказывает метки для входных данных.

        :param X: Признаки.
        :return: Массив предсказанных меток.
        """

        return self.model.predict(X)


