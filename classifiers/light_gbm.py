"""
classifiers.xgboost

Модуль содержит реализацию xgboost.XGBClassifier.
"""

import numpy as np
import lightgbm as lgb

from .base import BaseClassifier


class LightGDMClassifier(BaseClassifier):
    """
    Классификатор на основе XGBClassifier.
    """

    def __init__(self, **kwargs):
        """
        Инициализация XBoostClassifier

        :param kwargs: Параметры для xgboost.XGBClassifier.
        """
        super().__init__(**kwargs)
        self.model = lgb(**kwargs)

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
