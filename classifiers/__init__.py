"""
classifiers

Модуль содержит импорты и алиасы для основных классификаторов
"""

from .base import BaseClassifier
from .logistic_regression import LogisticRegressionClassifier
from .xgboost import XGBoostClassifier

__all__ = [
    'BaseClassifier',
    'LogisticRegressionClassifier',
    'XGBoostClassifier'
]
