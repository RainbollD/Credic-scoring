"""
classifiers

Модуль содержит импорты и алиасы для основных классификаторов
"""

from .base import BaseClassifier
from .logistic_regression import LogisticRegressionClassifier

__all__ = [
    'BaseClassifier',
    'LogisticRegressionClassifier'
]
