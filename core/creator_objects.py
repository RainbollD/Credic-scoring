import importlib
import inspect
from typing import Dict, Any


def filter_params_class(cls, params):
    """
    Фильтрация всех параметров

    :param cls: Класс, для которого берутся параметры.
    :param params: Все передающиеся параметры.

    :return: Необходимые классу (cls) параметры.
    """
    cls_params = inspect.signature(cls.__init__)
    valid_params = set(cls_params.parameters.keys()) - {'self'}
    return {k: v for k, v in params.items() if k in valid_params}


def get_object_class(class_path: str, params: Dict[str, Any]):
    """
    Создание объектов по строке к классу и параметрам.

    :param class_path: Строка к классу.
    :param params: Словарь параметров.
    :return: Объект класса.
    """
    if params is None:
        params = {}

    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    filtered_params = filter_params_class(cls, params)
    return cls(**filtered_params)
