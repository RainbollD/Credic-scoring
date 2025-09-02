import importlib
import inspect
from typing import Dict, Any, Union


def filter_params_class(cls, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Фильтрация параметров конструктора класса.
    Если в сигнатуре конструктора присутствуют **kwargs, возвращаем все параметры.
    Если есть недопустимые параметры, выводим предупреждение.

    :param cls: Класс, для которого берутся параметры.
    :param params: Все передающиеся параметры.
    :return: Отфильтрованные параметры.
    """
    cls_params = inspect.signature(cls.__init__)
    parameters = cls_params.parameters

    # Проверка на **kwargs
    has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values())

    if has_kwargs:
        return params

    valid_params = set(parameters.keys()) - {'self'}

    # Проверка на недопустимые параметры
    invalid_params = [k for k in params if k not in valid_params]
    if invalid_params:
        raise TypeError(f"Недопустимые параметры: {', '.join(invalid_params)}")

    return {k: v for k, v in params.items() if k in valid_params}


def get_object_class(class_path: str, params: Dict[str, Any] = None):
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
