import argparse
import yaml
from typing import Dict, Any, Union

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from core.creator_objects import get_object_class


class ExperimentRunner:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.experiment_name = self.config.get('experiment_name')
        self.random_state = self.config.get('random_state')

        self.divide_dataset = self.config.get("divide_dataset")
        if not self.divide_dataset:
            self.train_path, self.test_path = self.config.get('corpora').values()
        else:
            self.data_path = self.config.get('corpora').values()
        self.splitter_path = self.config.get('splitter')

        self.classificators_params = self.config.get("classificators")
        self.scaler_params = self.config.get('scaler')

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Загружает данные из .yaml файла.
        :param config_path: Путь до конфига.
        :return:
        """
        with open(config_path, 'r', encoding='utf8') as file:
            return yaml.safe_load(file)

    def _get_objects_from_configs(self, objects_params: Union[dict, list]):
        """
        Возвращает объекты класса модели, передавая в класс необходимые параметры.
        :param objects_params: Данные, содержащие класс и передаваемые данные.
        :return: Объект класса модели.
        """
        objects = []
        if not isinstance(objects_params, list):
            objects_params = [objects_params]

        if len(objects_params) == 1:
            return get_object_class(**objects_params[0])

        for object in objects_params:
            objects.append(get_object_class(**object))

        return objects

    def _load_csv(self, csv_path: str):
        """
        Загрузка .csv файла.
        :param csv_path: Путь до файла.
        :return:
        """
        return pd.read_csv(csv_path)

    def _preprocessing_data(self, X, x_test):
        scaler = self._get_objects_from_configs(self.scaler_params)
        return scaler.fit_transform(X), scaler.transform(x_test)

    def _get_dataset(self):
        """
        Загрузка тренировочного и тестового датасетов из .csv файла.
        :return:
        """
        if self.divide_dataset:
            pass

        train_data = self._load_csv(self.train_path).dropna()
        test_data = self._load_csv(self.test_path).drop("SeriousDlqin2yrs", axis=1).dropna()

        X = train_data.drop("SeriousDlqin2yrs", axis=1)
        y = train_data['SeriousDlqin2yrs']

        X_scaled, x_test_scaled = self._preprocessing_data(X, test_data)
        return X_scaled, y, x_test_scaled

    def run(self):
        """
        Основной метод контроля и запуска алгоритма.
        """
        classificator = self._get_objects_from_configs(self.classificators_params)
        splitter = self._get_objects_from_configs(self.splitter_path)

        X, y, x_test = self._get_dataset()
        X_train, X_val, y_train, y_val = splitter.split(X, y)

        classificator.fit(X_train, y_train)
        predict = classificator.predict(X_val)
        acc = accuracy_score(y_val, predict)
        recall = recall_score(y_val, predict)
        precision = precision_score(y_val, predict)
        f1 = f1_score(y_val, predict)
        print(y_val.value_counts())
        print(f"accuracy_score={acc}, recall={recall}, precision={precision}, f1={f1}")


def main():
    """
    Запуск runner через командную строку.
    """
    parser = argparse.ArgumentParser(description='Парсер данных командной строки.')
    parser.add_argument(
        "-c",
        "--experiment-config",
        type=str,
        required=True,
        help="Путь к файлу конфигурации"
    )
    args = parser.parse_args()
    runner = ExperimentRunner(args.experiment_config)
    runner.run()


if __name__ == '__main__':
    main()
