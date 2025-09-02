## 💳 Credit Scoring

### Описание

Проект предназначен для удобного проведения экспериментов с различными классификаторами, разделителями и нормализаторами данных.
Все настройки выполняются через YAML-конфигурационные файлы.

### 🛠 Установка зависимостей

Рекомендуется использовать Python 3.10+ и виртуальное окружение:

```bash
  python3 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
```

### 🚀 Запуск экспериментов

#### 1. Запуск серии экспериментов через управляющий скрипт

```bash
  python3 main.py
```

Скрипт main.py автоматически запускает серию экспериментов, перечисленных внутри файла (например, config/experiments.yaml).

#### 2. Запуск одного эксперимента напрямую

```bash
  python3 -m core.runner -c config/credit_scoring/all_models.yaml
```

-c — путь к YAML-файлу с конфигурацией эксперимента.

Примеры конфигураций находятся в директории config/.

### 📘 Пример конфигурационного файла эксперимента

```yaml
experiment_name: credit_scoring/all_models
random_state: 111

divide_dataset: False

corpora:
  train: data/GiveMeSomeCredit-training.csv
  test: data/GiveMeSomeCredit-testing.csv

splitter:
  class_path: "core.splitters.TrainTestSplitter"
  params:
    shuffle: true
    test_size: 0.3
    random_state: 42

classificators:
  - class_path: "classifiers.XGBoostClassifier"
    params:
      scale_pos_weight: 15
      random_state: 42

scaler:
  - class_path: "core.scalers.StandardScaler"

metrics: [ "f1", "precision", "recall", "accuracy", "auc" ]
```