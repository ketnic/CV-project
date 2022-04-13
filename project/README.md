# Программный модуль для группировки фотографий интерьеров квартир

## Требуется 
Python 3.8.6
Pip
Make (необязательно)

## Инициализация venv
1. `python -m venv venv` (или `python3 -m venv venv`)
2. `.\venv\Scripts\activate.bat` (или `source ./venv/bin/activate`)
3. `make i` (или `pip install -r requirements.txt`)

## Запуск демонстрационного приложения
1. Инициализировать venv (см. выше)
2. `make run` (или `python run.py`)

## Запуск модулей нейронных сетей
1. Инициализировать venv (см. выше)
2. Запустить make...

## Удалить игнорируемые файлы
`make clean`

## Структура проекта
```
├── app // десктопное приложение
│   └── init.py
├── data // кэш
│   ├── c_data // данные для классификатора
│   └── s_data // данные для сиамской сети
├── dataset
│   └── merge_datasets.py
├── models
│   ├── c_model.h5 // модель классификатора
│   └── s_model.h5 // модель сиамской сети
├── networks
│   ├── classifier
│   │   ├── output // папка для графиков
│   │   ├── autotrain.py
│   │   ├── classify_images.py
│   │   ├── config.py
│   │   ├── create_confusion_matrix.py
│   │   ├── create_model.py
│   │   ├── process_photos.py
│   │   ├── test_classifier.py
│   │   ├── test_grad_cam.py
│   │   ├── test_prediction.py
│   │   └── train.py
│   │
│   ├── similarity
│   │   ├── output // папка для графиков
│   │   ├── autotrain.py
│   │   ├── compare_images.py
│   │   ├── config.py
│   │   ├── create_confusion_matrix.py
│   │   ├── create_siamese_model.py
│   │   ├── CustomDataset.py
│   │   ├── make_datasets.py
│   │   ├── make_samples.py
│   │   ├── roc_auc.py
│   │   ├── test_clustering.py
│   │   └── train.py
│   └── group_images.py
├── utils
│   ├── plots
│   │   ├── confusion_matrix.py
│   │   ├── ggplot.py
│   │   └── roc.py
│   ├── activation.py
│   ├── convertible.py
│   ├── EarlyStoppingByMetricValue.py
│   ├── grad_cam.py
│   ├── image.py
│   ├── inline_print.py
│   ├── log.py
│   ├── loss.py
│   ├── models.py
│   └── optimizers.py
├── .gitignore
├── config.py
├── Makefile
├── README.md
├── requirements.txt
└── run.py
```

## Установка Make на Windows
https://stackoverflow.com/questions/2532234/how-to-run-a-makefile-in-windows
