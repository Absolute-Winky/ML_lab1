# Mercedes Used Car Price Prediction

Лабораторная работа №1 по машинному обучению: регрессия цены подержанного Mercedes-Benz.

## Задача

Предсказать цену автомобиля (`price`) по его характеристикам. Датасет — [Mercedes Used Car Listing](https://www.kaggle.com/datasets/mysarahmadbhat/mercedes-used-car-listing) (Kaggle), 13 119 объявлений о продаже Mercedes в Великобритании.

Датасет содержит 8 предикторов и числовой таргет:

| Тип | Признаки |
|---|---|
| Категориальные (3) | `model`, `transmission`, `fuelType` |
| Количественные (5) | `year`, `mileage`, `tax`, `mpg`, `engineSize` |
| Таргет | `price` |

## Модель

**Ridge-регрессия** (линейная регрессия с L2-регуляризацией) — разобрана на лекциях, не использовалась на практике.

- В фреймворке: `sklearn.linear_model.Ridge`, подбор `alpha` через `GridSearchCV` (5-fold CV).
- Собственная реализация: `src/ridge_gd.py`, класс `MyRidgeGD` — мини-батчевый градиентный спуск. Гиперпараметры: `lr`, `n_iter`, `alpha`, `batch_size`.
- Baseline: `DummyRegressor` (среднее / медиана таргета).

## Структура проекта

```
.
├── data/
│   └── merc.csv              # датасет
├── src/
│   ├── preprocess.py         # очистка, признаки, сплит, препроцессоры
│   └── ridge_gd.py           # собственная реализация Ridge (mini-batch GD)
├── lab.ipynb                 # EDA, обучение, графики, метрики
├── requirements.txt
└── README.md
```

## Пайплайн

1. **Очистка**: удаление выбросов (`engineSize=0`, `mpg>150`, `year<2000`, `price>100000`), редкие модели (<50 объявлений) объединяются в `Other`. Добавлены признаки `car_age = 2021 - year` и `mpy = mileage / car_age`.
2. **Сплит**: train/val/test = 60/20/20 со стратификацией по 5 квантильным корзинам цены.
3. **Препроцессинг**: `StandardScaler` для числовых признаков, `OneHotEncoder` для категориальных. `fit` только на train — data leak отсутствует.
4. **Baseline**: медиана таргета.
5. **sklearn Ridge**: подбор `alpha` по сетке `[0.1, 1, 10, 100, 1000]`, 5-fold CV на train.
6. **Собственная Ridge**: перебор `lr` и `alpha`, сравнение с фреймворком.

## Результаты

MAE по сплитам:

| Модель | train | val | test |
|---|---|---|---|
| Baseline (медиана) | 7678.2 | 7671.6 | 7647.5 |
| Ridge (sklearn), alpha=1 | 3063.5 | 3014.4 | 3152.7 |
| MyRidgeGD, lr=0.2, alpha=1 | 3072.5 | 3036.4 | 3166.7 |

На test: Ridge (sklearn) — MSE 20 576 433, R² 0.8256; MyRidgeGD — MSE 21 029 100, R² 0.8217; baseline — R² -0.0374.

Собственная реализация отличается от sklearn менее чем на 1%, переобучения нет (train/val/test близки).

## Запуск

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows
pip install -r requirements.txt
```

Открыть `lab.ipynb` в Jupyter / VS Code и выполнить ячейки сверху вниз.

## Стек

Python 3.12, pandas, numpy, scikit-learn, matplotlib, seaborn, jupyter.
