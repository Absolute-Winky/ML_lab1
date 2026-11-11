# Lab 1 - предсказание цены Mercedes

Лабораторная по ML: регрессия, предсказываю цену подержанного Mercedes по его характеристикам.

## Датасет

Kaggle: https://www.kaggle.com/datasets/mysarahmadbhat/mercedes-used-car-listing

13119 объявлений, таргет `price`. Признаки:
- категориальные: `model`, `transmission`, `fuelType`
- числовые: `year`, `mileage`, `tax`, `mpg`, `engineSize`

## Модель

Ridge-регрессия (линейная регрессия с L2-регуляризацией).

- sklearn: `Ridge`, `alpha` подбирал через `GridSearchCV` (5-fold CV)
- своя реализация: `src/ridge_gd.py`, mini-batch градиентный спуск, гиперпараметры `lr`, `n_iter`, `alpha`, `batch_size`
- baseline: `DummyRegressor` по медиане

## Файлы

- `data/merc.csv` - датасет
- `src/preprocess.py` - очистка, признаки, сплит, препроцессинг
- `src/ridge_gd.py` - своя Ridge
- `lab.ipynb` - EDA, графики, обучение, метрики

## Результаты на test

| Модель | MAE | R2 |
|---|---|---|
| baseline (медиана) | 7647.5 | -0.0374 |
| Ridge sklearn | 3152.7 | 0.8256 |
| своя Ridge | 3166.7 | 0.8217 |

Своя модель почти как sklearn, расхождение меньше 1%. Переобучения нет, train/val/test близко друг к другу.
