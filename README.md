# Исследование популярности альбомов Spotify

Анализ датасета Spotify Albums Data 2023: 438 973 трека, 78 172 альбома.
Цель — понять, какие факторы связаны с популярностью альбома.

## Работы
| Ноутбук | Что сделано | Отчёт |
|---|---|---|
| `lab3_missing_outliers.py` | Очистка данных, пропуски, выбросы | https://nikita-keltsin.github.io/spotify-albums-analysis/lab3_missing_outliers.html |
| `lab4_eda.py` | Разведочный анализ, гипотезы, признаки, корреляции | https://nikita-keltsin.github.io/spotify-albums-analysis/lab4_eda.html |

## Ключевые результаты
- Доля пропусков снижена с 44,6% до 0 при сохранении всех альбомов; случайность пропусков проверена критериями χ² и Манна–Уитни.
- Сравнено 6 методов поиска выбросов; после винсоризации асимметрия длительности снизилась с 24,1 до 1,8.
- Все 6 гипотез о факторах популярности подтвердились. Сильнейшие признаки из текста: мейджор-лейбл (r = 0,47), караоке и каверы (ρ = −0,42).
- Показана утечка целевой переменной при наивном target encoding: корреляция 0,85 против 0,67 при out-of-fold.

## Стек
Python, pandas, NumPy, SciPy, scikit-learn, matplotlib, seaborn, marimo

## Запуск
Положите CSV-файлы датасета в папку `data/`, затем:
    pip install marimo pandas numpy scipy scikit-learn matplotlib seaborn
    marimo edit lab4_eda.py