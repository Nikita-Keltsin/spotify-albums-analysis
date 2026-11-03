# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "pandas",
#     "scipy",
#     "matplotlib",
#     "seaborn",
#     "scikit-learn",
# ]
# ///
import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Лабораторная работа №3. Обработка пропусков и выбросов

    **Набор данных:** Spotify Albums Data 2023 (`spotify-albums_data_2023_1.csv`) — треки релизов Spotify вместе с характеристиками альбомов.

    Структура работы соответствует методическим указаниям:

    1. Первичный анализ набора данных
    2. Анализ пропущенных значений
    3. Выбор способов обработки пропущенных значений
    4. Сравнение способов заполнения пропусков
    5. Исследование распределений числовых признаков
    6. Обнаружение выбросов
    7. Обработка выбросов
    8. Сравнение исходного и обработанного наборов данных
    """)
    return


@app.cell
def _(mo):
    import time
    import warnings
    from enum import Enum

    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    from sklearn.impute import SimpleImputer, MissingIndicator, KNNImputer
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler

    warnings.filterwarnings("ignore")
    sns.set(style="ticks")
    pd.set_option("display.max_columns", 50)

    RANDOM_STATE = 42
    DATA_PATTERN = "spotify-albums_data_2023*.csv"  # подходит и ..._2023.csv, и ..._2023_1.csv

    # Ищем датасет рядом с ноутбуком и в папке data/ (относительно ноутбука и папки запуска)
    from pathlib import Path

    _nb_dir = Path(mo.notebook_dir() or Path.cwd())
    _search_dirs = [_nb_dir, _nb_dir / "data", Path.cwd(), Path.cwd() / "data"]
    _found = [p for d in _search_dirs for p in sorted(d.glob(DATA_PATTERN))]
    DATA_PATH = _found[0] if _found else None
    if DATA_PATH is None:
        raise FileNotFoundError(
            f"Не найден датасет {DATA_PATTERN}. Положите CSV рядом с ноутбуком или в папку data/. "
            "Проверены папки:\n" + "\n".join(str(d) for d in _search_dirs))


    def fmt_int(value):
        """Целое число с разделителем разрядов."""
        return f"{int(value):,}".replace(",", " ")

    return (
        DATA_PATH,
        Enum,
        IterativeImputer,
        KNNImputer,
        MissingIndicator,
        RANDOM_STATE,
        RandomForestRegressor,
        SimpleImputer,
        StandardScaler,
        fmt_int,
        np,
        pd,
        plt,
        sns,
        stats,
        time,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Первичный анализ набора данных

    ### 1.1 Загрузка данных
    """)
    return


@app.cell
def _(DATA_PATH, pd):
    df_raw = pd.read_csv(DATA_PATH, low_memory=False)
    df_raw.shape
    return (df_raw,)


@app.cell
def _(df_raw):
    df_raw.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.2 Описание набора данных
    """)
    return


@app.cell(hide_code=True)
def _(df_raw, fmt_int, mo, pd):
    _years = pd.to_datetime(df_raw["release_date"].str[:10], errors="coerce", format="%Y-%m-%d").dt.year
    mo.md(f"""
    **Предметная область.** Музыкальный стриминговый сервис Spotify. Каждая строка — один трек, вышедший в составе
    релиза (альбома, сингла или сборника). К треку присоединены характеристики релиза: тип, лейбл, дата выхода,
    популярность. Поэтому признаки делятся на два уровня: **уровень трека** и **уровень альбома**
    (одинаковые для всех треков одного `album_id`).

    | Характеристика | Значение |
    |---|---|
    | Количество объектов (строк = треков) | {fmt_int(df_raw.shape[0])} |
    | Количество признаков | {df_raw.shape[1]} |
    | Уникальных треков (`track_id`) | {fmt_int(df_raw['track_id'].nunique())} |
    | Уникальных альбомов (`album_id`) | {fmt_int(df_raw['album_id'].nunique())} |
    | Уникальных исполнителей альбомов (`artist_id`) | {fmt_int(df_raw['artist_id'].nunique())} |
    | Уникальных лейблов | {fmt_int(df_raw['label'].nunique())} |
    | Годы релизов (как записаны в данных) | {int(_years.min())} – {int(_years.max())} |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Назначение признаков**

    | Признак | Уровень | Описание |
    |---|---|---|
    | `track_name` | трек | название трека |
    | `track_id` | трек | идентификатор трека в Spotify |
    | `track_number` | трек | порядковый номер трека в релизе |
    | `duration_ms`, `duration_sec` | трек | длительность трека в миллисекундах / секундах |
    | `artist_0` … `artist_11` | трек | исполнители трека: `artist_0` — основной, остальные — приглашённые (feat.) |
    | `artists` | трек | список исполнителей (в выгрузке не заполнен) |
    | `album_type` | альбом | тип релиза: `album`, `single`, `compilation` |
    | `total_tracks` | альбом | количество треков в релизе |
    | `album_name` | альбом | название релиза |
    | `release_date` | альбом | дата выхода релиза |
    | `label` | альбом | лейбл (правообладатель) |
    | `album_popularity` | альбом | популярность релиза в Spotify по шкале 0–100 |
    | `album_id`, `artist_id` | альбом | идентификаторы релиза и его основного исполнителя |

    **Целевой признак.** Явной целевой переменной в наборе нет. Естественный кандидат для задачи регрессии —
    `album_popularity` (предсказание популярности релиза по его характеристикам). В этой работе он рассматривается
    как обычный числовой признак.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.3 Типы признаков
    """)
    return


@app.cell
def _(df_raw, pd):
    ALBUM_LEVEL_COLS = ["album_type", "total_tracks", "album_name", "release_date", "label",
                        "album_popularity", "album_id", "artist_id"]

    _kinds = {
        "track_name": "текстовый (название)",
        "track_id": "идентификатор",
        "track_number": "числовой дискретный (порядковый)",
        "duration_ms": "числовой непрерывный",
        "duration_sec": "числовой непрерывный",
        "album_type": "категориальный номинальный",
        "total_tracks": "числовой дискретный",
        "album_name": "текстовый (название)",
        "release_date": "дата (хранится строкой)",
        "label": "категориальный номинальный (высокая кардинальность)",
        "album_popularity": "числовой дискретный (шкала 0–100)",
        "album_id": "идентификатор",
        "artist_id": "идентификатор",
    }


    def _kind(col):
        if df_raw[col].isna().all():
            return "— (столбец полностью пуст)"
        if col.startswith("artist_") and col != "artist_id":
            return "категориальный номинальный (имя исполнителя)"
        return _kinds.get(col, "?")


    types_df = pd.DataFrame({
        "тип pandas": df_raw.dtypes.astype(str),
        "тип признака": [_kind(c) for c in df_raw.columns],
        "уровень": ["альбом" if c in ALBUM_LEVEL_COLS else "трек" for c in df_raw.columns],
        "уникальных": df_raw.nunique(),
        "пропусков (NaN)": df_raw.isnull().sum(),
        "пример значения": [df_raw[c].dropna().iloc[0] if df_raw[c].notna().any() else None
                            for c in df_raw.columns],
    })
    types_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Вывод по типам признаков.**

    - **Идентификаторы** (`track_id`, `album_id`, `artist_id`) и **текстовые названия** (`track_name`, `album_name`) — не являются признаками для моделей, но нужны для группировки и проверки согласованности.
    - **Числовые:** `track_number`, `total_tracks`, `album_popularity` — дискретные; `duration_ms` / `duration_sec` — непрерывные (это один и тот же признак в разных единицах).
    - **Категориальные:** `album_type` (3 категории), `label` (≈35 тыс. категорий — высокая кардинальность), `artist_0` … `artist_6` (имена исполнителей).
    - **Дата:** `release_date` хранится строкой вида `2015-06-09 00:00:00 UTC` — её нужно привести к типу `datetime` и выделить числовой признак `release_year`.
    - `artists`, `artist_7` … `artist_11` полностью пусты, поэтому pandas прочитал их как `float64`.
    - Признаки уровня альбома повторяются для всех треков релиза. Для их анализа данные нужно агрегировать по `album_id`, иначе альбомы с большим числом треков получат непропорционально большой вес.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.4 Проверка на некорректные и подозрительные значения

    Ищем значения, которые формально не являются `NaN`, но не могут быть корректными или являются «заглушками».
    """)
    return


@app.cell
def _(df_raw, np, pd):
    _dt_raw = pd.to_datetime(df_raw["release_date"].str[:10], errors="coerce", format="%Y-%m-%d")
    year_raw = _dt_raw.dt.year
    _str_cols = [c for c in df_raw.columns if not pd.api.types.is_numeric_dtype(df_raw[c])]
    PLACEHOLDERS = {"", "-", "--", "?", ".", "nan", "none", "null", "n/a", "na",
                    "unknown", "undefined", "no label", "[no label]"}

    placeholder_scan = {}
    for _c in _str_cols:
        _s = df_raw[_c].dropna().astype(str).str.strip().str.lower()
        _vc = _s[_s.isin(PLACEHOLDERS)].value_counts()
        if len(_vc):
            placeholder_scan[_c] = _vc.to_dict()

    _album_incons = {
        _c: int((df_raw.groupby("album_id")[_c].nunique(dropna=False) > 1).sum())
        for _c in ["album_type", "total_tracks", "album_name", "release_date", "label",
                   "album_popularity", "artist_id"]
    }
    _n = len(df_raw)
    _checks = [
        ("Полностью пустые столбцы",
         sum(df_raw[c].isna().all() for c in df_raw.columns),
         ", ".join(c for c in df_raw.columns if df_raw[c].isna().all())),
        ("duration_sec ≠ duration_ms / 1000",
         int((np.abs(df_raw["duration_ms"] / 1000 - df_raw["duration_sec"]) > 1e-6).sum()),
         "признаки полностью дублируют друг друга"),
        ("release_date: время не равно 00:00:00 UTC",
         int((~df_raw["release_date"].dropna().str.endswith("00:00:00 UTC")).sum()),
         "время не несёт информации"),
        ("release_date: дата = 1 января",
         int(((_dt_raw.dt.month == 1) & (_dt_raw.dt.day == 1)).sum()),
         "точность даты — только год (характерно для Spotify)"),
        ("release_date: год < 1920",
         int((year_raw < 1920).sum()),
         f"альбомов: {df_raw.loc[year_raw < 1920, 'album_id'].nunique()}"),
        ("label: заглушки ('-', 'unknown', 'no label', 'none')",
         sum(placeholder_scan.get("label", {}).values()),
         str(placeholder_scan.get("label", {}))),
        ("duration_sec = 0",
         int((df_raw["duration_sec"] <= 0).sum()),
         "трек нулевой длины невозможен"),
        ("duration_sec < 10 с",
         int((df_raw["duration_sec"] < 10).sum()),
         "формально возможно (сэмплы, интро)"),
        ("duration_sec > 30 мин",
         int((df_raw["duration_sec"] > 1800).sum()),
         "формально возможно (миксы, аудиокниги)"),
        ("track_number > total_tracks",
         int((df_raw["track_number"] > df_raw["total_tracks"]).sum()),
         "нарушение логической связи признаков"),
        ("album_popularity различается внутри одного альбома",
         _album_incons["album_popularity"],
         "альбомов; остальные признаки альбома согласованы: "
         + str({k: v for k, v in _album_incons.items() if k != "album_popularity"})),
        ("album_popularity = 0",
         int((df_raw["album_popularity"] == 0).sum()),
         "минимум шкалы, не пропуск (см. ниже)"),
        ("Повторяющиеся track_id",
         int(df_raw["track_id"].duplicated().sum()),
         f"в разных альбомах; полных дубликатов строк: {int(df_raw.duplicated().sum())}"),
    ]
    checks_df = pd.DataFrame(_checks, columns=["Проверка", "Строк", "Комментарий"])
    checks_df["Доля, %"] = (checks_df["Строк"] / _n * 100).round(4)
    checks_df.loc[0, "Доля, %"] = np.nan
    checks_df[["Проверка", "Строк", "Доля, %", "Комментарий"]]
    return placeholder_scan, year_raw


@app.cell
def _(df_raw, mo, pd, placeholder_scan, year_raw):
    mo.vstack([
        mo.md("**Найденные строковые заглушки по столбцам** (значение → количество):"),
        pd.DataFrame({"заглушки": {k: str(v) for k, v in placeholder_scan.items()}}),
        mo.md("**Релизы с годом выпуска ранее 1920:**"),
        df_raw.loc[year_raw < 1920, ["album_name", "artist_0", "label", "release_date", "album_popularity"]]
            .drop_duplicates("album_name"),
        mo.md("**Треки с нулевой длительностью:**"),
        df_raw.loc[df_raw["duration_sec"] <= 0, ["track_name", "artist_0", "album_name", "duration_ms"]],
        mo.md("**Треки с `track_number > total_tracks` (первые 10):**"),
        df_raw.loc[df_raw["track_number"] > df_raw["total_tracks"],
                   ["track_name", "album_name", "track_number", "total_tracks"]].head(10),
        mo.md("**Пример повторяющегося `track_id`:**"),
        df_raw[df_raw["track_id"].duplicated(keep=False)].sort_values("track_id")
            [["track_id", "track_name", "album_name", "album_type", "release_date"]].head(4),
    ])
    return


@app.cell
def _(df_raw, mo, pd):
    # Доля нулевой популярности по типам релиза и десятилетиям (на уровне альбомов)
    _alb = df_raw.drop_duplicates("album_id").assign(
        decade=(pd.to_datetime(df_raw["release_date"].str[:10], errors="coerce").dt.year // 10 * 10).astype("Int64"),
        zero_pop=lambda d: d["album_popularity"].eq(0),
    )
    mo.hstack([
        _alb.groupby("album_type")["zero_pop"].mean().rename("доля альбомов с popularity = 0").to_frame(),
        _alb.groupby("decade")["zero_pop"].agg(["mean", "size"])
            .rename(columns={"mean": "доля popularity = 0", "size": "альбомов"}).tail(8),
    ], justify="start", gap=2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Выводы по проверке значений.**

    1. Шесть столбцов (`artists`, `artist_7` – `artist_11`) полностью пусты — информации в них нет, они будут удалены.
    2. `duration_ms` и `duration_sec` полностью дублируют друг друга. В итоговом наборе остаётся `duration_sec`.
    3. `release_date` хранится строкой, время у всех записей одинаковое (`00:00:00 UTC`). Дату приводим к типу `datetime` и выделяем `release_year`. 11.7% дат приходятся на 1 января: для части релизов Spotify хранит только год. Это не ошибка, но точность таких дат — год.
    4. **Годы релиза ранее 1920 (101 строка, 6 альбомов) — скрытые пропуски.** `1899-12-30` — «нулевая» дата Excel / OLE Automation, `1900-01-01` — типичная заглушка. 1886 (электронная обработка музыки Сати) и 1901 (аудиокнига по Киплингу) — даты создания произведений, а не релиза. 1912 у сборника «Happy Birthday To Maxwell — In Many Styles» невозможен. Заменяем такие даты на `NaN`. Самые ранние из оставшихся дат (с 1929 года) — реальные исторические записи (Штраус, Билли Холидей).
    5. **Заглушки лейбла** (`-`, `Unknown`, `No Label`, `none` — 30 строк) — скрытые пропуски, заменяем на `NaN`. Похожие значения в `track_name`, `album_name`, `artist_*` (например, альбом «No Label» или трек «Unknown») могут быть реальными названиями, поэтому их не трогаем.
    6. **Нулевая длительность** (2 трека) невозможна → `NaN`. Это те же строки, у которых пусты название трека и исполнитель (см. п. 2.2).
    7. **`track_number > total_tracks`** (25 строк) нарушает логическую связь признаков. Исправляется в итоговом наборе приведением `track_number` к `total_tracks`.
    8. Популярность одного альбома различается между его треками (вероятно, данные собраны в разные моменты). Приводим её к единому среднему значению. Остальные признаки уровня альбома согласованы.
    9. **`album_popularity = 0` (19.7% треков) — не пропуск**, а нижняя граница шкалы Spotify. Доля нулей выше у синглов (36.5%) и у релизов 2010-х (43%), то есть у массовых малопопулярных загрузок. У релизов 1960–1990-х она составляет лишь 4–7%. Значение содержательное, оставляем его.
    10. Повторяющиеся `track_id` (25) — один и тот же трек в разных релизах (сингл и альбом, оригинал и сборник). Полных дубликатов строк нет, поэтому строки не удаляем.
    11. Очень короткие (< 10 с) и очень длинные (> 30 мин) треки формально возможны. Они анализируются в разделе выбросов.

    Ниже создаётся рабочая копия `df`, в которой скрытые пропуски заменены на `NaN`.
    """)
    return


@app.cell
def _(df_raw, np, pd):
    LABEL_PLACEHOLDERS = {"-", "unknown", "no label", "none"}
    MIN_VALID_YEAR = 1920

    df = df_raw.copy()

    # 1. Дата релиза: строка -> datetime; невозможные даты (заглушки) -> NaT
    _dt = pd.to_datetime(df["release_date"].str[:10], errors="coerce", format="%Y-%m-%d")
    _bad_year = _dt.dt.year < MIN_VALID_YEAR
    df["release_date"] = _dt.mask(_bad_year)
    df["release_year"] = df["release_date"].dt.year

    # 2. Заглушки в названии лейбла -> NaN
    _label_ph = df["label"].str.strip().str.lower().isin(LABEL_PLACEHOLDERS)
    df.loc[_label_ph, "label"] = np.nan

    # 3. Нулевая длительность -> NaN
    _zero_dur = df["duration_sec"] <= 0
    df.loc[_zero_dur, ["duration_ms", "duration_sec"]] = np.nan

    # 4. Популярность альбома должна быть одинаковой для всех его треков
    df["album_popularity"] = (df.groupby("album_id")["album_popularity"]
                                .transform("mean").round().astype(int))

    hidden_missing = pd.Series({
        "release_date: год < 1920 → NaT": int(_bad_year.sum()),
        "label: заглушка → NaN": int(_label_ph.sum()),
        "duration_sec = 0 → NaN": int(_zero_dur.sum()),
    }, name="строк")
    hidden_missing.to_frame()
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.5 Описательные статистики числовых признаков

    Статистики считаются по рабочей копии `df`, в которой скрытые пропуски уже заменены на `NaN` (иначе нулевые длительности и даты-заглушки исказили бы минимум и среднее).
    """)
    return


@app.cell
def _(df, pd):
    NUM_COLS = ["track_number", "duration_sec", "total_tracks", "album_popularity", "release_year"]


    def describe_numeric(frame, cols):
        """Основные описательные статистики числовых признаков."""
        data = frame[cols]
        return pd.DataFrame({
            "count": data.count(),
            "min": data.min(),
            "Q1": data.quantile(0.25),
            "median": data.median(),
            "mean": data.mean(),
            "Q3": data.quantile(0.75),
            "max": data.max(),
            "std": data.std(),
            "skew": data.skew(),
            "kurtosis": data.kurt(),
        })


    describe_numeric(df, NUM_COLS)
    return (describe_numeric,)


@app.cell
def _(describe_numeric, df, fmt_int, mo, np):
    # Таблица уровня альбома: одна строка = один релиз
    _artist_cols_07 = [f"artist_{i}" for i in range(7)]
    albums = (
        df.assign(
            n_artists_row=df[_artist_cols_07].notna().sum(axis=1),
            log_duration=np.log(df["duration_sec"]),
        )
        .groupby("album_id", sort=False)
        .agg(
            album_name=("album_name", "first"),
            album_type=("album_type", "first"),
            label=("label", "first"),
            release_year=("release_year", "first"),
            album_popularity=("album_popularity", "first"),
            total_tracks=("total_tracks", "first"),
            artist_id=("artist_id", "first"),
            artist_name=("artist_0", lambda s: s.mode().iat[0] if s.notna().any() else np.nan),
            n_rows=("track_id", "size"),
            mean_log_duration=("log_duration", "mean"),
            mean_n_artists=("n_artists_row", "mean"),
        )
    )
    mo.vstack([
        mo.md(f"**Статистики признаков уровня альбома** (по {fmt_int(len(albums))} уникальным релизам):"),
        describe_numeric(albums, ["total_tracks", "album_popularity", "release_year"]),
    ])
    return (albums,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Выводы по числовым признакам.**

    - `duration_sec`: медиана — 205 с (≈ 3.4 мин), межквартильный диапазон — 161–252 с. При этом минимум равен 0.18 с, а максимум — 17 919 с (≈ 5 ч). Асимметрия ≈ 24, эксцесс ≈ 2090: у распределения крайне тяжёлый правый хвост.
    - `total_tracks`: медиана на уровне треков — 13, на уровне альбомов — 1 (70% релизов — синглы). Максимум — 984, асимметрия по альбомам ≈ 21. Разница между уровнями показывает, что признаки альбома нужно анализировать по альбомам.
    - `track_number`: медиана — 6, максимум — 110. В Spotify нумерация идёт внутри диска, поэтому у корректных записей номер не превышает 53 даже в релизах из сотен треков. Значения больше `total_tracks` — ошибки (п. 1.4).
    - `album_popularity`: шкала 0–100, медиана — 8 по трекам и 3 по альбомам, среднее больше медианы (асимметрия 1.2–1.8). Большинство релизов малопопулярны.
    - `release_year`: 1929–2023 (после исправления заглушек), медиана — 2017 по трекам и 2020 по альбомам, асимметрия отрицательная (−2.3 / −3.2). В наборе преобладают новые релизы, старые записи образуют длинный левый хвост.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.6 Категориальные признаки
    """)
    return


@app.cell
def _(albums, df, pd):
    def describe_categorical(series):
        vc = series.value_counts()
        return {
            "уникальных": series.nunique(),
            "самая частая категория": vc.index[0],
            "её частота": int(vc.iloc[0]),
            "её доля, %": round(vc.iloc[0] / series.notna().sum() * 100, 3),
            "доля 10 самых частых, %": round(vc.head(10).sum() / series.notna().sum() * 100, 3),
        }


    cat_summary = pd.DataFrame({
        "album_type (треки)": describe_categorical(df["album_type"]),
        "album_type (альбомы)": describe_categorical(albums["album_type"]),
        "label (треки)": describe_categorical(df["label"]),
        "label (альбомы)": describe_categorical(albums["label"]),
        "artist_0 (треки)": describe_categorical(df["artist_0"]),
    }).T
    cat_summary
    return


@app.cell
def _(albums, df, mo, pd):
    def top_freq(series, top=10):
        vc = series.value_counts()
        return pd.DataFrame({"частота": vc.head(top),
                             "доля, %": (vc.head(top) / series.notna().sum() * 100).round(3)})


    mo.hstack([
        mo.vstack([mo.md("**album_type** (альбомы)"), top_freq(albums["album_type"])]),
        mo.vstack([mo.md("**label** (альбомы)"), top_freq(albums["label"])]),
        mo.vstack([mo.md("**artist_0** (треки)"), top_freq(df["artist_0"])]),
    ], justify="start", gap=2)
    return


@app.cell
def _(albums, df, pd, plt):
    _fig, _axes = plt.subplots(1, 3, figsize=(18, 5))
    _at = pd.DataFrame({
        "треки": df["album_type"].value_counts(normalize=True),
        "альбомы": albums["album_type"].value_counts(normalize=True),
    })
    _at.plot.bar(ax=_axes[0], rot=0)
    _axes[0].set_title("album_type: доли на уровне треков и альбомов")
    _axes[0].set_ylabel("доля")
    albums["label"].value_counts().head(15)[::-1].plot.barh(ax=_axes[1], color="tab:green")
    _axes[1].set_title("Топ-15 лейблов по числу альбомов")
    df["artist_0"].value_counts().head(15)[::-1].plot.barh(ax=_axes[2], color="tab:purple")
    _axes[2].set_title("Топ-15 исполнителей по числу треков")
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Выводы по категориальным признакам.**

    - `album_type` — 3 категории. На уровне треков преобладает `album` (70.1%), а на уровне релизов — `single` (70.3%): синглов много, но в каждом мало треков.
    - `label` — очень высокая кардинальность (34 928 лейблов) и «плоское» распределение. Самый частый лейбл по альбомам (`Independent`) занимает лишь 1.1%, десять самых частых вместе — 6.2%. Среди частых — караоке-лейблы и агрегаторы (Singer's Edge Karaoke, Stingray Music, Ameritz). Встречаются варианты написания одного лейбла (`Ameritz Music Ltd` / `2013 Ameritz Music Ltd`).
    - `artist_0` — 51 946 исполнителей. Самый частый — Иоганн Себастьян Бах (0.7% треков), за ним Johnny Cash и караоке-проекты.
    - У `label` и `artist_*` мода покрывает около 1% значений, поэтому заполнение модой для них заведомо искажает распределение (проверяется в разделе 4).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Анализ пропущенных значений

    ### 2.1 Распределение пропусков по признакам
    """)
    return


@app.cell
def _(df, df_raw, pd):
    def missing_table(frame):
        cnt = frame.isnull().sum()
        return pd.DataFrame({"пропусков": cnt, "доля, %": (cnt / len(frame) * 100)})


    _raw_m = missing_table(df_raw)
    _work_m = missing_table(df[df_raw.columns])
    missing_summary = pd.DataFrame({
        "NaN в исходных данных": _raw_m["пропусков"],
        "с учётом скрытых пропусков": _work_m["пропусков"],
        "доля, %": _work_m["доля, %"].round(4),
    })
    missing_summary = (missing_summary[missing_summary["с учётом скрытых пропусков"] > 0]
                       .sort_values("с учётом скрытых пропусков", ascending=False))
    cols_with_na = list(missing_summary.index)
    missing_summary
    return cols_with_na, missing_summary


@app.cell
def _(fmt_int, missing_summary, plt):
    _share = missing_summary["доля, %"].sort_values()
    _colors = ["lightgray" if v >= 100 else ("tab:orange" if c in [f"artist_{i}" for i in range(1, 7)] else "tab:red")
               for c, v in _share.items()]
    _fig, _ax = plt.subplots(figsize=(11, 7))
    _ax.barh(_share.index, _share.values, color=_colors)
    _ax.set_xscale("log")
    _ax.set_xlabel("доля пропусков, % (логарифмическая шкала)")
    _ax.set_title("Доли пропусков по признакам")
    for _i, (_c, _v) in enumerate(_share.items()):
        _ax.text(_v * 1.1, _i, f"{_v:.3g}% ({fmt_int(missing_summary.loc[_c, 'с учётом скрытых пропусков'])})",
                 va="center", fontsize=9)
    _ax.set_xlim(right=_share.max() * 8)
    from matplotlib.patches import Patch
    _ax.legend(handles=[Patch(color="lightgray", label="столбец пуст полностью"),
                        Patch(color="tab:orange", label="приглашённые исполнители (структурные пропуски)"),
                        Patch(color="tab:red", label="прочие признаки")], loc="lower right")
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.2 Распределение пропусков по объектам
    """)
    return


@app.cell
def _(RANDOM_STATE, cols_with_na, df, plt, sns):
    _fig, _axes = plt.subplots(1, 2, figsize=(18, 6), gridspec_kw={"width_ratios": [1.3, 1]})
    # (а) случайная выборка строк: структура пропусков в столбцах исполнителей
    _sample = df[cols_with_na].sample(1500, random_state=RANDOM_STATE)
    _sample = _sample.loc[_sample.isnull().sum(axis=1).sort_values().index]
    sns.heatmap(_sample.isnull(), cbar=False, cmap="viridis", ax=_axes[0], yticklabels=False)
    _axes[0].set_title("Матрица пропусков: случайные 1500 строк (жёлтый = пропуск)")
    _axes[0].set_ylabel("объекты (отсортированы по числу пропусков)")
    # (б) все строки с пропусками в «редких» признаках
    _rare = ["track_name", "album_name", "artist_0", "duration_sec", "label", "release_date"]
    _rare_rows = df.loc[df[_rare].isnull().any(axis=1), _rare + ["album_id"]]
    _rare_rows = _rare_rows.sort_values(["label", "release_date", "album_id"], na_position="first")
    sns.heatmap(_rare_rows[_rare].isnull(), cbar=False, cmap="viridis", ax=_axes[1], yticklabels=False)
    _axes[1].set_title(f"Все {len(_rare_rows)} строк с пропусками в редко пропускаемых признаках")
    plt.tight_layout()
    _fig
    return


@app.cell
def _(df, mo, pd):
    artist_cols = [f"artist_{i}" for i in range(12)]
    _monotone = pd.Series({
        f"artist_{k} пуст, а artist_{k + 1} заполнен": int((df[f"artist_{k}"].isna() & df[f"artist_{k + 1}"].notna()).sum())
        for k in range(6)
    }, name="строк")
    _n_art = df[artist_cols].notna().sum(axis=1).value_counts().sort_index()
    _n_art = pd.DataFrame({"треков": _n_art, "доля, %": (_n_art / len(df) * 100).round(3)})
    _n_art.index.name = "заполнено столбцов artist_*"

    _rare_cols = ["track_name", "album_name", "artist_0", "duration_sec", "label", "release_date"]
    row_missing = df[_rare_cols].isnull().sum(axis=1)
    _row_dist = row_missing.value_counts().sort_index().rename("строк").to_frame()
    _row_dist.index.name = "пропусков в строке (без artist_1…11, artists)"

    mo.vstack([
        mo.hstack([
            mo.vstack([mo.md("**Проверка монотонности пропусков исполнителей**"), _monotone.to_frame()]),
            mo.vstack([mo.md("**Число исполнителей у трека**"), _n_art]),
            mo.vstack([mo.md("**Число пропусков в строке**"), _row_dist]),
        ], justify="start", gap=2),
        mo.md("**Строки с ≥ 2 пропусками:**"),
        df.loc[row_missing >= 2, ["track_id", "track_name", "artist_0", "duration_sec", "album_name",
                                  "label", "release_date", "track_number", "total_tracks"]],
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.3 Случайны ли пропуски?

    Пропуски в `label` и `release_date` относятся к уровню альбома, поэтому проверяем их на таблице альбомов. Сравниваем альбомы с пропуском и без него по типу релиза, популярности, числу треков, а также смотрим, сконцентрированы ли пропуски у отдельных исполнителей.
    """)
    return


@app.cell
def _(albums, pd, stats):
    def missing_vs_present(alb, col):
        miss = alb[col].isna()
        by_artist = alb.groupby("artist_id")[col].apply(lambda s: s.isna().mean())
        artists_with_miss = by_artist[by_artist > 0]
        _chi2_p = stats.chi2_contingency(pd.crosstab(miss, alb["album_type"]))[1]
        _mw_pop = stats.mannwhitneyu(alb.loc[miss, "album_popularity"], alb.loc[~miss, "album_popularity"]).pvalue
        _mw_tt = stats.mannwhitneyu(alb.loc[miss, "total_tracks"], alb.loc[~miss, "total_tracks"]).pvalue
        return {
            "альбомов с пропуском": int(miss.sum()),
            "доля пропусков, %": round(miss.mean() * 100, 4),
            "доля синглов: с пропуском / без, %":
                f"{(alb.loc[miss, 'album_type'] == 'single').mean() * 100:.1f} / "
                f"{(alb.loc[~miss, 'album_type'] == 'single').mean() * 100:.1f}",
            "медиана популярности: с пропуском / без":
                f"{alb.loc[miss, 'album_popularity'].median():.0f} / {alb.loc[~miss, 'album_popularity'].median():.0f}",
            "медиана total_tracks: с пропуском / без":
                f"{alb.loc[miss, 'total_tracks'].median():.0f} / {alb.loc[~miss, 'total_tracks'].median():.0f}",
            "p-value χ² (album_type)": round(_chi2_p, 5),
            "p-value Манна–Уитни (popularity)": round(_mw_pop, 5),
            "p-value Манна–Уитни (total_tracks)": round(_mw_tt, 5),
            "исполнителей, у которых есть пропуск": len(artists_with_miss),
            "средняя доля пропусков у этих исполнителей, %": round(artists_with_miss.mean() * 100, 1),
        }


    mar_df = pd.DataFrame({
        "label": missing_vs_present(albums, "label"),
        "release_year": missing_vs_present(albums, "release_year"),
    })
    mar_df
    return


@app.cell
def _(albums, mo):
    mo.vstack([
        mo.md("**Альбомы с пропущенным лейблом:**"),
        albums.loc[albums["label"].isna(), ["album_name", "artist_name", "album_type", "album_popularity",
                                             "total_tracks", "release_year"]],
        mo.md("**Альбомы с пропущенным (или невозможным) годом релиза:**"),
        albums.loc[albums["release_year"].isna(), ["album_name", "artist_name", "album_type", "label",
                                                    "album_popularity", "total_tracks"]],
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Выводы по анализу пропусков.**

    По признакам пропуски делятся на три группы:

    1. **Полностью пустые столбцы** (`artists`, `artist_7` – `artist_11`, 100%) — дефект выгрузки, восстанавливать нечего.
    2. **Структурные пропуски** в `artist_1` – `artist_6` (77–99%). Нарушений монотонности нет: если `artist_k` пуст, то пусты и все следующие столбцы. Пропуск означает «у трека нет k-го исполнителя», то есть сам несёт информацию. Это **неслучайный пропуск (MNAR)**. 77% треков записаны одним исполнителем.
    3. **Редкие пропуски** (< 0.03%): `release_date` (124 строки с учётом заглушек), `label` (82), `track_name` (5), `artist_0` (3), `duration_sec` (2), `album_name` (1).

    **По объектам:** у 438 761 строки среди редко пропускаемых признаков нет ни одного пропуска, у 209 строк — один, у одной — два (сингл без названия трека и альбома), у двух — три. Последние две — «битые» записи: нет названия трека и исполнителя, длительность равна нулю.

    **Характер пропусков:**

    - `label`: пропуски сосредоточены у 22 исполнителей, у которых в среднем 83% релизов без лейбла (например, все 11 синглов Kaizen). Они чаще встречаются у синглов (90.6% против 70.3%, p = 0.04) и у релизов с малым числом треков (p = 0.003); популярность не отличается (p = 0.73). Пропуски **не являются полностью случайными (не MCAR)**: они зависят от исполнителя (дистрибьютора) и типа релиза, то есть от наблюдаемых признаков (**MAR**). Отсюда выбор заполнения по исполнителю.
    - `release_year`: 9 альбомов, у 6 из них дата-заглушка. Почти все — релизы с нулевой популярностью (медиана 0 против 3, p = 0.01), чаще полноформатные альбомы (p = 0.026) с большим числом треков (p = 0.004). Заглушки связаны с конкретными источниками (например, у лейбла `Canaria` — `1899-12-30`). Пропуски **систематические (не MCAR)** и отражают качество метаданных источника.
    - `track_name`, `album_name`, `artist_0`, `duration_sec` — единичные случаи, заметную часть которых дают две «битые» записи.

    Групп с пропусками мало, поэтому p-value носят ориентировочный характер. Однако концентрация пропусков у отдельных исполнителей выражена очень сильно.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Выбор способов обработки пропущенных значений

    | Признак | Пропусков | Способ обработки | Обоснование и преимущества | Возможные искажения |
    |---|---|---|---|---|
    | `artists`, `artist_7` – `artist_11` | 100% | удаление столбцов | данных нет, восстанавливать нечего | нет |
    | `artist_1` – `artist_6` | 77–99% (структурные) | замена производными признаками `n_artists` (число исполнителей) и `featured_artists` (приглашённые через `; `, пустая строка, если их нет); исходные столбцы удаляются | пропуск означает «исполнителя нет», и эта информация сохраняется; вместо 6 разреженных столбцов — 2 плотных | нет потери информации; позиция исполнителя (2-й, 3-й…) сохраняется только порядком в строке `featured_artists` |
    | 2 «битые» строки (нет `track_name`, `artist_0`, `duration_sec`) | 2 строки | удаление строк | запись восстановить нельзя, доля — 0.0005% | потеря 2 треков; распределения не меняются |
    | `track_name`, `album_name`, `artist_0` (остальные) | 3 / 1 / 1 | отдельная категория `Unknown` | текст нельзя восстановить статистически, а строка сохраняет числовую информацию | нет: поля не используются как числовые признаки |
    | `label` | 82 строки (32 альбома) | мода лейбла **того же исполнителя**, иначе — отдельная категория `Unknown`; флаг `label_imputed` | пропуски зависят от исполнителя (MAR), а исполнитель обычно выпускается на одном лейбле; `Unknown` не вносит ложной информации (сравнение — п. 4.2) | если исполнитель сменил лейбл, значение может оказаться неверным; появляется новая категория `Unknown` |
    | `release_date` / `release_year` | 124 строки (9 альбомов) | медиана года релизов **того же исполнителя**, иначе — MissForest по признакам альбома; дата — 1 января найденного года; флаг `release_year_imputed` | лучший результат в имитационном эксперименте (п. 4.1); учитывает и исполнителя, и другие признаки альбома | для исполнителей с одним релизом ошибка составляет несколько лет; разброс заполненных значений меньше истинного |

    Удаление строк с пропущенными `label` и `release_date` тоже допустимо по объёму (< 0.03%). Однако пропуски неслучайны, и удаление выбросило бы релизы конкретных исполнителей и источников, поэтому выбрано заполнение.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Сравнение способов заполнения пропусков

    Для сравнения выбраны:
    - **числовой признак** `release_year` (уровень альбома);
    - **категориальный признак** `label` (уровень альбома);
    - дополнительно **категориальный признак со структурными пропусками** `artist_1`.

    Используются функции из лекции (`impute_column` на основе `SimpleImputer` и `MissingIndicator`), а также многомерные методы `KNNImputer` и `IterativeImputer` (MissForest) и групповое заполнение.
    """)
    return


@app.cell
def _(MissingIndicator, SimpleImputer, np, pd):
    def impute_column(dataset, column, strategy_param, fill_value_param=None):
        """Заполнение пропусков в одном признаке (функция из лекции)."""
        temp_data = dataset[[column]].to_numpy()
        if not np.issubdtype(temp_data.dtype, np.number):
            temp_data = temp_data.astype(object)
        size = temp_data.shape[0]
        indicator = MissingIndicator(features="all")
        mask_missing_values_only = indicator.fit_transform(temp_data)
        imputer = SimpleImputer(strategy=strategy_param, fill_value=fill_value_param)
        all_data = imputer.fit_transform(temp_data)
        missed_data = temp_data[mask_missing_values_only]
        filled_data = all_data[mask_missing_values_only]
        return all_data.reshape((size,)), filled_data, missed_data


    def numeric_summary(series_dict):
        """Характеристики распределения для набора вариантов одного признака."""
        return pd.DataFrame({
            name: {
                "count": s.count(), "mean": s.mean(), "median": s.median(), "std": s.std(),
                "Q1": s.quantile(0.25), "Q3": s.quantile(0.75), "skew": s.skew(), "kurtosis": s.kurt(),
            }
            for name, s in series_dict.items()
        }).T

    return impute_column, numeric_summary


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.1 Числовой признак `release_year`

    #### 4.1.1 Заполнение реальных пропусков показателями центра распределения и «хвостом»

    Распределение года релиза скошено влево (длинный хвост в сторону старых записей), поэтому «хвост распределения» берётся слева: $extreme\_value = Q1 - 3 \cdot IQR$.
    """)
    return


@app.cell
def _(albums, fmt_int, impute_column, mo, numeric_summary, pd):
    _q1, _q3 = albums["release_year"].quantile([0.25, 0.75])
    YEAR_TAIL_VALUE = _q1 - 3 * (_q3 - _q1)

    _year_variants = {"Исходные данные (без пропусков)": albums["release_year"].dropna()}
    _filled_values = {}
    for _name, _strategy, _fill in [("Среднее", "mean", None), ("Медиана", "median", None),
                                    ("Мода", "most_frequent", None),
                                    (f"Константа = Q1 − 3·IQR = {YEAR_TAIL_VALUE:.0f}", "constant", YEAR_TAIL_VALUE)]:
        _all, _filled, _ = impute_column(albums, "release_year", _strategy, _fill)
        _year_variants[_name] = pd.Series(_all)
        _filled_values[_name] = _filled[0]

    mo.vstack([
        mo.md(f"Пропусков в `release_year`: **{albums['release_year'].isna().sum()}** альбомов из {fmt_int(len(albums))}. "
              "Подставляемые значения: " + ", ".join(f"{k} → **{v:.1f}**" for k, v in _filled_values.items())),
        numeric_summary(_year_variants),
    ])
    return (YEAR_TAIL_VALUE,)


@app.cell
def _(YEAR_TAIL_VALUE, albums, impute_column, pd, plt, sns):
    def research_impute_numeric_column(dataset, num_column, const_value=None, ax=None):
        """KDE исходного признака и вариантов заполнения (функция из лекции)."""
        strategy_params = ["mean", "median", "most_frequent", "constant"]
        strategy_params_names = ["Среднее", "Медиана", "Мода", f"Константа = {const_value}"]
        new_df = pd.DataFrame({"Исходные данные": dataset[num_column].to_numpy()})
        for strategy, col_name in zip(strategy_params, strategy_params_names):
            if strategy == "constant" and const_value is None:
                continue
            new_df[col_name], _, _ = impute_column(dataset, num_column, strategy,
                                                   fill_value_param=const_value if strategy == "constant" else None)
        return sns.kdeplot(data=new_df, ax=ax, bw_adjust=1.5, common_norm=False)


    _fig, _axes = plt.subplots(1, 2, figsize=(18, 5))
    research_impute_numeric_column(albums, "release_year", round(YEAR_TAIL_VALUE), ax=_axes[0])
    _axes[0].set_title("release_year: все альбомы")
    research_impute_numeric_column(albums, "release_year", round(YEAR_TAIL_VALUE), ax=_axes[1])
    _axes[1].set_xlim(1975, 2024)
    _axes[1].set_title("то же, приближение к 1975–2024")
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Реальных пропусков в `release_year` очень мало, поэтому на графиках кривые совпадают и качество разных стратегий по ним сравнить нельзя.
    Поэтому проведём **имитационный эксперимент**: у 5% альбомов с известным годом искусственно удалим год (MCAR-маска на уровне альбома), заполним пропуски разными способами и сравним с истинными значениями.

    Дополнительно к одномерным стратегиям используются:
    - **KNNImputer** (k = 10, веса по расстоянию; признаки предварительно стандартизованы);
    - **MissForest** — `IterativeImputer` со случайным лесом;
    - **медиана года по исполнителю** (`artist_id`) — групповое заполнение: релизы одного исполнителя обычно близки по времени. Если у исполнителя нет других альбомов с известным годом, используется запасной метод.

    Признаки для многомерных методов (уровень альбома): популярность, $\ln$(число треков), средний $\ln$(длительности трека), среднее число исполнителей на трек, $\ln$(размера лейбла), тип релиза (one-hot).
    """)
    return


@app.cell
def _(
    IterativeImputer,
    KNNImputer,
    RANDOM_STATE,
    RandomForestRegressor,
    StandardScaler,
    np,
    pd,
):
    def album_feature_matrix(alb, target_col):
        """Матрица признаков уровня альбома для многомерной импьютации."""
        X = pd.DataFrame(index=alb.index)
        X[target_col] = alb[target_col]
        X["album_popularity"] = alb["album_popularity"]
        X["log_total_tracks"] = np.log(alb["total_tracks"])
        X["mean_log_duration"] = alb["mean_log_duration"]
        X["mean_n_artists"] = alb["mean_n_artists"]
        X["log_label_size"] = np.log(alb["label"].map(alb["label"].value_counts()).fillna(1))
        for t in ["album", "single", "compilation"]:
            X[f"type_{t}"] = (alb["album_type"] == t).astype(float)
        return X


    def knn_impute(X, n_neighbors=10):
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        Xi = KNNImputer(n_neighbors=n_neighbors, weights="distance").fit_transform(Xs)
        return pd.DataFrame(scaler.inverse_transform(Xi), index=X.index, columns=X.columns)


    def missforest_impute(X):
        imputer = IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=20, min_samples_leaf=5, n_jobs=-1,
                                            random_state=RANDOM_STATE),
            max_iter=3, random_state=RANDOM_STATE)
        return pd.DataFrame(imputer.fit_transform(X), index=X.index, columns=X.columns)


    def group_median_impute(values, groups):
        """Медиана по группе (например, по исполнителю); NaN, если в группе нет известных значений."""
        med = values.groupby(groups).median()
        return groups.map(med).where(values.isna(), values)

    return (
        album_feature_matrix,
        group_median_impute,
        knn_impute,
        missforest_impute,
    )


@app.cell
def _(
    RANDOM_STATE,
    YEAR_TAIL_VALUE,
    album_feature_matrix,
    albums,
    fmt_int,
    group_median_impute,
    impute_column,
    knn_impute,
    missforest_impute,
    mo,
    np,
    pd,
    stats,
    time,
):
    SIM_SHARE = 0.05
    year_known_albums = albums[albums["release_year"].notna()]
    _rng = np.random.default_rng(RANDOM_STATE)
    year_sim_mask = pd.Series(_rng.random(len(year_known_albums)) < SIM_SHARE, index=year_known_albums.index)
    year_true = year_known_albums.loc[year_sim_mask, "release_year"]

    _X = album_feature_matrix(year_known_albums, "release_year")
    _X.loc[year_sim_mask, "release_year"] = np.nan

    year_sim_preds, _times = {}, {}
    for _name, _strategy, _fill in [("Среднее", "mean", None), ("Медиана", "median", None),
                                    ("Мода", "most_frequent", None),
                                    ("Хвост Q1−3·IQR", "constant", YEAR_TAIL_VALUE)]:
        _t = time.time()
        _all, _, _ = impute_column(_X, "release_year", _strategy, _fill)
        year_sim_preds[_name] = pd.Series(_all, index=_X.index)[year_sim_mask]
        _times[_name] = time.time() - _t

    _t = time.time()
    year_sim_preds["KNN (k=10)"] = knn_impute(_X)["release_year"][year_sim_mask]
    _times["KNN (k=10)"] = time.time() - _t

    _t = time.time()
    year_sim_preds["MissForest"] = missforest_impute(_X)["release_year"][year_sim_mask]
    _times["MissForest"] = time.time() - _t

    _t = time.time()
    _by_artist = group_median_impute(_X["release_year"], year_known_albums["artist_id"])[year_sim_mask]
    _times["Медиана по исполнителю"] = time.time() - _t
    year_artist_coverage = _by_artist.notna().mean()
    year_sim_preds["Исполнитель → медиана"] = _by_artist.fillna(_X["release_year"].median())
    year_sim_preds["Исполнитель → KNN"] = _by_artist.fillna(year_sim_preds["KNN (k=10)"])
    year_sim_preds["Исполнитель → MissForest"] = _by_artist.fillna(year_sim_preds["MissForest"])
    _times["Исполнитель → медиана"] = _times.pop("Медиана по исполнителю") + _times["Медиана"]
    _times["Исполнитель → KNN"] = _times["Исполнитель → медиана"] + _times["KNN (k=10)"]
    _times["Исполнитель → MissForest"] = _times["Исполнитель → медиана"] + _times["MissForest"]
    year_sim_preds = {k: v.round() for k, v in year_sim_preds.items()}


    def _metrics(pred):
        err = pred - year_true
        return {
            "MAE, лет": err.abs().mean(),
            "RMSE, лет": np.sqrt((err ** 2).mean()),
            "медиана |ошибки|": err.abs().median(),
            "точно угадан год, %": (err == 0).mean() * 100,
            "смещение (bias)": err.mean(),
            "std заполненных / std истинных": pred.std() / year_true.std(),
            "KS-статистика (заполн. vs истинные)": stats.ks_2samp(pred, year_true).statistic,
        }


    year_sim_metrics = pd.DataFrame({k: _metrics(v) for k, v in year_sim_preds.items()}).T
    year_sim_metrics["время, с"] = pd.Series(_times)
    year_sim_metrics = year_sim_metrics.sort_values("MAE, лет")
    mo.vstack([
        mo.md(f"Скрыто **{fmt_int(year_sim_mask.sum())}** значений из {fmt_int(len(year_known_albums))}. "
              f"Медиана по исполнителю применима к **{year_artist_coverage:.1%}** скрытых альбомов "
              "(у остальных нет других релизов с известным годом)."),
        year_sim_metrics.round(3),
    ])
    return (
        year_artist_coverage,
        year_known_albums,
        year_sim_mask,
        year_sim_metrics,
        year_sim_preds,
        year_true,
    )


@app.cell
def _(np, plt, year_sim_metrics, year_sim_preds, year_true):
    _order = list(year_sim_metrics.index)
    _fig, _axes = plt.subplots(2, 5, figsize=(22, 8), sharex=True, sharey=True)
    _bins = np.arange(1950, 2025, 1)
    for _ax, _name in zip(_axes.flat, _order):
        _ax.hist(year_true, bins=_bins, density=True, alpha=0.45, color="gray", label="истинные")
        _ax.hist(year_sim_preds[_name].clip(1950, 2024), bins=_bins, density=True, histtype="step",
                 lw=1.6, color="tab:red", label="заполненные")
        _ax.set_title(f"{_name}\nMAE = {year_sim_metrics.loc[_name, 'MAE, лет']:.2f}", fontsize=10)
        _ax.set_ylim(0, 0.2)
    for _ax in _axes.flat[len(_order):]:
        _ax.axis("off")
    _axes.flat[0].legend()
    _fig.suptitle("Имитационный эксперимент: распределение заполненных значений release_year и истинных значений")
    plt.tight_layout()
    _fig
    return


@app.cell
def _(pd, plt, sns, year_known_albums, year_sim_mask, year_sim_preds):
    # Влияние на распределение признака целиком: KDE до/после для нескольких стратегий
    _full_true = year_known_albums["release_year"]
    _kde_df = pd.DataFrame({"Истинные (без маски)": _full_true.to_numpy()})
    for _name in ["Медиана", "Хвост Q1−3·IQR", "KNN (k=10)", "Исполнитель → MissForest"]:
        _kde_df[_name] = _full_true.where(~year_sim_mask, year_sim_preds[_name]).to_numpy()
    _fig, _ax = plt.subplots(figsize=(14, 5))
    sns.kdeplot(data=_kde_df, ax=_ax, bw_adjust=1.5)
    _ax.set_xlim(1960, 2025)
    _ax.set_title("release_year после заполнения 5% искусственных пропусков")
    _fig
    return


@app.cell(hide_code=True)
def _(
    YEAR_TAIL_VALUE,
    albums,
    fmt_int,
    mo,
    year_artist_coverage,
    year_sim_mask,
    year_sim_metrics,
):
    _m = year_sim_metrics
    _uni = ["Среднее", "Медиана", "Мода", "Хвост Q1−3·IQR"]
    mo.md(f"""
    **Выводы по `release_year`.**

    - На реальных пропусках (9 альбомов из {fmt_int(len(albums))}) любая стратегия почти не меняет характеристики признака: среднее и std меняются в третьем знаке после запятой. Сильнее остальных сдвигает среднее и асимметрию заполнение «хвостом» ({YEAR_TAIL_VALUE:.0f}), но и этот сдвиг пренебрежимо мал.
    - Имитационный эксперимент ({fmt_int(year_sim_mask.sum())} скрытых значений) показывает реальное различие методов:
        - **Одномерные стратегии** подставляют одно и то же значение (std заполненных = 0), и в распределении появляется искусственный пик. MAE: медиана — {_m.loc['Медиана', 'MAE, лет']:.2f} года, среднее — {_m.loc['Среднее', 'MAE, лет']:.2f}, мода — {_m.loc['Мода', 'MAE, лет']:.2f} при смещении {_m.loc['Мода', 'смещение (bias)']:+.1f} года, «хвост» — {_m.loc['Хвост Q1−3·IQR', 'MAE, лет']:.1f}. Заполнение «хвостом» полезно только как маркер неслучайного пропуска, но не как оценка значения.
        - **Многомерные методы** точнее и лучше сохраняют форму распределения: MissForest — MAE {_m.loc['MissForest', 'MAE, лет']:.2f}, KNN — {_m.loc['KNN (k=10)', 'MAE, лет']:.2f}; KS-статистика {_m.loc['MissForest', 'KS-статистика (заполн. vs истинные)']:.2f} и {_m.loc['KNN (k=10)', 'KS-статистика (заполн. vs истинные)']:.2f} против {_m.loc[_uni, 'KS-статистика (заполн. vs истинные)'].min():.2f}–{_m.loc[_uni, 'KS-статистика (заполн. vs истинные)'].max():.2f} у одномерных.
        - **Медиана по исполнителю** применима к {year_artist_coverage:.1%} пропусков. В комбинации с многомерным методом для остальных она даёт лучший результат: **«{_m.index[0]}»** — MAE {_m.iloc[0]['MAE, лет']:.2f} года, медианная ошибка {_m.iloc[0]['медиана |ошибки|']:.0f} год, точно угадано {_m.iloc[0]['точно угадан год, %']:.1f}% значений.
    - Общий недостаток всех методов — заниженный разброс заполненных значений (отношение std {_m.loc[['KNN (k=10)', 'MissForest', 'Исполнитель → MissForest'], 'std заполненных / std истинных'].min():.2f}–{_m.loc[['KNN (k=10)', 'MissForest', 'Исполнитель → MissForest'], 'std заполненных / std истинных'].max():.2f} у лучших): оценки «стягиваются» к типичным значениям.
    - **Выбор для итогового набора: медиана по исполнителю, для остальных — MissForest.**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.2 Категориальный признак `label`

    #### 4.2.1 Реальные пропуски

    Сравниваем три стратегии: заполнение **модой** (самым частым лейблом), **отдельной категорией** `Unknown` и **модой лейбла у того же исполнителя** (с запасным `Unknown`).
    """)
    return


@app.cell
def _(albums, impute_column, mo, pd):
    def group_mode_impute(values, groups):
        """Самое частое значение в группе; NaN, если известных значений в группе нет."""
        known = pd.DataFrame({"g": groups, "v": values}).dropna()
        modes = known.groupby("g")["v"].agg(lambda s: s.value_counts().index[0])
        return groups.map(modes).where(values.isna(), values)


    _label_mode, _, _ = impute_column(albums, "label", "most_frequent")
    _label_const, _, _ = impute_column(albums, "label", "constant", "Unknown")
    _label_artist = group_mode_impute(albums["label"], albums["artist_id"])
    label_variants = {
        "Мода": pd.Series(_label_mode, index=albums.index),
        "Константа Unknown": pd.Series(_label_const, index=albums.index),
        "Мода по исполнителю → Unknown": _label_artist.fillna("Unknown"),
    }
    _miss = albums["label"].isna()
    _cmp = albums.loc[_miss, ["album_name", "artist_name"]].copy()
    for _k, _v in label_variants.items():
        _cmp[_k] = _v[_miss]
    mo.vstack([
        mo.md(f"Пропусков в `label`: **{_miss.sum()}** альбомов. Заполненные значения:"),
        _cmp,
    ])
    return group_mode_impute, label_variants


@app.cell
def _(albums, label_variants, pd):
    _freq = pd.DataFrame({"Исходные (без пропусков)": albums["label"].value_counts(normalize=True)})
    for _k, _v in label_variants.items():
        _freq[_k] = _v.value_counts(normalize=True)
    _top = _freq.sort_values("Исходные (без пропусков)", ascending=False).head(8)
    (_top * 100).round(4).rename(columns=lambda c: c + ", %")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 4.2.2 Имитационный эксперимент

    Скрываем лейбл у 10% альбомов и сравниваем стратегии по доле верно восстановленных значений и по тому, как меняются частоты категорий.
    """)
    return


@app.cell
def _(
    RANDOM_STATE,
    albums,
    fmt_int,
    group_mode_impute,
    impute_column,
    mo,
    np,
    pd,
):
    _known_lab = albums[albums["label"].notna()]
    _rng2 = np.random.default_rng(RANDOM_STATE + 1)
    label_sim_mask = pd.Series(_rng2.random(len(_known_lab)) < 0.10, index=_known_lab.index)
    _lab_masked = _known_lab["label"].mask(label_sim_mask)
    _lab_df = _known_lab.assign(label=_lab_masked)

    _mode_all, _, _ = impute_column(_lab_df, "label", "most_frequent")
    _by_artist_lab = group_mode_impute(_lab_masked, _known_lab["artist_id"])
    label_artist_coverage = _by_artist_lab[label_sim_mask].notna().mean()
    label_mode_value = _lab_masked.mode().iat[0]
    label_sim_variants = {
        "Мода": pd.Series(_mode_all, index=_known_lab.index),
        "Константа Unknown": _lab_masked.fillna("Unknown"),
        "Мода по исполнителю → Unknown": _by_artist_lab.fillna("Unknown"),
        "Мода по исполнителю → мода": _by_artist_lab.fillna(label_mode_value),
    }
    _true_lab = _known_lab["label"]
    label_true_freq = _true_lab.value_counts(normalize=True)


    def _label_metrics(filled):
        pred = filled[label_sim_mask]
        freq = filled.value_counts(normalize=True)
        tvd = 0.5 * label_true_freq.subtract(freq, fill_value=0).abs().sum()
        return {
            "точность восстановления, %": (pred == _true_lab[label_sim_mask]).mean() * 100,
            f"доля «{label_mode_value}» после заполнения, %": (filled == label_mode_value).mean() * 100,
            "доля самой частой категории, %": freq.iloc[0] * 100,
            "уникальных категорий": filled.nunique(),
            "TVD с истинным распределением": tvd,
        }


    label_sim_metrics = pd.DataFrame({k: _label_metrics(v) for k, v in label_sim_variants.items()}).T
    mo.vstack([
        mo.md(f"Скрыто **{fmt_int(label_sim_mask.sum())}** лейблов из {fmt_int(len(_known_lab))}. "
              f"Истинная доля «{label_mode_value}»: {label_true_freq.iloc[0] * 100:.3f}%. "
              f"Мода по исполнителю применима к **{label_artist_coverage:.1%}** скрытых значений. "
              "TVD — расстояние полной вариации между частотами категорий (0 = распределения совпадают)."),
        label_sim_metrics.round(4),
    ])
    return (
        label_artist_coverage,
        label_mode_value,
        label_sim_metrics,
        label_sim_variants,
        label_true_freq,
    )


@app.cell
def _(label_sim_variants, label_true_freq, pd, plt):
    _cats = list(label_true_freq.head(10).index) + ["Unknown"]
    _plot_df = pd.DataFrame({"Истинные": label_true_freq.reindex(_cats).fillna(0)})
    for _k, _v in label_sim_variants.items():
        _plot_df[_k] = _v.value_counts(normalize=True).reindex(_cats).fillna(0)
    _fig, _ax = plt.subplots(figsize=(18, 6))
    (_plot_df * 100).plot.bar(ax=_ax, rot=30, logy=True)
    _ax.set_ylabel("доля альбомов, % (лог. шкала)")
    _ax.set_title("label: частоты топ-10 категорий и категории Unknown до и после заполнения 10% пропусков")
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(
    albums,
    label_artist_coverage,
    label_mode_value,
    label_sim_metrics,
    label_true_freq,
    label_variants,
    mo,
):
    _lm = label_sim_metrics
    _n_real_artist = int((label_variants["Мода по исполнителю → Unknown"][albums["label"].isna()] != "Unknown").sum())
    _mode_col = [c for c in _lm.columns if c.startswith("доля «")][0]
    mo.md(f"""
    **Выводы по `label`.**

    - **Мода** подставляет `{label_mode_value}` всем {int(albums['label'].isna().sum())} альбомам, хотя нет оснований считать их независимыми релизами. В эксперименте мода угадывает лишь {_lm.loc['Мода', 'точность восстановления, %']:.1f}% лейблов и раздувает долю `{label_mode_value}` с {label_true_freq.iloc[0] * 100:.1f}% до {_lm.loc['Мода', _mode_col]:.1f}%: частотное распределение заметно искажается.
    - **Отдельная категория `Unknown`** не восстанавливает ни одного значения, но честно помечает пропуск и не меняет соотношение частот существующих категорий. Плата за это — новая крупная категория.
    - **Мода по исполнителю** восстанавливает {_lm.loc['Мода по исполнителю → Unknown', 'точность восстановления, %']:.1f}% лейблов (применима к {label_artist_coverage:.1%} пропусков) Варианты с модой по исполнителю дают наименьшее расстояние до истинного распределения: TVD ≈ {_lm.loc[['Мода по исполнителю → Unknown', 'Мода по исполнителю → мода'], 'TVD с истинным распределением'].min():.3f} против {_lm.loc['Мода', 'TVD с истинным распределением']:.3f} у моды. Если оставшиеся пропуски заполнять общей модой, доля `{label_mode_value}` растёт до {_lm.loc['Мода по исполнителю → мода', _mode_col]:.1f}%. Если категорией `Unknown` — частоты реальных лейблов почти не меняются.
    - На реальных пропусках мода по исполнителю восстановила лейбл для {_n_real_artist} альбомов из {int(albums['label'].isna().sum())}: у всех синглов Kaizen — лейбл `Kaizen`, у Anthony Hamilton — `RCA Records Label`. Остальным присвоена категория `Unknown`.
    - **Выбор для итогового набора: мода по исполнителю, иначе `Unknown`, плюс флаг `label_imputed`.**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.3 Категориальный признак со структурными пропусками `artist_1`

    `artist_1` пуст у треков без приглашённого исполнителя. Сравним заполнение модой и отдельной категорией, а также вариант с индикатором пропуска (`MissingIndicator`).
    """)
    return


@app.cell
def _(MissingIndicator, df, impute_column, mo, pd):
    _a1_mode, _, _ = impute_column(df, "artist_1", "most_frequent")
    _a1_const, _, _ = impute_column(df, "artist_1", "constant", "Нет")
    _a1_flag = MissingIndicator(features="all").fit_transform(df[["artist_1"]].astype(object)).ravel().astype(int)
    _a1 = {
        "Исходные (без пропусков)": df["artist_1"].dropna(),
        "Мода": pd.Series(_a1_mode),
        "Константа «Нет»": pd.Series(_a1_const),
    }
    _tbl = {}
    for _k, _v in _a1.items():
        _vc = _v.value_counts(normalize=True).head(3) * 100
        _tbl[_k] = [f"{i}: {p:.2f}%" for i, p in _vc.items()]
    mo.vstack([
        mo.md(f"Пропусков в `artist_1`: **{df['artist_1'].isna().mean():.2%}**. "
              f"Индикатор пропуска `artist_1_missing`: доля единиц = {_a1_flag.mean():.4f}. "
              "Три самые частые категории для каждого варианта:"),
        pd.DataFrame(_tbl, index=["1-я", "2-я", "3-я"]),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Выводы по `artist_1`.** При заполнении модой «Sound Healing Center» становится приглашённым исполнителем у 77% треков: распределение разрушается, и в данных появляется ложная информация. Отдельная категория «Нет» сохраняет пропорции между реальными исполнителями и смысл пропуска. Индикатор пропуска совпадает с признаком «у трека один исполнитель». Для структурных пропусков правильная обработка — отдельная категория или производные признаки. В итоговом наборе столбцы `artist_1` – `artist_6` заменены на `n_artists` и `featured_artists`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.4 Применение выбранных способов обработки пропусков
    """)
    return


@app.cell
def _(
    album_feature_matrix,
    albums,
    df,
    group_median_impute,
    group_mode_impute,
    missforest_impute,
    mo,
    np,
    pd,
):
    FEAT_COLS = [f"artist_{i}" for i in range(1, 7)]

    df_filled = df.copy()

    # 1. Полностью пустые столбцы
    empty_cols = [c for c in df_filled.columns if df_filled[c].isna().all()]
    df_filled = df_filled.drop(columns=empty_cols)

    # 2. «Битые» записи: одновременно нет названия трека, исполнителя и длительности
    broken_rows = df_filled.index[df_filled["track_name"].isna() & df_filled["artist_0"].isna()
                                  & df_filled["duration_sec"].isna()]
    df_filled = df_filled.drop(index=broken_rows)

    # 3. Текстовые поля: отдельная категория
    for _c in ["track_name", "album_name", "artist_0"]:
        df_filled[_c] = df_filled[_c].fillna("Unknown")

    # 4. Приглашённые исполнители: структурные пропуски -> признаки n_artists и featured_artists
    _stacked = df_filled[FEAT_COLS].stack().dropna()
    df_filled["n_artists"] = 1 + df_filled[FEAT_COLS].notna().sum(axis=1)
    df_filled["featured_artists"] = (_stacked.groupby(level=0).agg("; ".join)
                                     .reindex(df_filled.index).fillna(""))
    df_filled = df_filled.drop(columns=FEAT_COLS)

    # 5. label (уровень альбома): мода лейбла исполнителя -> Unknown
    _alb_label = group_mode_impute(albums["label"], albums["artist_id"])
    _label_src = np.where(albums["label"].notna(), "исходное",
                          np.where(_alb_label.notna(), "мода по исполнителю", "Unknown"))
    _alb_label = _alb_label.fillna("Unknown")
    df_filled["label_imputed"] = df_filled["label"].isna().astype(int)
    df_filled["label"] = df_filled["album_id"].map(_alb_label)

    # 6. release_year (уровень альбома): медиана по исполнителю -> MissForest
    _alb_year = group_median_impute(albums["release_year"], albums["artist_id"])
    _X_full = album_feature_matrix(albums, "release_year")
    _mf_year = missforest_impute(_X_full)["release_year"]
    _year_src = np.where(albums["release_year"].notna(), "исходное",
                         np.where(_alb_year.notna(), "медиана по исполнителю", "MissForest"))
    _alb_year = _alb_year.fillna(_mf_year).round().astype(int)
    df_filled["release_year_imputed"] = df_filled["release_year"].isna().astype(int)
    df_filled["release_year"] = df_filled["album_id"].map(_alb_year).astype(int)
    df_filled["release_date"] = df_filled["release_date"].fillna(
        pd.to_datetime(df_filled["release_year"].astype(str) + "-01-01"))

    imputation_log = (albums.assign(label_new=_alb_label, label_source=_label_src,
                                    year_new=_alb_year, year_source=_year_src)
                      .loc[lambda d: (d["label_source"] != "исходное") | (d["year_source"] != "исходное"),
                           ["album_name", "artist_name", "label", "label_new", "label_source",
                            "release_year", "year_new", "year_source"]])

    mo.vstack([
        mo.md(f"Удалены пустые столбцы: `{', '.join(empty_cols)}`; удалено «битых» строк: **{len(broken_rows)}**. "
              f"Осталось пропусков во всём наборе: **{int(df_filled.isnull().sum().sum())}**."),
        mo.md("**Журнал заполнения пропусков уровня альбома:**"),
        imputation_log,
    ])
    return (df_filled,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Исследование распределений числовых признаков

    Для анализа выбраны два признака с экстремальными значениями:
    - `duration_sec` — длительность трека (уровень трека): от секунд до нескольких часов;
    - `total_tracks` — число треков в релизе (уровень альбома): от 1 до почти 1000.

    `total_tracks` анализируется **на уровне альбомов** (одна строка — один релиз). На уровне треков релиз из 984 треков встречался бы 984 раза, и распределение было бы смещено в сторону больших значений.
    """)
    return


@app.cell
def _(df_filled, np, plt, sns, stats):
    def diagnostic_plots(df_, variable, title, log=False):
        """Гистограмма, Q-Q plot, скрипичная диаграмма и ящик с усами (функция из лекции)."""
        x = np.log(df_[variable]) if log else df_[variable]
        fig = plt.figure(figsize=(14, 8))
        plt.subplot(2, 2, 1)
        x.hist(bins=60)
        plt.title("Гистограмма")
        plt.subplot(2, 2, 2)
        stats.probplot(x, dist="norm", plot=plt)
        plt.title("Q-Q plot (нормальное распределение)")
        plt.subplot(2, 2, 3)
        sns.violinplot(x=x)
        plt.subplot(2, 2, 4)
        sns.boxplot(x=x)
        fig.suptitle(title)
        plt.tight_layout()
        return fig


    albums_filled = df_filled.drop_duplicates("album_id").set_index("album_id")
    diagnostic_plots(df_filled, "duration_sec", "duration_sec — исходный масштаб")
    return albums_filled, diagnostic_plots


@app.cell
def _(df_filled, diagnostic_plots):
    diagnostic_plots(df_filled, "duration_sec", "ln(duration_sec)", log=True)
    return


@app.cell
def _(RANDOM_STATE, df_filled, mo, np, pd, plt, stats):
    def distribution_fit_report(x, sample_size=5000, seed=RANDOM_STATE):
        """Сравнение эмпирического распределения с нормальным, логнормальным и гамма-распределением."""
        x = pd.Series(x).dropna()
        x = x[x > 0]
        xs = x.sample(min(sample_size, len(x)), random_state=seed)
        lx, lxs = np.log(x), np.log(xs)
        candidates = {
            "Нормальное": (stats.norm, stats.norm.fit(xs)),
            "Логнормальное": (stats.lognorm, stats.lognorm.fit(xs, floc=0)),
            "Гамма": (stats.gamma, stats.gamma.fit(xs, floc=0)),
        }
        ks_rows = {}
        for name, (dist, params) in candidates.items():
            ks = stats.kstest(xs, dist.cdf, args=params)
            ks_rows[name] = {"KS-статистика D": ks.statistic, "p-value": ks.pvalue}
        ks_df = pd.DataFrame(ks_rows).T

        moments = pd.DataFrame({
            "x": {"асимметрия": stats.skew(x), "эксцесс": stats.kurtosis(x)},
            "ln x": {"асимметрия": stats.skew(lx), "эксцесс": stats.kurtosis(lx)},
        }).T
        tests = pd.Series({
            "Шапиро–Уилк для ln x (выборка 5000): W": stats.shapiro(lxs).statistic,
            "Шапиро–Уилк для ln x: p-value": stats.shapiro(lxs).pvalue,
            "Д'Агостино для ln x: p-value": stats.normaltest(lxs).pvalue,
        }, name="значение").to_frame()

        # Логнормальное распределение с параметрами по всей выборке (MLE)
        s, scale = lx.std(ddof=0), np.exp(lx.mean())
        levels = [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]
        q_df = pd.DataFrame({
            "эмпирический": x.quantile(levels).to_numpy(),
            "логнормальный": stats.lognorm.ppf(levels, s, scale=scale),
            "гамма": stats.gamma.ppf(levels, *candidates["Гамма"][1]),
        }, index=[f"{p:.1%}" for p in levels])
        q_df["эмп. / логнорм."] = q_df["эмпирический"] / q_df["логнормальный"]
        q_df["эмп. / гамма"] = q_df["эмпирический"] / q_df["гамма"]
        return {"ks": ks_df, "moments": moments, "tests": tests, "quantiles": q_df, "lognorm": (s, scale)}


    def qq_panel(x, title, lognorm_params):
        x = pd.Series(x).dropna()
        x = x[x > 0]
        s, scale = lognorm_params
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        stats.probplot(x, dist="norm", plot=axes[0])
        axes[0].set_title("x против нормального")
        stats.probplot(x, dist=stats.lognorm, sparams=(s, 0, scale), plot=axes[1])
        axes[1].set_title(f"x против логнормального (σ={s:.3f}, scale={scale:.1f})")
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        stats.probplot(np.log(x), dist="norm", plot=axes[2])
        axes[2].set_title("ln x против нормального")
        fig.suptitle(title)
        plt.tight_layout()
        return fig


    dur_fit = distribution_fit_report(df_filled["duration_sec"])
    mo.vstack([
        mo.md("**duration_sec.** Гипотеза H₀: длительность трека имеет **логнормальное** распределение "
              "(ln x — нормальное). Причина: длительность — положительная величина, и отклонения от типичной длины "
              "действуют мультипликативно (в разы)."),
        mo.hstack([mo.vstack([mo.md("Критерий Колмогорова–Смирнова (выборка 5000)"), dur_fit["ks"]]),
                   mo.vstack([mo.md("Моменты"), dur_fit["moments"]]),
                   mo.vstack([mo.md("Тесты нормальности ln x"), dur_fit["tests"]])], justify="start", gap=2),
        mo.md("**Эмпирические и теоретические (логнормальные) квантили:**"),
        dur_fit["quantiles"],
    ])
    return distribution_fit_report, dur_fit, qq_panel


@app.cell
def _(df_filled, dur_fit, qq_panel):
    qq_panel(df_filled["duration_sec"], "duration_sec: Q-Q диаграммы", dur_fit["lognorm"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Выводы по `duration_sec`.**

    - В исходном масштабе распределение резко скошено вправо (асимметрия ≈ 24, эксцесс ≈ 2090). Q-Q диаграмма против нормального распределения сильно изогнута, гипотеза о нормальности отвергается однозначно.
    - После логарифмирования распределение в центральной части становится почти симметричным. Но остаются **левосторонняя асимметрия (−1.09) и тяжёлые хвосты (эксцесс 5.9)**. Критерии Шапиро–Уилка и Д'Агостино отвергают нормальность $\ln x$ (p ≈ 0).
    - По статистике Колмогорова–Смирнова ближе всего гамма-распределение (D ≈ 0.066), затем логнормальное (≈ 0.092) и нормальное (≈ 0.114). Формально отвергаются все три. На выборках в тысячи наблюдений критерии чувствительны к малейшим отклонениям, поэтому важнее величина D и вид графиков.
    - Сравнение квантилей: от 5%- до 99%-квантиля эмпирические значения отличаются от логнормальных не более чем на 15%, но **хвосты тяжелее теоретических**. 0.1%-квантиль в 3.7 раза меньше логнормального (12 с против 45 с), 99.9%-квантиль в 1.5 раза больше (1274 с против 846 с). У гамма-распределения правый хвост ещё легче (99.9% ≈ 618 с).
    - **Вывод:** центральная часть распределения приближённо согласуется с логнормальной гипотезой, но в обоих хвостах экстремальных значений больше, чем предсказывает модель. Эти значения — кандидаты в выбросы, и искать их лучше в логарифмическом масштабе.
    """)
    return


@app.cell
def _(albums_filled, diagnostic_plots):
    diagnostic_plots(albums_filled, "total_tracks", "total_tracks (альбомы) — исходный масштаб")
    return


@app.cell
def _(albums_filled, diagnostic_plots):
    diagnostic_plots(albums_filled, "total_tracks", "ln(total_tracks) (альбомы)", log=True)
    return


@app.cell
def _(albums_filled, distribution_fit_report, mo, pd):
    tt_fit = distribution_fit_report(albums_filled["total_tracks"])
    tt_non_single = albums_filled.loc[albums_filled["album_type"] != "single", "total_tracks"]
    tt_fit_ns = distribution_fit_report(tt_non_single)
    mo.vstack([
        mo.md("**total_tracks.** Гипотеза H₀: число треков в релизе имеет логнормальное распределение."),
        mo.hstack([mo.vstack([mo.md("KS — все релизы"), tt_fit["ks"]]),
                   mo.vstack([mo.md("KS — альбомы и сборники (без синглов)"), tt_fit_ns["ks"]]),
                   mo.vstack([mo.md("Моменты — все / без синглов"),
                              pd.concat([tt_fit["moments"], tt_fit_ns["moments"]], keys=["все", "без синглов"])])],
                  justify="start", gap=2),
        mo.md("**Квантили: все релизы / без синглов:**"),
        pd.concat([tt_fit["quantiles"], tt_fit_ns["quantiles"]], axis=1, keys=["все релизы", "без синглов"]),
    ])
    return (tt_non_single,)


@app.cell
def _(albums_filled, np, plt, stats, tt_non_single):
    _fig, _axes = plt.subplots(1, 3, figsize=(18, 5))
    for _t, _c in zip(["single", "album", "compilation"], ["tab:blue", "tab:orange", "tab:green"]):
        _axes[0].hist(np.log(albums_filled.loc[albums_filled["album_type"] == _t, "total_tracks"]),
                      bins=np.arange(-0.25, 7.2, 0.25), alpha=0.6, label=_t, color=_c)
    _axes[0].set_title("ln(total_tracks) по типам релиза")
    _axes[0].set_xlabel("ln(total_tracks)")
    _axes[0].legend()
    stats.probplot(np.log(albums_filled["total_tracks"]), dist="norm", plot=_axes[1])
    _axes[1].set_title("ln(total_tracks), все релизы, против нормального")
    stats.probplot(np.log(tt_non_single), dist="norm", plot=_axes[2])
    _axes[2].set_title("ln(total_tracks), альбомы и сборники, против нормального")
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Выводы по `total_tracks`.**

    - Распределение по всем релизам — **смесь двух подсовокупностей**: синглы (70% релизов, почти всегда 1 трек) и альбомы / сборники (обычно 10–17 треков). Гистограмма $\ln x$ двухмодальная, Q-Q диаграмма имеет горизонтальную «полку» при x = 1. Логнормальная гипотеза для всех релизов отвергается (D ≈ 0.32): эмпирическая медиана равна 1, теоретическая — 2.6.
    - Для альбомов и сборников (без синглов) логнормальная модель подходит заметно лучше: D ≈ 0.125, лучший результат среди трёх распределений. В центре квантили совпадают с теоретическими с точностью 5–12%. Правый хвост при этом намного тяжелее: 99%-квантиль — 68 треков против 44, 99.9%-квантиль — 193 против 65.
    - Признак дискретный, поэтому Q-Q диаграмма ступенчатая. Это следствие дискретности, а не отклонение от гипотезы.
    - **Вывод:** логнормальная гипотеза приемлема только для альбомов и сборников и только в центральной части. Правый хвост содержит экстремально большие релизы, а методы, рассчитанные на симметричное одномодальное распределение, на этом признаке будут работать плохо.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Обнаружение выбросов

    Используются методы из лекции и методических указаний, а также их варианты для логарифмированного признака (обоснованы результатами раздела 5):

    | Метод | Границы |
    |---|---|
    | Правило трёх сигм | $\mu \pm 3\sigma$ |
    | Правило трёх сигм для $\ln x$ | $\exp(\mu_{\ln x} \pm 3\sigma_{\ln x})$ |
    | Межквартильный размах, K = 1.5 | $Q1 - 1.5\cdot IQR,\; Q3 + 1.5\cdot IQR$ |
    | Межквартильный размах, K = 3 | $Q1 - 3\cdot IQR,\; Q3 + 3\cdot IQR$ |
    | IQR для $\ln x$, K = 1.5 | $\exp(Q1_{\ln x} - 1.5\cdot IQR_{\ln x}),\; \exp(Q3_{\ln x} + 1.5\cdot IQR_{\ln x})$ |
    | Квантили 5% / 95% | $q_{0.05},\; q_{0.95}$ |
    """)
    return


@app.cell
def _(Enum, albums_filled, df_filled, np, pd):
    class OutlierBoundaryType(Enum):
        SIGMA = "3σ"
        SIGMA_LOG = "3σ по ln x"
        IQR = "IQR, K=1.5"
        IQR_3 = "IQR, K=3"
        IQR_LOG = "IQR по ln x, K=1.5"
        QUANTILE = "квантили 5% / 95%"


    def get_outlier_boundaries(x, outlier_boundary_type):
        """Нижняя и верхняя границы выбросов (функция из лекции, расширенная)."""
        x = pd.Series(x).dropna()
        obt = OutlierBoundaryType
        if outlier_boundary_type in (obt.SIGMA_LOG, obt.IQR_LOG):
            x = np.log(x)
        if outlier_boundary_type in (obt.SIGMA, obt.SIGMA_LOG):
            lower, upper = x.mean() - 3 * x.std(), x.mean() + 3 * x.std()
        elif outlier_boundary_type in (obt.IQR, obt.IQR_3, obt.IQR_LOG):
            k = 3 if outlier_boundary_type == obt.IQR_3 else 1.5
            q1, q3 = x.quantile(0.25), x.quantile(0.75)
            lower, upper = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
        elif outlier_boundary_type == obt.QUANTILE:
            lower, upper = x.quantile(0.05), x.quantile(0.95)
        else:
            raise NameError("Unknown Outlier Boundary Type")
        if outlier_boundary_type in (obt.SIGMA_LOG, obt.IQR_LOG):
            lower, upper = np.exp(lower), np.exp(upper)
        return lower, upper


    def outlier_mask(x, lower, upper):
        return (x < lower) | (x > upper)


    OUTLIER_DATA = {
        "duration_sec": df_filled["duration_sec"],
        "total_tracks": albums_filled["total_tracks"],
    }

    _rows, outlier_masks = [], {}
    for _feat, _x in OUTLIER_DATA.items():
        for _obt in OutlierBoundaryType:
            _lo, _up = get_outlier_boundaries(_x, _obt)
            _m = outlier_mask(_x, _lo, _up)
            outlier_masks[(_feat, _obt.value)] = _m
            _rows.append({
                "признак": _feat, "метод": _obt.value,
                "нижняя граница": _lo, "верхняя граница": _up,
                "ниже": int((_x < _lo).sum()), "выше": int((_x > _up).sum()),
                "всего": int(_m.sum()), "доля, %": _m.mean() * 100,
            })
    outlier_report = pd.DataFrame(_rows).set_index(["признак", "метод"])
    outlier_report.round(3)
    return (
        OUTLIER_DATA,
        OutlierBoundaryType,
        get_outlier_boundaries,
        outlier_mask,
        outlier_masks,
        outlier_report,
    )


@app.cell
def _(OUTLIER_DATA, OutlierBoundaryType, np, outlier_report, plt, sns):
    _fig, _axes = plt.subplots(1, 2, figsize=(20, 6))
    _palette = dict(zip([o.value for o in OutlierBoundaryType], sns.color_palette("tab10", 6)))
    for _ax, (_feat, _x) in zip(_axes, OUTLIER_DATA.items()):
        _ax.hist(_x, bins=np.logspace(np.log10(_x.min()), np.log10(_x.max()), 80), color="lightgray")
        _ax.set_xscale("log")
        _ax.set_yscale("log")
        for _obt in OutlierBoundaryType:
            _lo, _up = outlier_report.loc[(_feat, _obt.value), ["нижняя граница", "верхняя граница"]]
            for _b, _ls in [(_lo, "--"), (_up, "-")]:
                if _x.min() <= _b <= _x.max():
                    _ax.axvline(_b, color=_palette[_obt.value], ls=_ls, lw=1.6)
            _ax.plot([], [], color=_palette[_obt.value], label=_obt.value)
        _ax.set_title(f"{_feat}: гистограмма (лог. оси) и границы выбросов (-- нижняя, — верхняя)")
        _ax.legend(fontsize=9)
    plt.tight_layout()
    _fig
    return


@app.cell
def _(OUTLIER_DATA, OutlierBoundaryType, np, outlier_masks, pd, plt, sns):
    def jaccard_matrix(feature):
        names = [o.value for o in OutlierBoundaryType]
        mat = pd.DataFrame(index=names, columns=names, dtype=float)
        for a in names:
            for b in names:
                ma, mb = outlier_masks[(feature, a)], outlier_masks[(feature, b)]
                union = (ma | mb).sum()
                mat.loc[a, b] = (ma & mb).sum() / union if union else np.nan
        return mat


    _fig, _axes = plt.subplots(1, 2, figsize=(20, 7))
    for _ax, _feat in zip(_axes, OUTLIER_DATA):
        sns.heatmap(jaccard_matrix(_feat), annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, ax=_ax)
        _ax.set_title(f"{_feat}: коэффициент Жаккара между множествами выбросов")
    plt.tight_layout()
    _fig
    return


@app.cell
def _(OutlierBoundaryType, mo, outlier_masks, pd):
    def nested_check(feature):
        """Вложенность множеств: какая доля выбросов метода A найдена и методом B."""
        names = [o.value for o in OutlierBoundaryType]
        return pd.DataFrame({
            b: {a: (outlier_masks[(feature, a)] & outlier_masks[(feature, b)]).sum()
                   / max(outlier_masks[(feature, a)].sum(), 1) for a in names}
            for b in names
        })


    mo.vstack([
        mo.md("**Доля выбросов метода-строки, которые также найдены методом-столбцом** — duration_sec:"),
        nested_check("duration_sec").round(3),
        mo.md("total_tracks:"),
        nested_check("total_tracks").round(3),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Что представляют собой экстремальные значения

    Чтобы понять, являются ли экстремальные значения ошибками или корректными наблюдениями, посмотрим на сами записи и классифицируем их по ключевым словам в названиях треков и альбомов.
    """)
    return


@app.cell
def _(OutlierBoundaryType, df_filled, get_outlier_boundaries, mo, pd):
    CONTENT_PATTERNS = {
        "аудиокнига / лекция / подкаст": r"chapter|kapitel|cap[ií]tulo|unabridged|ungek[üu]rzt|lesung|h[öo]rbuch|"
                                         r"audiobook|interview|lecture|podcast|episode|folge|teil \d|novel|story|stories",
        "DJ-микс / сет": r"\bmix(ed)?\b|continuous|\bset\b|megamix|non-?stop|\bdj\b",
        "звуки для сна / медитации": r"sleep|rain|noise|meditat|relax|ambient|ocean|waves|\bhz\b|binaural|"
                                     r"frequenc|healing|\bspa\b|nature|asmr|yoga|calm",
        "классика / концертная запись": r"symphon|concerto|sonata|suite|\bop\.|opus|\blive\b|opera|requiem|mass\b",
        "интро / скит / интерлюдия": r"\bintro\b|\boutro\b|skit|interlude|prelude|\binterlude\b",
        "сэмплы / звуковые эффекты": r"sample|\bsfx\b|effect|\bloop\b|ringtone|s\.e\.|jingle|drum|\btone\b|beep",
        "сборник / бокс-сет / караоке": r"complete|collection|\bbox\b|anthology|essential|recordings|hits|greatest|"
                                        r"\bbest\b|top \d+|\d{2,} songs|karaoke|workout|party|vol\.",
    }


    def classify_content(text_series, patterns):
        text = text_series.str.lower()
        result = pd.Series("прочее", index=text.index)
        for name, pat in reversed(list(patterns.items())):
            result = result.mask(text.str.contains(pat, regex=True, na=False), name)
        return result


    _dur_lo, _dur_up = get_outlier_boundaries(df_filled["duration_sec"], OutlierBoundaryType.SIGMA_LOG)
    _text = df_filled["track_name"] + " | " + df_filled["album_name"]
    content_class = classify_content(_text, CONTENT_PATTERNS)
    dur_content = pd.DataFrame({
        "все треки, %": content_class.value_counts(normalize=True) * 100,
        f"длиннее {_dur_up:.0f} с, %": content_class[df_filled["duration_sec"] > _dur_up].value_counts(normalize=True) * 100,
        f"короче {_dur_lo:.0f} с, %": content_class[df_filled["duration_sec"] < _dur_lo].value_counts(normalize=True) * 100,
    }).fillna(0).round(2)
    mo.vstack([
        mo.md("**duration_sec: тип контента среди всех треков и среди экстремальных значений (границы 3σ по ln x)**"),
        dur_content,
        mo.md("**10 самых длинных треков:**"),
        df_filled.nlargest(10, "duration_sec")[["track_name", "album_name", "artist_0", "duration_sec"]]
            .assign(**{"длительность, ч": lambda d: (d["duration_sec"] / 3600).round(2)}),
        mo.md("**10 самых коротких треков:**"),
        df_filled.nsmallest(10, "duration_sec")[["track_name", "album_name", "artist_0", "duration_sec"]],
    ])
    return CONTENT_PATTERNS, classify_content


@app.cell
def _(
    CONTENT_PATTERNS,
    OutlierBoundaryType,
    albums_filled,
    classify_content,
    get_outlier_boundaries,
    mo,
    pd,
):
    _tt_lo, _tt_up = get_outlier_boundaries(albums_filled["total_tracks"], OutlierBoundaryType.IQR_LOG)
    _alb_class = classify_content(albums_filled["album_name"], CONTENT_PATTERNS)
    tt_content = pd.DataFrame({
        "все релизы, %": _alb_class.value_counts(normalize=True) * 100,
        "все альбомы и сборники, %": _alb_class[albums_filled["album_type"] != "single"].value_counts(normalize=True) * 100,
        f"total_tracks > {_tt_up:.0f}, %": _alb_class[albums_filled["total_tracks"] > _tt_up].value_counts(normalize=True) * 100,
    }).fillna(0).round(2)
    mo.vstack([
        mo.md(f"**total_tracks: тип релиза по названию; выше границы IQR по ln x ({_tt_up:.0f}) — "
              f"{int((albums_filled['total_tracks'] > _tt_up).sum())} релизов**"),
        tt_content,
        mo.md("**Тип релиза (`album_type`) у альбомов выше границы:**"),
        albums_filled.loc[albums_filled["total_tracks"] > _tt_up, "album_type"].value_counts().to_frame(),
        mo.md("**15 релизов с наибольшим числом треков:**"),
        albums_filled.nlargest(15, "total_tracks")[["album_name", "artist_0", "album_type", "label", "total_tracks"]],
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Сравнение методов обнаружения выбросов.**

    `duration_sec`:

    - **3σ** даёт отрицательную нижнюю границу (−167 с) и находит только длинные треки (длиннее 600 с, 0.70%). Для скошенного распределения правило не подходит: среднее и σ сами «раздуты» хвостом, а короткие выбросы метод не видит.
    - **IQR, K = 1.5** отмечает 4.6% треков, в основном длиннее 6.5 мин (389 с). Это слишком строго: такие треки обычны для прогрессив-рока, электроники и классики. **IQR, K = 3** отмечает 1.1%, тоже только длинные треки; его множество полностью вложено в множество K = 1.5.
    - **Квантили 5% / 95%** по определению отсекают 10% данных при любой форме распределения (короче 84 с и длиннее 373 с). Это края распределения, а не выбросы.
    - **IQR по ln x** отмечает 6.3%, **3σ по ln x** — 1.6% (5 834 трека короче 47 с и 1 188 треков длиннее 811 с). Только эти методы симметрично учитывают оба хвоста в мультипликативном масштабе. Множество 3σ по ln x целиком вложено в множества IQR по ln x и квантильного метода.
    - Коэффициенты Жаккара показывают, что методы находят сильно различающиеся множества. Методы в исходном масштабе из-за отрицательных нижних границ совпадают с логарифмическими только на длинных треках.

    `total_tracks` (по альбомам):

    - Из-за смеси синглов и альбомов квартили сжаты (Q1 = 1, Q3 = 9). Поэтому **IQR, K = 1.5** и **квантили** объявляют выбросами все релизы длиннее 20–21 трека (4.1–4.5%), то есть обычные расширенные издания. **3σ** (больше 40 треков, 1.3%) и **IQR, K = 3** (больше 33, 1.7%) тоже отмечают обычные двойные альбомы и сборники.
    - **3σ по ln x** (больше 92 треков, 138 релизов) и **IQR по ln x** (больше 243 треков, 9 релизов) выделяют действительно экстремальные релизы. Эти множества вложены друг в друга и во множества остальных методов.

    **Различия объясняются формой распределения.** Правило 3σ и квантильный метод неявно предполагают симметрию. IQR в исходном масштабе устойчив к хвостам, но на скошенных данных и на смеси подсовокупностей даёт слишком узкие границы. Логарифмирование приближает распределения к нормальному, и границы в логарифмическом масштабе отделяют только по-настоящему нетипичные значения.

    **Являются ли экстремальные значения выбросами?**

    - **Длинные треки.** Среди треков длиннее 811 с в 6 раз чаще встречаются классика и концертные записи (27.3% против 4.5% по всему набору), почти в 3 раза чаще — DJ-миксы (8.2% против 2.9%) и чаще — аудиокниги. Самые длинные треки — многочасовые сеты, медитации, интервью. Это **корректные наблюдения**, хотя такой контент нетипичен для музыкальных треков.
    - **Короткие треки.** Среди треков короче 47 с в 17 раз чаще встречаются интро, скиты и интерлюдии (13.4% против 0.8%) и в 12 раз чаще — сэмплы и звуковые эффекты (4.3% против 0.4%). Это тоже в основном корректные наблюдения. Сомнительны единичные значения: трек «Sky High (Feat. Popoff)» длительностью 0.18 с — скорее всего, ошибка, потому что песня с приглашённым исполнителем не может длиться долю секунды.
    - **Релизы с числом треков больше 243** — прежде всего **аудиокниги** (две трети, например «The Ultimate Horror Collection: 60+ Novels And Stories…» из 984 глав) и полные собрания записей. Значения корректны, но такие релизы относятся к другой совокупности, чем музыкальные альбомы.
    - **Итог:** экстремальные значения в основном не являются ошибками. Однако они сильно влияют на среднее и дисперсию и соответствуют нетипичному контенту, поэтому их следует обработать, не удаляя наблюдения.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Обработка выбросов

    **Выбор способа обработки.** Экстремальные значения в основном корректны, поэтому удалять строки неоправданно: пришлось бы выбросить реальные треки (классику, миксы) и целые релизы. Выбрана **замена выбросов границами (винсоризация)**, а сам факт выброса сохраняется **флагом** в отдельном столбце:

    - `duration_sec` — границы **3σ по ln x** (≈ [47; 811] с). Логарифмический масштаб соответствует результатам раздела 5, обрабатываются оба хвоста, затронуто 1.6% треков.
    - `total_tracks` — верхняя граница **IQR по ln x** (243 трека). Это самый консервативный метод: он затрагивает только явно аномальные релизы (9 альбомов, в основном аудиокниги). Остальные методы из-за смеси синглов и альбомов отмечают как выбросы обычные альбомы и сборники.

    Для сравнения приведён и вариант с удалением выбросов.
    """)
    return


@app.cell
def _(
    OutlierBoundaryType,
    albums_filled,
    df_filled,
    get_outlier_boundaries,
    mo,
    np,
    numeric_summary,
    outlier_mask,
    pd,
):
    DUR_BOUNDS = get_outlier_boundaries(df_filled["duration_sec"], OutlierBoundaryType.SIGMA_LOG)
    TT_UPPER = int(np.floor(get_outlier_boundaries(albums_filled["total_tracks"], OutlierBoundaryType.IQR_LOG)[1]))


    def trim(x, lower, upper):
        return x[~outlier_mask(x, lower, upper)]


    def winsorize(x, lower, upper):
        return x.clip(lower, upper)


    _dur = df_filled["duration_sec"]
    _tt = albums_filled["total_tracks"]
    outlier_variants = {
        "duration_sec": {
            "исходные": _dur,
            "удаление": trim(_dur, *DUR_BOUNDS),
            "замена границами": winsorize(_dur, *DUR_BOUNDS),
        },
        "total_tracks": {
            "исходные": _tt,
            "удаление": trim(_tt, -np.inf, TT_UPPER),
            "замена границами": winsorize(_tt, None, TT_UPPER),
        },
    }
    outlier_effect = pd.concat(
        {feat: numeric_summary(v).assign(min=[s.min() for s in v.values()], max=[s.max() for s in v.values()])
         for feat, v in outlier_variants.items()}
    )
    mo.vstack([
        mo.md(f"Границы для `duration_sec`: **[{DUR_BOUNDS[0]:.1f}; {DUR_BOUNDS[1]:.1f}] с**; "
              f"верхняя граница для `total_tracks`: **{TT_UPPER}**."),
        outlier_effect.round(3),
    ])
    return DUR_BOUNDS, TT_UPPER, outlier_variants


@app.cell
def _(np, outlier_variants, plt, sns):
    _fig, _axes = plt.subplots(1, 2, figsize=(18, 5))
    for _ax, (_feat, _vars) in zip(_axes, outlier_variants.items()):
        for (_name, _s), _ls in zip(_vars.items(), ["-", "--", ":"]):
            sns.kdeplot(np.log(_s), ax=_ax, label=_name, ls=_ls, lw=2, bw_adjust=1.5)
        _ax.set_xlabel(f"ln({_feat})")
        _ax.set_title(f"{_feat}: распределение до и после обработки выбросов")
        _ax.legend()
    plt.tight_layout()
    _fig
    return


@app.cell
def _(diagnostic_plots, outlier_variants, pd):
    diagnostic_plots(pd.DataFrame({"duration_sec": outlier_variants["duration_sec"]["замена границами"]}),
                     "duration_sec", "duration_sec после замены выбросов границами (3σ по ln x)")
    return


@app.cell
def _(TT_UPPER, diagnostic_plots, outlier_variants, pd):
    diagnostic_plots(pd.DataFrame({"total_tracks": outlier_variants["total_tracks"]["замена границами"]}),
                     "total_tracks", f"total_tracks (альбомы) после замены выбросов границей {TT_UPPER}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Выводы по обработке выбросов.**

    - `duration_sec`: после замены границами std снизилось со 127.9 до 93.3 с, асимметрия — с 24.1 до 1.76, эксцесс — с ≈ 2090 до 7.3. Медиана и квартили не изменились, среднее уменьшилось лишь на 1.7 с, число строк сохранилось. При удалении характеристики похожи (std 86.6, асимметрия 1.42), но теряется 7 022 трека (1.6%), а медиана смещается вверх (206.2 вместо 205.3 с), потому что коротких выбросов больше, чем длинных.
    - `total_tracks`: максимум снизился с 984 до 243, std — с 11.3 до 10.2, асимметрия — с 21.4 до 7.6, эксцесс — с ≈ 1381 до 115. Медиана и квартили не изменились. Признак по-прежнему скошен вправо, но это естественное свойство смеси синглов и альбомов, а не следствие выбросов.
    - На диаграммах после обработки видны скопления значений на границах — неизбежное следствие винсоризации. Флаги `duration_outlier` и `total_tracks_outlier` сохраняют информацию о том, что значение было экстремальным.
    - **Последствия:** распределения стали пригоднее для методов, чувствительных к масштабу и экстремальным значениям (линейные модели, KNN, методы на основе среднего и дисперсии), и при этом не потеряно ни одного объекта. Цена — частичная потеря информации о величине экстремальных значений (её частично сохраняют флаги).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Сравнение исходного и обработанного наборов данных

    ### 8.1 Формирование итогового набора
    """)
    return


@app.cell
def _(DUR_BOUNDS, TT_UPPER, df_filled, mo, np, outlier_mask):
    df_final = df_filled.copy()

    # Выбросы duration_sec: флаг + замена границами (3σ по ln x)
    df_final["duration_outlier"] = outlier_mask(df_final["duration_sec"], *DUR_BOUNDS).astype(int)
    df_final["duration_sec"] = df_final["duration_sec"].clip(*DUR_BOUNDS)

    # Выбросы total_tracks: флаг + замена границей (IQR по ln x, уровень альбома)
    df_final["total_tracks_outlier"] = (df_final["total_tracks"] > TT_UPPER).astype(int)
    df_final["total_tracks"] = df_final["total_tracks"].clip(upper=TT_UPPER)

    # Логическое ограничение track_number <= total_tracks (исправляет и исходные 25 нарушений)
    _tn_before = df_final["track_number"].copy()
    df_final["track_number"] = np.minimum(df_final["track_number"], df_final["total_tracks"])
    track_number_fixed = int((_tn_before != df_final["track_number"]).sum())

    # duration_ms дублирует duration_sec
    df_final = df_final.drop(columns=["duration_ms"])

    df_final = df_final[[
        "track_id", "track_name", "track_number", "duration_sec", "duration_outlier",
        "artist_0", "n_artists", "featured_artists",
        "album_id", "album_name", "album_type", "total_tracks", "total_tracks_outlier",
        "release_date", "release_year", "release_year_imputed",
        "label", "label_imputed", "album_popularity", "artist_id",
    ]]
    mo.vstack([
        mo.md(f"Значений `track_number`, приведённых к ограничению `track_number ≤ total_tracks`: **{track_number_fixed}**."),
        df_final.head(),
    ])
    return (df_final,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 8.2 Сводная статистика изменений
    """)
    return


@app.cell
def _(df_final, df_raw, mo, pd):
    _n_alb_before = df_raw["album_id"].nunique()
    _n_alb_after = df_final["album_id"].nunique()
    overall_cmp = pd.DataFrame({
        "исходный набор": {
            "строк": len(df_raw), "столбцов": df_raw.shape[1], "альбомов": _n_alb_before,
            "пропущенных ячеек (NaN)": int(df_raw.isnull().sum().sum()),
            "доля пропущенных ячеек, %": df_raw.isnull().mean().mean() * 100,
            "строк хотя бы с одним NaN": int(df_raw.isnull().any(axis=1).sum()),
            "объём в памяти, МБ": df_raw.memory_usage(deep=True).sum() / 2 ** 20,
        },
        "итоговый набор": {
            "строк": len(df_final), "столбцов": df_final.shape[1], "альбомов": _n_alb_after,
            "пропущенных ячеек (NaN)": int(df_final.isnull().sum().sum()),
            "доля пропущенных ячеек, %": df_final.isnull().mean().mean() * 100,
            "строк хотя бы с одним NaN": int(df_final.isnull().any(axis=1).sum()),
            "объём в памяти, МБ": df_final.memory_usage(deep=True).sum() / 2 ** 20,
        },
    })
    overall_cmp["изменение"] = overall_cmp["итоговый набор"] - overall_cmp["исходный набор"]
    _removed = sorted(set(df_raw.columns) - set(df_final.columns))
    _added = sorted(set(df_final.columns) - set(df_raw.columns))
    mo.vstack([
        overall_cmp.round(3),
        mo.md(f"**Удалённые столбцы ({len(_removed)}):** `{', '.join(_removed)}`  \n"
              f"**Добавленные столбцы ({len(_added)}):** `{', '.join(_added)}`"),
    ])
    return


@app.cell
def _(df_final, df_raw, np, pd):
    # Сравнение по общим столбцам на общих строках
    _raw_aligned = df_raw.loc[df_final.index].copy()
    _raw_aligned["release_date"] = pd.to_datetime(_raw_aligned["release_date"].str[:10], errors="coerce")
    _common = [c for c in df_final.columns if c in df_raw.columns]


    def _changed(a, b):
        both_na = a.isna() & b.isna()
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            diff = ~np.isclose(a.astype(float), b.astype(float), equal_nan=True)
            return int((pd.Series(diff, index=a.index) & ~both_na).sum())
        return int(((a != b) & ~both_na).sum())


    changes_df = pd.DataFrame({
        "NaN до": df_raw[_common].isnull().sum(),
        "NaN после": df_final[_common].isnull().sum(),
        "изменено значений": {c: _changed(_raw_aligned[c], df_final[c]) for c in _common},
    })
    changes_df["изменено, %"] = changes_df["изменено значений"] / len(df_final) * 100
    changes_df.round(4)
    return


@app.cell
def _(describe_numeric, df_final, df_raw, pd):
    df_before = df_raw.assign(release_year=pd.to_datetime(df_raw["release_date"].str[:10], errors="coerce").dt.year)
    _cols = ["duration_sec", "total_tracks", "track_number", "album_popularity", "release_year"]
    _stats_before = describe_numeric(df_before, _cols)
    _stats_after = describe_numeric(df_final, _cols)
    stats_cmp = pd.concat({"до": _stats_before, "после": _stats_after}, axis=1).swaplevel(axis=1)
    stats_cmp = stats_cmp[[c for s in ["count", "min", "median", "mean", "max", "std", "skew"]
                           for c in [(s, "до"), (s, "после")]]]
    stats_cmp.round(3)
    return (df_before,)


@app.cell
def _(describe_numeric, df_final, df_raw, mo, pd):
    albums_before = df_raw.drop_duplicates("album_id")
    albums_after = df_final.drop_duplicates("album_id")
    mo.vstack([
        mo.md("**Признаки уровня альбома (одна строка = один релиз):**"),
        pd.concat({"до": describe_numeric(albums_before, ["total_tracks", "album_popularity"]),
                   "после": describe_numeric(albums_after, ["total_tracks", "album_popularity"])}).round(3),
        mo.md("**Флаги обработки (доля строк, %):**"),
        (df_final[["duration_outlier", "total_tracks_outlier", "release_year_imputed", "label_imputed"]]
            .mean() * 100).round(4).rename("доля, %").to_frame(),
    ])
    return albums_after, albums_before


@app.cell
def _(albums_after, albums_before, df_before, df_final, df_raw, np, pd, plt):
    _fig, _axes = plt.subplots(2, 2, figsize=(18, 10))
    _pairs = [
        ("duration_sec", np.log(df_raw.loc[df_raw["duration_sec"] > 0, "duration_sec"]), np.log(df_final["duration_sec"]), "ln(duration_sec), треки"),
        ("total_tracks", np.log(albums_before["total_tracks"]), np.log(albums_after["total_tracks"]), "ln(total_tracks), альбомы"),
        ("release_year", df_before["release_year"], df_final["release_year"], "release_year, треки"),
        ("track_number", np.log(df_raw["track_number"]), np.log(df_final["track_number"]), "ln(track_number), треки"),
    ]
    for _ax, (_name, _b, _a, _title) in zip(_axes.flat, _pairs):
        _bins = np.histogram_bin_edges(pd.concat([_b, _a]).dropna(), bins=80)
        _ax.hist(_b.dropna(), bins=_bins, alpha=0.5, label="исходный", density=True)
        _ax.hist(_a.dropna(), bins=_bins, histtype="step", lw=1.8, label="итоговый", density=True, color="tab:red")
        _ax.set_title(_title)
        _ax.legend()
    _axes[1, 0].set_xlim(1880, 2025)
    plt.suptitle("Распределения ключевых числовых признаков до и после обработки")
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Выводы по итоговому набору данных.**

    - **Объём данных почти не изменился.** Удалены 2 «битые» строки из 438 973 (0.0005%), все 78 172 альбома сохранены. Столбцов стало 20 вместо 26: удалены 6 пустых столбцов, 6 разреженных столбцов приглашённых исполнителей и дублирующий `duration_ms`; добавлены `release_year`, `n_artists`, `featured_artists` и 4 флага обработки. Объём в памяти сократился примерно на 23%.
    - **Пропусков не осталось.** В исходных данных было 5.09 млн пустых ячеек (44.6%, почти все — в пустых и разреженных столбцах), в итоговом — 0. Значения восстановлены для 82 строк `label` и 124 строк `release_date` / `release_year`.
    - **Доли изменённых значений:** длительность — 1.6% строк (винсоризация), `total_tracks` — 0.1% строк (9 альбомов), `track_number` — 25 логически некорректных значений. Остальные признаки изменены в сотых и тысячных долях процента.
    - **Характеристики признаков.** Медианы и квартили всех числовых признаков сохранились. Изменились только характеристики, чувствительные к хвостам: std `duration_sec` — на −27%, асимметрия — с 24.1 до 1.76; асимметрия `total_tracks` — с 13.2 до 4.5 по трекам и с 21.4 до 7.6 по альбомам. Минимальный год релиза вырос с 1886 до 1929 благодаря исправлению дат-заглушек. `album_popularity` практически не изменилась.
    - Итоговый набор не содержит пропусков, невозможных значений и логических противоречий. Флаги `*_imputed` и `*_outlier` позволяют при моделировании учитывать, какие значения были восстановлены или ограничены.

    **Важно.** В работе не обучается модель, поэтому статистики для заполнения пропусков и границы выбросов вычислены по всему набору. При построении модели их нужно вычислять **только на обучающей выборке** (разбивая данные по `album_id`, чтобы треки одного альбома не попали в обе выборки) и применять к тестовой выборке с теми же параметрами. Иначе информация из тестовых данных «просочится» в обучение.
    """)
    return


if __name__ == "__main__":
    app.run()
