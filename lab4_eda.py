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
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Лабораторная работа №4. Разведочный анализ данных (EDA)

    **Набор данных:** Spotify Albums Data 2023. Используется очищенная версия `spotify_albums_2023_clean.csv`, полученная в ЛР №3: пропуски заполнены, выбросы обработаны.

    Структура работы соответствует методическим указаниям:

    1. **2.1** Первичный анализ признакового пространства
    2. **2.2** Кодирование категориальных признаков
    3. **2.3** Предметно обусловленное конструирование признаков
    4. **2.4** Линеаризация зависимостей
    5. **2.5** Корреляционный анализ (Пирсон и Спирмен)
    6. **2.6** Сравнение коэффициентов корреляции
    7. **2.7** Масштабирование признаков
    """)
    return


@app.cell
def _(mo):
    import warnings
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    from sklearn.model_selection import train_test_split, KFold
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler

    warnings.filterwarnings("ignore")
    sns.set(style="ticks")
    pd.set_option("display.max_columns", 60)

    RANDOM_STATE = 42
    TARGET = "album_popularity"
    DATA_FILE = "spotify_albums_2023_clean.csv"

    # Ищем датасет рядом с ноутбуком и в папке data/
    _nb_dir = Path(mo.notebook_dir() or Path.cwd())
    _search_dirs = [_nb_dir, _nb_dir / "data", Path.cwd(), Path.cwd() / "data"]
    _found = [d / DATA_FILE for d in _search_dirs if (d / DATA_FILE).exists()]
    if not _found:
        raise FileNotFoundError(
            f"Не найден файл {DATA_FILE}. Положите его рядом с ноутбуком или в папку data/. Проверены папки:\n"
            + "\n".join(str(d) for d in _search_dirs))
    DATA_PATH = _found[0]


    def fmt_int(value):
        """Целое число с пробелом-разделителем разрядов."""
        return f"{int(value):,}".replace(",", " ")


    def describe_numeric(frame, cols):
        """Описательные статистики числовых признаков, включая форму распределения."""
        _d = frame[cols].describe().T
        _d["skew"] = frame[cols].skew()
        _d["kurtosis"] = frame[cols].kurtosis()
        _d["nunique"] = frame[cols].nunique()
        _d["zeros, %"] = (frame[cols] == 0).mean() * 100
        return _d


    def binned_trend(x, y, n_bins=30, min_count=30):
        """Средние значения y по интервалам x: показывает форму зависимости поверх облака точек."""
        _df = pd.DataFrame({"x": np.asarray(x, float), "y": np.asarray(y, float)})
        if _df["x"].nunique() <= 100:
            _grp = _df.groupby("x")
        else:
            _grp = _df.groupby(pd.qcut(_df["x"], n_bins, duplicates="drop"), observed=True)
        _res = _grp.agg(x=("x", "mean"), y=("y", "mean"), n=("y", "size"))
        return _res[_res["n"] >= min_count]


    def count_encode(series, normalize=False):
        """Count (frequency) encoding: категория -> число (доля) её вхождений."""
        _vc = series.value_counts(normalize=normalize)
        return series.map(_vc)


    def target_encode(train_cat, train_y, apply_cat, smoothing=10, min_samples_leaf=1):
        """Target (mean) encoding со сглаживанием (та же формула, что в category_encoders.TargetEncoder).

        Код категории = λ(n)·mean_cat + (1 − λ(n))·prior, λ(n) = 1 / (1 + exp(−(n − min_samples_leaf) / smoothing)).
        Неизвестные категории получают prior — общее среднее целевого признака на обучающей части.
        """
        _prior = train_y.mean()
        _st = train_y.groupby(train_cat.to_numpy()).agg(["mean", "count"])
        _lam = 1 / (1 + np.exp(-(_st["count"] - min_samples_leaf) / smoothing))
        _codes = _prior * (1 - _lam) + _st["mean"] * _lam
        return apply_cat.map(_codes).fillna(_prior).to_numpy()


    def scatter_trend(ax, x, y, title, xlabel=None):
        """Диаграмма рассеяния + линия средних по интервалам + коэффициенты корреляции."""
        _rp = np.corrcoef(x, y)[0, 1]
        _rs = stats.spearmanr(x, y)[0]
        ax.scatter(x, y, s=3, alpha=0.06, color="tab:blue", rasterized=True)
        _t = binned_trend(x, y)
        ax.plot(_t["x"], _t["y"], "-o", color="tab:red", ms=3, lw=2, label="среднее по интервалам")
        ax.set_title(f"{title}\nPearson = {_rp:.3f},  Spearman = {_rs:.3f}", fontsize=10)
        ax.set_xlabel(xlabel or title)
        ax.set_ylabel(TARGET)
        return _rp, _rs

    return (
        DATA_PATH,
        KFold,
        MinMaxScaler,
        OneHotEncoder,
        RANDOM_STATE,
        RobustScaler,
        StandardScaler,
        TARGET,
        binned_trend,
        count_encode,
        describe_numeric,
        fmt_int,
        np,
        pd,
        plt,
        scatter_trend,
        sns,
        stats,
        target_encode,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.1 Первичный анализ признакового пространства

    ### 2.1.1 Описание набора данных
    """)
    return


@app.cell
def _(DATA_PATH, pd):
    df_tracks = pd.read_csv(DATA_PATH, low_memory=False)
    df_tracks["release_date"] = pd.to_datetime(df_tracks["release_date"], format="%Y-%m-%d")
    df_tracks.head()
    return (df_tracks,)


@app.cell(hide_code=True)
def _(TARGET, df_tracks, fmt_int, mo):
    mo.md(f"""
    **Предметная область.** Музыкальный стриминговый сервис Spotify. Каждая строка — один трек, вышедший в составе
    релиза (альбома, сингла или сборника). К треку присоединены характеристики релиза: тип, лейбл, дата выхода
    и популярность. Поэтому признаки бывают двух уровней: **уровень трека** и **уровень альбома**
    (у всех треков одного `album_id` значения одинаковые). Данные выгружены в декабре 2023 г.
    Самая поздняя дата релиза в наборе — {df_tracks['release_date'].max():%d.%m.%Y}.

    | Характеристика | Значение |
    |---|---|
    | Количество объектов (строк = треков) | {fmt_int(df_tracks.shape[0])} |
    | Количество исходных признаков | {df_tracks.shape[1]} |
    | Уникальных альбомов (`album_id`) | {fmt_int(df_tracks['album_id'].nunique())} |
    | Уникальных исполнителей альбомов (`artist_id`) | {fmt_int(df_tracks['artist_id'].nunique())} |
    | Уникальных лейблов | {fmt_int(df_tracks['label'].nunique())} |
    | Пропущенных значений (кроме структурных NaN в `featured_artists`) | {int(df_tracks.drop(columns='featured_artists').isnull().sum().sum())} |
    | Годы релизов | {df_tracks['release_year'].min()} – {df_tracks['release_year'].max()} |

    **Целевой признак — `{TARGET}`**: популярность релиза в Spotify по шкале 0–100. Spotify вычисляет её
    по числу недавних прослушиваний. Задача — **регрессия**: предсказать популярность альбома по его характеристикам.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Назначение признаков**

    | Признак | Уровень | Описание |
    |---|---|---|
    | `track_id` | трек | идентификатор трека в Spotify |
    | `track_name` | трек | название трека |
    | `track_number` | трек | порядковый номер трека в релизе |
    | `duration_sec` | трек | длительность трека, с (выбросы ограничены в ЛР3) |
    | `duration_outlier` | трек | флаг: длительность была выбросом и ограничена (ЛР3) |
    | `artist_0` | трек | основной исполнитель трека |
    | `n_artists` | трек | число исполнителей трека (1 + приглашённые) |
    | `featured_artists` | трек | приглашённые исполнители через «;». NaN — приглашённых нет |
    | `album_id` | альбом | идентификатор релиза |
    | `album_name` | альбом | название релиза |
    | `album_type` | альбом | тип релиза: `album`, `single`, `compilation` |
    | `total_tracks` | альбом | число треков в релизе (выбросы ограничены в ЛР3) |
    | `total_tracks_outlier` | альбом | флаг: `total_tracks` был выбросом |
    | `release_date` | альбом | дата выхода релиза |
    | `release_year` | альбом | год выхода релиза |
    | `release_year_imputed` | альбом | флаг: год был восстановлен в ЛР3 |
    | `label` | альбом | лейбл (правообладатель) |
    | `label_imputed` | альбом | флаг: лейбл был восстановлен в ЛР3 |
    | **`album_popularity`** | альбом | **целевой признак**: популярность релиза, 0–100 |
    | `artist_id` | альбом | идентификатор основного исполнителя релиза |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.1.2 Типы признаков
    """)
    return


@app.cell
def _(df_tracks, pd):
    SEMANTIC_TYPES = {
        "track_id": ("идентификатор", "трек"),
        "track_name": ("текст (строка)", "трек"),
        "track_number": ("числовой дискретный (порядковый)", "трек"),
        "duration_sec": ("числовой непрерывный", "трек"),
        "duration_outlier": ("бинарный", "трек"),
        "artist_0": ("категориальный номинальный", "трек"),
        "n_artists": ("числовой дискретный", "трек"),
        "featured_artists": ("категориальный номинальный, многозначный", "трек"),
        "album_id": ("идентификатор", "альбом"),
        "album_name": ("текст (строка)", "альбом"),
        "album_type": ("категориальный номинальный", "альбом"),
        "total_tracks": ("числовой дискретный", "альбом"),
        "total_tracks_outlier": ("бинарный", "альбом"),
        "release_date": ("дата", "альбом"),
        "release_year": ("числовой дискретный (временной)", "альбом"),
        "release_year_imputed": ("бинарный", "альбом"),
        "label": ("категориальный номинальный", "альбом"),
        "label_imputed": ("бинарный", "альбом"),
        "album_popularity": ("числовой дискретный, целевой", "альбом"),
        "artist_id": ("категориальный номинальный (id исполнителя)", "альбом"),
    }
    types_df = pd.DataFrame({
        "тип pandas": df_tracks.dtypes.astype(str),
        "уникальных": df_tracks.nunique(),
        "пример": df_tracks.iloc[0].astype(str).str[:30],
        "тип признака": {c: SEMANTIC_TYPES[c][0] for c in df_tracks.columns},
        "уровень": {c: SEMANTIC_TYPES[c][1] for c in df_tracks.columns},
    })
    types_df
    return (types_df,)


@app.cell(hide_code=True)
def _(mo, types_df):
    _counts = types_df["тип признака"].str.split(",| \\(").str[0].value_counts()
    _rows_md = "\n".join(f"    | {k} | {v} |" for k, v in _counts.items())
    mo.md(f"""
    **Итог по типам** (без учёта уточнений в скобках):

    | тип признака | кол-во |
    |---|---|
{_rows_md}

    - **Числовые:** `duration_sec` (непрерывный); `track_number`, `n_artists`, `total_tracks`, `release_year` и целевой `album_popularity` (дискретные).
      Популярность — целое число от 0 до 100. По смыслу это непрерывная оценка, поэтому решаем задачу регрессии.
    - **Категориальные номинальные:** `album_type` (3 категории), `label`, `artist_id`/`artist_0` (десятки тысяч категорий — высокая кардинальность),
      `featured_artists` (многозначный: в одной ячейке может быть несколько исполнителей).
    - **Бинарные:** четыре флага `*_outlier`/`*_imputed`, созданные в ЛР3.
    - **Дата:** `release_date` — источник для признаков «год», «месяц», «день недели».
    - **Текст и идентификаторы:** `track_name`, `album_name`, `track_id`, `album_id`. Идентификаторы не несут смысла
      как признаки модели. Из названий можно извлечь ключевые слова (см. 2.3).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.1.3 Описательные и частотные характеристики

    **Числовые признаки (уровень трека)**
    """)
    return


@app.cell
def _(describe_numeric, df_tracks):
    NUM_TRACK_COLS = ["track_number", "duration_sec", "n_artists", "total_tracks", "release_year", "album_popularity"]
    describe_numeric(df_tracks, NUM_TRACK_COLS).round(3)
    return (NUM_TRACK_COLS,)


@app.cell
def _(NUM_TRACK_COLS, df_tracks, plt):
    _fig, _axes = plt.subplots(2, 3, figsize=(18, 8))
    for _ax, _c in zip(_axes.flat, NUM_TRACK_COLS):
        _ax.hist(df_tracks[_c], bins=60, color="tab:blue", alpha=0.8)
        _ax.set_title(_c)
    plt.suptitle("Распределения числовых признаков (уровень трека)")
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - `duration_sec` после винсоризации в ЛР3 почти симметрична. Медиана около 3.5 мин, типичная длина поп-трека.
    - `track_number`, `n_artists`, `total_tracks` — счётные признаки с правой асимметрией: много малых значений и длинный хвост.
      У `n_artists` 75% треков без приглашённых исполнителей.
    - `release_year` имеет сильную левую асимметрию: большая часть релизов вышла после 2010 г., но в каталоге есть и старые альбомы.
    - `album_popularity`: много нулей и длинный правый хвост (подробно — на уровне альбома в 2.1.4).
    """)
    return


@app.cell
def _(df_tracks, pd):
    CAT_TRACK_COLS = ["album_type", "label", "artist_0", "artist_id", "featured_artists", "album_name", "track_name"]
    _rows = {}
    for _c in CAT_TRACK_COLS:
        _vc = df_tracks[_c].value_counts()
        _rows[_c] = {
            "уникальных": _vc.size,
            "самое частое": str(_vc.index[0])[:35],
            "частота": _vc.iloc[0],
            "доля самого частого, %": _vc.iloc[0] / len(df_tracks) * 100,
            "доля топ-10, %": _vc.iloc[:10].sum() / len(df_tracks) * 100,
            "NaN": int(df_tracks[_c].isnull().sum()),
        }
    cat_stats = pd.DataFrame(_rows).T
    cat_stats
    return


@app.cell
def _(df_tracks, mo, pd):
    _flags = ["duration_outlier", "total_tracks_outlier", "release_year_imputed", "label_imputed"]
    _albums_tmp = df_tracks.drop_duplicates("album_id")
    mo.hstack([
        mo.vstack([
            mo.md("**`album_type`: частоты**"),
            pd.DataFrame({
                "треков": df_tracks["album_type"].value_counts(),
                "треков, %": df_tracks["album_type"].value_counts(normalize=True).mul(100).round(2),
                "альбомов": _albums_tmp["album_type"].value_counts(),
                "альбомов, %": _albums_tmp["album_type"].value_counts(normalize=True).mul(100).round(2),
            }),
        ]),
        mo.vstack([
            mo.md("**Бинарные флаги: доля единиц, %**"),
            (df_tracks[_flags].mean() * 100).round(3).rename("доля 1, %").to_frame(),
        ]),
    ], justify="start", gap=3)
    return


@app.cell
def _(df_tracks, plt):
    _albums_tmp = df_tracks.drop_duplicates("album_id")
    _fig, _axes = plt.subplots(1, 2, figsize=(18, 6))
    _albums_tmp["label"].value_counts().head(15)[::-1].plot.barh(ax=_axes[0], color="tab:blue")
    _axes[0].set_title("Топ-15 лейблов по числу альбомов")
    _albums_tmp["artist_0"].value_counts().head(15)[::-1].plot.barh(ax=_axes[1], color="tab:green")
    _axes[1].set_title("Топ-15 исполнителей по числу альбомов")
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - `album_type` — единственный категориальный признак с малым числом категорий. **70%** релизов — синглы,
      но по трекам преобладают альбомы: в альбоме много треков.
    - `label` и `artist_id` имеют **очень высокую кардинальность** (десятки тысяч значений). Распределение частот
      сильно скошено: немного «крупных» лейблов и огромный хвост лейблов с одним релизом. Среди самых частых лейблов
      много **караоке-производителей** (Singer's Edge Karaoke, Stingray, Ameritz…). Это важное наблюдение для 2.3.
    - `featured_artists` пуст у 77% треков. Это **структурное** отсутствие: у трека просто нет приглашённых исполнителей.
    - Флаги ЛР3 почти постоянны (доля единиц от 0.02% до 1.6%), поэтому их предсказательная ценность мала.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.1.4 Переход к уровню альбома

    Целевой признак `album_popularity` — **характеристика альбома, а не трека**. Если анализировать таблицу треков:

    - популярность одного альбома повторяется в нескольких строках (до 50). Альбом из 50 треков получит в 50 раз
      больший вес в корреляциях, чем сингл;
    - треки одного альбома почти неразличимы по признакам уровня альбома, поэтому строки не независимы;
    - у признаков уровня трека (`track_number`, `duration_sec`) нет собственной целевой переменной.

    Поэтому для разделов 2.2–2.7 строим **таблицу альбомов**: одна строка — один релиз. Признаки уровня альбома
    переносятся как есть. Признаки уровня трека **агрегируются** (средняя длительность, доля треков с приглашёнными исполнителями).
    """)
    return


@app.cell
def _(df_tracks):
    ALBUM_LEVEL = ["album_name", "album_type", "artist_id", "label", "release_date", "release_year",
                   "total_tracks", "total_tracks_outlier", "release_year_imputed", "label_imputed", "album_popularity"]
    _g = df_tracks.groupby("album_id", sort=False)
    # Проверка: признаки уровня альбома действительно постоянны внутри альбома
    album_level_consistent = bool((_g[ALBUM_LEVEL].nunique() <= 1).all().all())

    albums = _g.agg(
        **{c: (c, "first") for c in ALBUM_LEVEL},
        artist_0=("artist_0", "first"),
        mean_duration=("duration_sec", "mean"),
        n_tracks_data=("track_id", "size"),
    ).reset_index()
    albums.head()
    return album_level_consistent, albums


@app.cell(hide_code=True)
def _(album_level_consistent, albums, describe_numeric, fmt_int, mo):
    mo.vstack([
        mo.md(f"""
    Таблица альбомов: **{fmt_int(len(albums))} строк × {albums.shape[1]} столбцов**.
    Признаки уровня альбома постоянны внутри каждого альбома: **{album_level_consistent}**.

    `n_tracks_data` — число треков альбома в выгрузке. Оно не больше 50 из-за ограничения API, тогда как
    `total_tracks` — фактическое число треков в релизе (до {albums['total_tracks'].max()}). Поэтому дальше используем `total_tracks`.
    """),
        describe_numeric(albums, ["total_tracks", "n_tracks_data", "release_year", "mean_duration", "album_popularity"]).round(3),
    ])
    return


@app.cell
def _(TARGET, albums, plt, sns):
    _fig, _axes = plt.subplots(1, 2, figsize=(18, 5))
    _axes[0].hist(albums[TARGET], bins=101, color="tab:purple")
    _axes[0].set_title(f"Распределение {TARGET} (альбомы)")
    _axes[0].set_xlabel(TARGET)
    sns.boxplot(data=albums, x="album_type", y=TARGET, ax=_axes[1], order=["single", "album", "compilation"])
    _axes[1].set_title(f"{TARGET} по типу релиза")
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(TARGET, albums, mo):
    mo.md(f"""
    **Целевой признак на уровне альбома.** Медиана — {albums[TARGET].median():.0f}, среднее — {albums[TARGET].mean():.1f},
    максимум — {albums[TARGET].max()}. У **{(albums[TARGET] == 0).mean() * 100:.1f}%** релизов популярность равна 0:
    их почти не слушают. Распределение сильно скошено вправо. Синглы в среднем заметно менее популярны,
    чем альбомы и сборники. Значит, `album_type` должен быть информативен.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.2 Кодирование категориальных признаков

    Модели машинного обучения работают с числами, поэтому категории нужно закодировать. Способ выбираем по
    **типу признака**, **числу категорий** и **наличию естественного порядка**:

    - мало категорий без порядка → **one-hot encoding**: не создаёт фиктивного порядка;
    - очень много категорий → one-hot даст десятки тысяч столбцов, поэтому нужны **count/frequency** или **target encoding**;
    - циклические величины (месяц) → **отображение на окружность** (sin/cos);
    - label encoding подходит только для деревьев или упорядоченных категорий. Для линейных моделей он вносит ложный порядок.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `album_type` → one-hot encoding

    Три категории без естественного порядка: «сингл < альбом < сборник» — не порядок, а разные форматы.
    Label encoding (0, 1, 2) сказал бы линейной модели, что сборник «вдвое дальше» от сингла, чем альбом. Это бессмысленно.
    Применяем one-hot с `drop="first"`: базовая категория `album` кодируется нулями в обоих столбцах.
    Так мы избегаем полной линейной зависимости столбцов (dummy trap).
    """)
    return


@app.cell
def _(OneHotEncoder, TARGET, albums, pd):
    ohe_type = OneHotEncoder(drop="first", sparse_output=False)
    _arr = ohe_type.fit_transform(albums[["album_type"]])
    TYPE_COLS = [f"type_{c}" for c in ohe_type.categories_[0][1:]]
    enc_type = pd.DataFrame(_arr.astype(int), columns=TYPE_COLS, index=albums.index)

    _check = pd.concat([albums[["album_type", TARGET]], enc_type], axis=1)
    _check.groupby("album_type").agg(
        альбомов=(TARGET, "size"), средняя_популярность=(TARGET, "mean"),
        **{c: (c, "first") for c in TYPE_COLS},
    ).round(2)
    return TYPE_COLS, enc_type


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `label` → count encoding

    В наборе 34 929 лейблов, и большинство встречается один раз. One-hot неприменим (≈35 тыс. столбцов).
    Target encoding для лейбла с одним альбомом просто «запоминает» его популярность, поэтому риск переобучения велик.

    **Count encoding** заменяет лейбл числом его релизов в наборе. Здесь это осмысленная величина — **«размер» лейбла**:
    крупный издатель или маленький независимый. Признак не использует целевую переменную, поэтому утечки нет.
    Недостаток: лейблы с одинаковым числом релизов становятся неразличимы. Для нас это приемлемо.
    """)
    return


@app.cell
def _(albums, count_encode, mo, pd):
    enc_label = pd.DataFrame({"label_count": count_encode(albums["label"])}, index=albums.index)
    _per_label = albums["label"].value_counts()
    _freq = count_encode(albums["label"], normalize=True)

    mo.vstack([
        mo.md(f"""
    Лейблов с одним релизом: **{(_per_label == 1).mean() * 100:.1f}%**; медиана — {_per_label.median():.0f} релиз на лейбл;
    максимум — {_per_label.max()} (`{_per_label.index[0]}`). Ниже — исходное значение, count encoding и frequency encoding
    (доля, `normalize=True`):
    """),
        pd.concat([albums[["label"]], enc_label, _freq.rename("label_freq")], axis=1)
            .drop_duplicates("label").sort_values("label_count", ascending=False).head(8),
    ])
    return (enc_label,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `artist_id` → count encoding + target encoding (out-of-fold)

    Исполнитель — главный фактор популярности: новый альбом звезды популярен почти независимо от остальных характеристик.
    Кардинальность огромная (≈37 тыс.), поэтому используем два способа:

    1. **Count encoding** → `artist_count`: число релизов исполнителя в наборе (продуктивность или «каталожность»).
    2. **Target (mean) encoding** → `artist_te`: средняя популярность **других** альбомов исполнителя.
       Реализован функцией `target_encode` (формула как в `category_encoders.TargetEncoder`) со **сглаживанием**: у редких исполнителей среднее
       стягивается к общему среднему.

    **Главный риск target encoding — утечка целевой переменной.** Если считать среднее по всей выборке,
    в код альбома войдёт его собственная популярность. Для исполнителя с одним альбомом код просто равен ответу.
    Поэтому кодируем **out-of-fold**: выборку делим на 5 частей. Каждую часть кодируем средними, вычисленными
    по остальным четырём. Если у исполнителя нет других альбомов, он получает общее среднее (априорное значение).

    Дополнительно из `artist_id` извлекаем бинарный признак `is_various_artists`: псевдоисполнитель «Various Artists»,
    то есть сборник разных исполнителей. Его id определяем как `artist_id` с наибольшим числом разных `artist_0`.
    """)
    return


@app.cell
def _(
    KFold,
    RANDOM_STATE,
    TARGET,
    albums,
    count_encode,
    df_tracks,
    np,
    pd,
    target_encode,
):
    # 1. Count encoding
    _artist_count = count_encode(albums["artist_id"])

    # 2. Target encoding out-of-fold
    artist_te_oof = np.zeros(len(albums))
    for _tr, _va in KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE).split(albums):
        artist_te_oof[_va] = target_encode(albums["artist_id"].iloc[_tr], albums[TARGET].iloc[_tr],
                                           albums["artist_id"].iloc[_va])

    # Для сравнения: «наивный» target encoding по всей выборке (с утечкой)
    artist_te_insample = target_encode(albums["artist_id"], albums[TARGET], albums["artist_id"])

    # 3. Various Artists
    VA_ID = df_tracks.groupby("artist_id")["artist_0"].nunique().idxmax()

    enc_artist = pd.DataFrame({
        "artist_count": _artist_count.to_numpy(),
        "artist_te": artist_te_oof,
        "is_various_artists": (albums["artist_id"] == VA_ID).astype(int).to_numpy(),
    }, index=albums.index)
    pd.concat([albums[["artist_0", "artist_id", TARGET]], enc_artist], axis=1).sample(8, random_state=1)
    return VA_ID, artist_te_insample, enc_artist


@app.cell
def _(
    TARGET,
    VA_ID,
    albums,
    artist_te_insample,
    df_tracks,
    enc_artist,
    mo,
    np,
    pd,
):
    _y = albums[TARGET].to_numpy()
    _single = (enc_artist["artist_count"] == 1).to_numpy()
    _leak = pd.DataFrame({
        "Pearson с целевым (все альбомы)": [np.corrcoef(artist_te_insample, _y)[0, 1],
                                             np.corrcoef(enc_artist["artist_te"], _y)[0, 1]],
        "Pearson с целевым (исполнители с 1 альбомом)": [np.corrcoef(artist_te_insample[_single], _y[_single])[0, 1],
                                                         np.nan],
    }, index=["по всей выборке (утечка)", "out-of-fold"]).round(3)
    _va_n = df_tracks.loc[df_tracks["artist_id"] == VA_ID, "artist_0"].nunique()
    mo.vstack([
        mo.md(f"**Проверка утечки.** У исполнителей с одним альбомом ({_single.mean() * 100:.0f}% альбомов) out-of-fold код "
              "постоянен и равен общему среднему, поэтому корреляция для них не определена (NaN). "
              "Это и есть правильное поведение: у таких альбомов нет «чужой» информации об исполнителе."),
        _leak,
        mo.md(f"""
    При кодировании по всей выборке корреляция завышена: часть «силы» признака — подсмотренный ответ.
    Для исполнителей с одним альбомом наивный код — **линейная функция целевого признака** (корреляция = 1):
    со сглаживанием это $0.5 \\cdot \\bar y + 0.5 \\cdot y_i$. Сглаживание не устраняет утечку, а лишь «разбавляет» её.
    Out-of-fold оценка честнее, и именно её мы используем дальше. **При обучении модели** кодировщик обучают
    только на обучающей выборке.

    `Various Artists`: id `{VA_ID}` объединяет {_va_n} разных исполнителей треков;
    таких сборников — {enc_artist['is_various_artists'].sum()} ({enc_artist['is_various_artists'].mean() * 100:.2f}%).
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `release_date` → год + месяц на окружности

    Дата — не категориальный признак, но её тоже нужно представить числами (по материалам лекции о признаках
    разных типов):

    - **год** `release_year` оставляем числовым признаком (в нём есть порядок и расстояния);
    - **месяц** — циклическая величина: декабрь и январь соседние, а число 12 и число 1 — нет. Поэтому используем
      отображение на окружность: $\text{month\_sin} = \sin(2\pi m/12)$, $\text{month\_cos} = \cos(2\pi m/12)$;
    - **день недели** (7 категорий) — в 2.3 из него сделан предметный бинарный признак `is_friday`.
    """)
    return


@app.cell
def _(TARGET, albums, np, pd, plt):
    _m = albums["release_date"].dt.month
    enc_date = pd.DataFrame({
        "month_sin": np.sin(2 * np.pi * _m / 12),
        "month_cos": np.cos(2 * np.pi * _m / 12),
    }, index=albums.index)

    _fig, _axes = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw={"width_ratios": [2, 1]})
    albums.groupby(_m)[TARGET].mean().plot.bar(ax=_axes[0], color="tab:blue")
    _axes[0].set_title("Средняя популярность по месяцу релиза")
    _axes[0].set_xlabel("месяц")
    _pts = enc_date.drop_duplicates().assign(m=_m).drop_duplicates("m").sort_values("m")
    _axes[1].scatter(_pts["month_cos"], _pts["month_sin"], s=80)
    for _, _r in _pts.iterrows():
        _axes[1].annotate(int(_r["m"]), (_r["month_cos"] * 1.12, _r["month_sin"] * 1.12), ha="center", va="center")
    _axes[1].set_aspect("equal")
    _axes[1].set_title("Месяцы на окружности: 12 и 1 — соседи")
    _axes[1].set_xlabel("month_cos")
    _axes[1].set_ylabel("month_sin")
    plt.tight_layout()
    _fig
    return (enc_date,)


@app.cell(hide_code=True)
def _(albums, mo):
    _jan1 = (albums["release_date"].dt.strftime("%m-%d") == "01-01").mean() * 100
    mo.md(f"""
    Пик популярности в январе частично объясняется **артефактом данных**: у {_jan1:.1f}% релизов дата записана как 1 января.
    Так бывает, когда известен только год. Обычно это старые каталожные альбомы, а они в среднем популярнее (см. 2.4).
    Остальные месяцы отличаются слабо. Скорее всего, месяц — малоинформативный признак.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Итоговая таблица кодирования

    | Признак | Тип | Категорий | Способ кодирования | Обоснование |
    |---|---|---|---|---|
    | `album_type` | номинальный | 3 | **One-hot** (`drop="first"`, база — `album`) | мало категорий, порядка нет; label encoding создал бы ложный порядок |
    | `label` | номинальный, высокая кардинальность | 34 929 | **Count encoding** → `label_count` | one-hot неприменим; число релизов = «размер» лейбла; не использует целевой признак |
    | `artist_id` | номинальный, высокая кардинальность | 37 012 | **Count** → `artist_count` и **target encoding out-of-fold** со сглаживанием → `artist_te`; бинарный `is_various_artists` | исполнитель — сильнейший фактор популярности; out-of-fold исключает утечку |
    | `artist_0` | номинальный | 51 946 | не кодируется отдельно | на уровне альбома дублирует `artist_id` |
    | `release_date` | дата | — | год (числовой), **месяц → sin/cos**, день недели → `is_friday` (2.3) | месяц циклический: 12 и 1 должны быть рядом |
    | `featured_artists` | номинальный многозначный | 39 273 | бинаризация «есть/нет» на уровне трека → доля по альбому `feat_share` (2.3) | конкретные имена слишком редки; важен сам факт коллаборации |
    | `*_outlier`, `*_imputed` | бинарные | 2 | уже 0/1, кодирование не нужно | — |
    | `album_name`, `track_name` | текст | — | не кодируются; ключевые слова → флаги (2.3) | свободный текст, а не категории |
    | `track_id`, `album_id` | идентификаторы | — | исключаются | уникальны для каждого объекта, обобщающей информации нет |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.3 Предметно обусловленное конструирование признаков

    Исходя из того, как устроена музыкальная индустрия, выдвигаем гипотезы о новых признаках и ожидаемом знаке связи
    с популярностью:

    | № | Признак | Формула / правило | Интерпретация и гипотеза | Ожидаемый знак |
    |---|---|---|---|---|
    | H1 | `feat_share` | доля треков альбома с непустым `featured_artists` | коллаборации приводят аудиторию обоих исполнителей → популярность выше | + |
    | H2 | `album_minutes` | `mean_duration × total_tracks / 60` | оценка общей длительности релиза (мин). Полноформатные LP выпускают состоявшиеся артисты, а короткие релизы — чаще начинающие | + |
    | H3 | `is_karaoke` | ключевые слова *karaoke, in the style of, tribute, originally performed by…* в лейбле, названии или имени исполнителя | караоке и каверы — «фоновый» каталог, его почти не слушают | − |
    | H4 | `is_major_label` | лейбл относится к мейджорам (Sony / Universal / Warner) или их импринтам: Columbia, Atlantic, Def Jam, Interscope… | у мейджоров больше ресурсов на продвижение, и они подписывают сильных артистов | + |
    | H5 | `is_friday` | релиз вышел в пятницу | с 2015 г. индустрия выпускает новинки по пятницам (Global Release Day). Пятничный релиз — признак профессионального выпуска | + |
    | H6 | `is_special_edition` | *deluxe, remaster(ed), anniversary, expanded, edition* в названии | переиздания выпускают только для успешных альбомов | + |
    """)
    return


@app.cell
def _(albums, df_tracks, np, pd):
    KARAOKE_PATTERN = (r"karaoke|\bin the style of\b|\bmade famous\b|\btribute\b|\bbacking tracks?\b"
                       r"|\boriginally performed\b|\bsing like\b")
    MAJOR_PATTERN = (r"\b(columbia|sony|warner|atlantic|rca|def jam|polydor|republic records|epic|interscope|capitol"
                     r"|island records|universal|emi|parlophone|virgin|motown|geffen|elektra|arista|rhino"
                     r"|legacy recordings|umg|ume|mercury|decca|deutsche grammophon|reprise|300 entertainment"
                     r"|aftermath|cash money|young money|big machine|bmg)\b")
    SPECIAL_PATTERN = r"\b(deluxe|remaster(ed)?|anniversary|expanded|edition)\b"

    _text = (albums["label"] + " | " + albums["album_name"] + " | " + albums["artist_0"]).str.lower()
    _feat_share = (df_tracks["featured_artists"].notna()
                   .groupby(df_tracks["album_id"]).mean()
                   .reindex(albums["album_id"]).to_numpy())

    dom = pd.DataFrame({
        "feat_share": _feat_share,
        "album_minutes": albums["mean_duration"] * albums["total_tracks"] / 60,
        "is_karaoke": _text.str.contains(KARAOKE_PATTERN, regex=True).astype(int),
        "is_major_label": albums["label"].str.lower().str.contains(MAJOR_PATTERN, regex=True).astype(int),
        "is_friday": (albums["release_date"].dt.dayofweek == 4).astype(int),
        "is_special_edition": albums["album_name"].str.lower().str.contains(SPECIAL_PATTERN, regex=True).astype(int),
    }, index=albums.index)

    HYPOTHESES = {  # признак: (ожидаемый знак, формулировка)
        "feat_share": (+1, "коллаборации повышают популярность"),
        "album_minutes": (+1, "полноформатные релизы популярнее"),
        "is_karaoke": (-1, "караоке и каверы непопулярны"),
        "is_major_label": (+1, "релизы мейджоров популярнее"),
        "is_friday": (+1, "пятничные релизы популярнее"),
        "is_special_edition": (+1, "переиздания делают для успешных альбомов"),
    }
    np.round(dom.describe().T, 3)
    return HYPOTHESES, dom


@app.cell
def _(TARGET, albums, dom, mo, pd):
    _y = albums[TARGET]
    _bin_cols = ["is_karaoke", "is_major_label", "is_friday", "is_special_edition"]
    _bin_tbl = pd.DataFrame({
        "доля альбомов с флагом, %": dom[_bin_cols].mean() * 100,
        "ср. популярность (флаг = 1)": [_y[dom[c] == 1].mean() for c in _bin_cols],
        "ср. популярность (флаг = 0)": [_y[dom[c] == 0].mean() for c in _bin_cols],
    })
    _bin_tbl["разница"] = _bin_tbl["ср. популярность (флаг = 1)"] - _bin_tbl["ср. популярность (флаг = 0)"]

    _examples = pd.concat([albums[["album_name", "label"]], dom[["is_karaoke", "is_major_label"]]], axis=1)
    mo.vstack([
        mo.md("**Бинарные предметные признаки: средняя популярность в группах**"),
        _bin_tbl.round(2),
        mo.md("**Примеры альбомов, отмеченных как караоке, и релизов мейджоров:**"),
        mo.hstack([
            _examples[_examples["is_karaoke"] == 1].sample(6, random_state=3)[["album_name", "label"]],
            _examples[_examples["is_major_label"] == 1].sample(6, random_state=3)[["album_name", "label"]],
        ], gap=2),
    ])
    return


@app.cell
def _(TARGET, albums, dom, pd, plt):
    _fig, _axes = plt.subplots(1, 3, figsize=(20, 5))
    _bins = pd.cut(dom["feat_share"], [-0.01, 0, 0.25, 0.5, 0.75, 0.99, 1],
                   labels=["0", "(0;0.25]", "(0.25;0.5]", "(0.5;0.75]", "(0.75;1)", "1"])
    albums.groupby(_bins, observed=True)[TARGET].mean().plot.bar(ax=_axes[0], color="tab:orange")
    _axes[0].set_title("Средняя популярность по доле треков с feat.")
    _axes[0].set_xlabel("feat_share")
    _bins2 = pd.qcut(dom["album_minutes"], 10)
    albums.groupby(_bins2, observed=True)[TARGET].mean().plot.bar(ax=_axes[1], color="tab:green")
    _axes[1].set_title("Средняя популярность по децилям album_minutes")
    _axes[1].tick_params(axis="x", rotation=60)
    _flags = ["is_karaoke", "is_major_label", "is_friday", "is_special_edition"]
    _means = pd.DataFrame({f: albums.groupby(dom[f])[TARGET].mean() for f in _flags}).T
    _means.plot.bar(ax=_axes[2], color=["tab:gray", "tab:red"])
    _axes[2].set_title("Средняя популярность: флаг = 0 / 1")
    _axes[2].tick_params(axis="x", rotation=20)
    _axes[2].legend(title="значение флага")
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Первые наблюдения** (строгая проверка через корреляции — в 2.6):

    - **`is_karaoke`** — самый яркий признак: средняя популярность караоке-релизов близка к нулю. Это 13% всех альбомов,
      то есть значительная часть «нулевой» популярности объясняется именно ими.
    - **`is_major_label`**: релизы мейджоров в среднем почти в 4 раза популярнее.
    - **`is_special_edition`**: переиздания редки (около 1.6%), но в среднем втрое популярнее остальных.
    - **`is_friday`**: пятничные релизы популярнее, хотя эффект умеренный.
    - **`feat_share`**: зависимость **немонотонная**. Альбом без feat. заметно менее популярен, чем альбом с несколькими
      коллаборациями, но при доле 100% популярность снова ниже. Скорее всего, здесь много синглов «исполнитель feat. исполнитель».
    - **`album_minutes`**: у коротких релизов (до ≈ 10 мин, в основном синглы) популярность почти постоянна и низкая.
      У полноформатных релизов (> 20 мин) она резко выше. Гипотеза подтверждается, но зависимость нелинейная,
      скорее «ступенька» (см. 2.4).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.4 Линеаризация зависимости

    Линейные модели и коэффициент Пирсона «видят» только **линейную** связь. Если зависимость $y(x)$ криволинейна,
    можно подобрать преобразование $z = f(x)$, при котором $y(z)$ станет ближе к прямой.

    Ниже — диаграммы рассеяния целевого признака для каждого непрерывного (и квазинепрерывного) признака альбома.
    Красная линия — **средняя популярность по интервалам** $x$ (30 квантильных интервалов либо отдельные значения
    для дискретных признаков). Сквозь шумное облако точек она показывает форму зависимости.
    """)
    return


@app.cell
def _(TARGET, albums, dom, enc_artist, enc_label, pd):
    CONT_FEATURES = ["total_tracks", "release_year", "mean_duration", "label_count",
                     "artist_count", "artist_te", "feat_share", "album_minutes"]
    cont_df = pd.concat([albums[["total_tracks", "release_year", "mean_duration", TARGET]],
                         enc_label, enc_artist[["artist_count", "artist_te"]],
                         dom[["feat_share", "album_minutes"]]], axis=1)
    return CONT_FEATURES, cont_df


@app.cell
def _(CONT_FEATURES, TARGET, cont_df, plt, scatter_trend):
    _fig, _axes = plt.subplots(2, 4, figsize=(22, 10))
    for _ax, _c in zip(_axes.flat, CONT_FEATURES):
        scatter_trend(_ax, cont_df[_c], cont_df[TARGET], _c)
    _axes[0, 0].legend(loc="upper right")
    plt.suptitle("Целевой признак в зависимости от непрерывных признаков", fontsize=14)
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Анализ формы зависимостей:**

    | Признак | Форма зависимости | Комментарий |
    |---|---|---|
    | `total_tracks` | рост с **насыщением** (вогнутая) | быстрый рост от 1 до ≈15 треков, дальше плато → **логарифм** |
    | `release_year` | **U-образная** | старые «каталожные» альбомы популярны, минимум в начале 2010-х, затем небольшой рост → **расстояние до минимума** |
    | `label_count` | **колоколообразная** в логарифмическом масштабе | маленькие лейблы непопулярны, средние — популярнее всего, гигантские (караоке-фабрики) снова непопулярны |
    | `album_minutes` | рост с насыщением | как у `total_tracks` → **логарифм** |
    | `artist_count` | колоколообразная, слабая | похожа на `label_count`, но выражена слабее |
    | `mean_duration` | слабая колоколообразная | связь очень слабая, линеаризация мало что даст |
    | `artist_te` | **почти линейная** | target encoding по построению монотонно связан с целевым признаком |
    | `feat_share` | ступенька в нуле + спад к 1 | скорее бинарный эффект «есть feat. / нет» |

    Для линеаризации выбираем **четыре** признака с явной нелинейностью:

    1. `total_tracks` → $z = \ln(\text{total\_tracks})$ — логарифм «выпрямляет» насыщающийся рост;
    2. `album_minutes` → $z = \ln(\text{album\_minutes})$ — аналогично;
    3. `label_count` → $z = (\ln(\text{label\_count}) - c)^2$, где $c$ — вершина параболы, подобранной МНК
       к зависимости $y$ от $\ln(\text{label\_count})$. Колокол превращается в монотонно убывающую зависимость;
    4. `release_year` → $z = |\text{release\_year} - y_0|$, где $y_0$ — год с минимальной средней популярностью.
       U-образная форма превращается в монотонно возрастающую (две ветви «складываются»).

    Квадратичная парабола для года не подходит: зависимость несимметрична, и вершина МНК-параболы уходит за пределы данных.
    """)
    return


@app.cell
def _(TARGET, cont_df, np, pd):
    _y = cont_df[TARGET]
    # Вершина параболы y ~ a*l^2 + b*l + c для l = ln(label_count)
    _l = np.log(cont_df["label_count"])
    _a, _b, _ = np.polyfit(_l, _y, 2)
    LABEL_VERTEX = -_b / (2 * _a)
    # Год с минимальной средней популярностью (среди лет с >= 300 альбомов)
    _by_year = cont_df.groupby("release_year")[TARGET].agg(["mean", "size"])
    YEAR_MIN = int(_by_year[_by_year["size"] >= 300]["mean"].idxmin())

    lin = pd.DataFrame({
        "log_total_tracks": np.log(cont_df["total_tracks"]),
        "log_album_minutes": np.log(cont_df["album_minutes"]),
        "label_size_dev": (_l - LABEL_VERTEX) ** 2,
        "year_dist": (cont_df["release_year"] - YEAR_MIN).abs(),
    }, index=cont_df.index)

    LIN_MAP = {  # исходный признак -> (новый признак, формула)
        "total_tracks": ("log_total_tracks", "ln(x)"),
        "album_minutes": ("log_album_minutes", "ln(x)"),
        "label_count": ("label_size_dev", f"(ln(x) − {LABEL_VERTEX:.2f})²"),
        "release_year": ("year_dist", f"|x − {YEAR_MIN}|"),
    }
    print(f"Вершина параболы для ln(label_count): c = {LABEL_VERTEX:.3f}  (≈ {np.exp(LABEL_VERTEX):.0f} релизов на лейбл)")
    print(f"Год минимальной популярности: y0 = {YEAR_MIN}")
    return LIN_MAP, YEAR_MIN, lin


@app.cell
def _(LIN_MAP, TARGET, cont_df, lin, plt, scatter_trend):
    _fig, _axes = plt.subplots(len(LIN_MAP), 2, figsize=(16, 5 * len(LIN_MAP)))
    for _i, (_src, (_dst, _formula)) in enumerate(LIN_MAP.items()):
        scatter_trend(_axes[_i, 0], cont_df[_src], cont_df[TARGET], f"ДО: {_src}")
        scatter_trend(_axes[_i, 1], lin[_dst], cont_df[TARGET], f"ПОСЛЕ: {_dst} = {_formula}")
    plt.suptitle("Диаграммы рассеяния до и после линеаризующего преобразования", fontsize=14, y=1.0)
    plt.tight_layout()
    _fig
    return


@app.cell
def _(LIN_MAP, TARGET, binned_trend, cont_df, lin, np, pd):
    def _linearity_r2(x, y):
        """R² прямой, проведённой через средние по интервалам: насколько форма зависимости близка к прямой."""
        _t = binned_trend(x, y)
        _w = _t["n"].to_numpy()
        _k, _b = np.polyfit(_t["x"], _t["y"], 1, w=np.sqrt(_w))
        _pred = _k * _t["x"] + _b
        _ss_res = np.sum(_w * (_t["y"] - _pred) ** 2)
        _ss_tot = np.sum(_w * (_t["y"] - np.average(_t["y"], weights=_w)) ** 2)
        return 1 - _ss_res / _ss_tot

    _rows = []
    _y = cont_df[TARGET]
    for _src, (_dst, _formula) in LIN_MAP.items():
        _rows.append({
            "исходный": _src, "преобразование": f"{_dst} = {_formula}",
            "Pearson до": cont_df[_src].corr(_y), "Pearson после": lin[_dst].corr(_y),
            "Spearman до": cont_df[_src].corr(_y, method="spearman"),
            "Spearman после": lin[_dst].corr(_y, method="spearman"),
            "R² линии трендов до": _linearity_r2(cont_df[_src], _y),
            "R² линии трендов после": _linearity_r2(lin[_dst], _y),
        })
    lin_compare = pd.DataFrame(_rows).set_index("исходный")
    lin_compare.round(3)
    return (lin_compare,)


@app.cell(hide_code=True)
def _(YEAR_MIN, lin_compare, mo):
    _lc = lin_compare
    mo.md(f"""
    **Выводы по линеаризации.**

    Столбец «R² линии трендов» показывает, насколько хорошо прямая описывает **красную линию средних**,
    то есть насколько форма зависимости близка к линейной (1 — идеально линейная). Коэффициент Пирсона показывает
    силу линейной связи с учётом шума.

    - **`total_tracks` → ln**: Pearson вырос с {_lc.loc['total_tracks', 'Pearson до']:.3f} до {_lc.loc['total_tracks', 'Pearson после']:.3f},
      R² тренда — с {_lc.loc['total_tracks', 'R² линии трендов до']:.2f} до {_lc.loc['total_tracks', 'R² линии трендов после']:.2f}.
      **Spearman не изменился**: логарифм монотонен и не меняет ранги. Значит, рост Пирсона объясняется
      исключительно «выпрямлением» формы зависимости.
    - **`album_minutes` → ln**: та же картина: Pearson {_lc.loc['album_minutes', 'Pearson до']:.3f} → {_lc.loc['album_minutes', 'Pearson после']:.3f}, Spearman не изменился.
    - **`label_count` → (ln x − c)²**: самый сильный эффект. Исходный Pearson {_lc.loc['label_count', 'Pearson до']:.3f} почти нулевой:
      колокол «гасит» линейную корреляцию, ведь слева связь положительная, а справа отрицательная.
      После преобразования Pearson равен {_lc.loc['label_count', 'Pearson после']:.3f}, а модуль связи вырос в несколько раз.
      Преобразование немонотонное, поэтому изменился и Spearman.
    - **`release_year` → |x − y₀|**: исходная связь отрицательна ({_lc.loc['release_year', 'Pearson до']:.3f}), но это «среднее»
      двух ветвей U. После преобразования связь положительная ({_lc.loc['release_year', 'Pearson после']:.3f}),
      и Spearman тоже заметно вырос ({_lc.loc['release_year', 'Spearman до']:.3f} → {_lc.loc['release_year', 'Spearman после']:.3f}):
      зависимость стала **ближе к монотонной**. Основная тенденция — рост популярности с удалением от {YEAR_MIN} г.
      Идеальной прямой нет: ветви «до» и «после» 2013 г. растут с разной скоростью, а у недавних релизов есть
      локальный максимум около 2016–2017 гг. Это пример того, что одним простым преобразованием можно
      лишь частично линеаризовать сложную зависимость.
    - Логарифм `album_minutes` выявил **два кластера**: синглы (≈ 3 мин, ln ≈ 1) и полноформатные альбомы
      (≈ 40 мин, ln ≈ 3.7). Внутри них тренд не строго линейный, но общая тенденция роста стала заметно ровнее.

    Невысокие абсолютные значения корреляций объясняются тем, что популярность — очень шумный признак:
    30% нулей, а главный фактор — личность исполнителя. Важно не абсолютное значение, а **рост по сравнению
    с исходным признаком**.

    **Замечание.** Параметры $c$ и $y_0$ подобраны по всей выборке. При обучении модели их нужно оценивать
    только на обучающей выборке.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.5 Корреляционный анализ

    ### 2.5.1 Коэффициент корреляции Пирсона

    $$
    \rho_P(X, Y) = \frac{\sum_{i=1}^{n}(x_i - \bar x)(y_i - \bar y)}
    {\sqrt{\sum_{i=1}^{n}(x_i - \bar x)^2 \; \sum_{i=1}^{n}(y_i - \bar y)^2}}
    $$

    Это ковариация, нормированная на произведение стандартных отклонений. $\rho_P \in [-1; 1]$.
    Он характеризует направление и силу **линейной** связи и чувствителен к выбросам.

    ### 2.5.2 Коэффициент корреляции Спирмена

    $$
    \rho_S(X, Y) = \rho_P(R_X, R_Y)
    $$

    Это коэффициент Пирсона между **рангами** наблюдений (при равных значениях используются средние ранги).
    Он характеризует силу **монотонной** связи, устойчив к выбросам и не меняется при монотонных преобразованиях признаков.

    ### Итоговый набор признаков
    """)
    return


@app.cell
def _(
    TARGET,
    TYPE_COLS,
    albums,
    dom,
    enc_artist,
    enc_date,
    enc_label,
    enc_type,
    lin,
    pd,
):
    FEATURE_GROUPS = {
        "исходный": ["total_tracks", "release_year", "mean_duration"],
        "кодированный": TYPE_COLS + ["label_count", "artist_count", "artist_te", "is_various_artists",
                                     "month_sin", "month_cos"],
        "сконструированный": ["feat_share", "album_minutes", "is_karaoke", "is_major_label", "is_friday",
                              "is_special_edition"],
        "линеаризованный": ["log_total_tracks", "log_album_minutes", "label_size_dev", "year_dist"],
    }
    FEATURE_GROUP_OF = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}
    ALL_FEATURES = [f for fs in FEATURE_GROUPS.values() for f in fs]

    data_final = pd.concat([albums[FEATURE_GROUPS["исходный"] + [TARGET]], enc_type, enc_label,
                            enc_artist, enc_date, dom, lin], axis=1)[ALL_FEATURES + [TARGET]]
    print(f"Итоговый набор: {data_final.shape[0]} объектов × {len(ALL_FEATURES)} признаков + целевой; "
          f"пропусков: {int(data_final.isnull().sum().sum())}")
    data_final.head()
    return ALL_FEATURES, FEATURE_GROUPS, FEATURE_GROUP_OF, data_final


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Проверка формул.** Реализуем оба коэффициента вручную по формулам методички и сравним
    с `pandas` и `scipy`:
    """)
    return


@app.cell
def _(TARGET, data_final, np, pd, stats):
    def pearson_manual(x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        dx, dy = x - x.mean(), y - y.mean()
        return np.sum(dx * dy) / np.sqrt(np.sum(dx ** 2) * np.sum(dy ** 2))

    def spearman_manual(x, y):
        # ранги со средним рангом для совпадающих значений
        return pearson_manual(stats.rankdata(x), stats.rankdata(y))

    _rows = []
    for _f in ["total_tracks", "label_count", "release_year"]:
        _x, _y = data_final[_f], data_final[TARGET]
        _rows.append({
            "признак": _f,
            "Pearson (вручную)": pearson_manual(_x, _y), "Pearson (pandas)": _x.corr(_y),
            "Pearson (scipy)": stats.pearsonr(_x, _y)[0],
            "Spearman (вручную)": spearman_manual(_x, _y), "Spearman (pandas)": _x.corr(_y, method="spearman"),
            "Spearman (scipy)": stats.spearmanr(_x, _y)[0],
        })
    pd.DataFrame(_rows).set_index("признак").round(6)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Значения совпадают до 6-го знака: библиотечные функции считают ровно по приведённым формулам.

    ## 2.6 Сравнение коэффициентов корреляции

    ### Корреляционные матрицы
    """)
    return


@app.cell
def _(data_final):
    corr_pearson = data_final.corr(method="pearson")
    corr_spearman = data_final.corr(method="spearman")
    return corr_pearson, corr_spearman


@app.cell
def _(corr_pearson, np, plt, sns):
    _fig, _ax = plt.subplots(figsize=(17, 14))
    sns.heatmap(corr_pearson, mask=np.triu(np.ones_like(corr_pearson, dtype=bool), k=1), annot=True, fmt=".2f",
                cmap="RdBu_r", vmin=-1, vmax=1, center=0, annot_kws={"size": 7}, ax=_ax, square=True,
                cbar_kws={"shrink": 0.7})
    _ax.set_title("Корреляционная матрица Пирсона", fontsize=14)
    plt.tight_layout()
    _fig
    return


@app.cell
def _(corr_spearman, np, plt, sns):
    _fig, _ax = plt.subplots(figsize=(17, 14))
    sns.heatmap(corr_spearman, mask=np.triu(np.ones_like(corr_spearman, dtype=bool), k=1), annot=True, fmt=".2f",
                cmap="RdBu_r", vmin=-1, vmax=1, center=0, annot_kws={"size": 7}, ax=_ax, square=True,
                cbar_kws={"shrink": 0.7})
    _ax.set_title("Корреляционная матрица Спирмена", fontsize=14)
    plt.tight_layout()
    _fig
    return


@app.cell
def _(corr_pearson, corr_spearman, np, plt, sns):
    corr_diff = corr_spearman - corr_pearson
    _fig, _ax = plt.subplots(figsize=(17, 14))
    sns.heatmap(corr_diff, mask=np.triu(np.ones_like(corr_diff, dtype=bool), k=0), annot=True, fmt=".2f",
                cmap="PuOr", vmin=-0.6, vmax=0.6, center=0, annot_kws={"size": 7}, ax=_ax, square=True,
                cbar_kws={"shrink": 0.7})
    _ax.set_title("Разность Spearman − Pearson", fontsize=14)
    plt.tight_layout()
    _fig
    return (corr_diff,)


@app.cell
def _(corr_diff, corr_pearson, corr_spearman, np, pd):
    _iu = np.triu_indices_from(corr_diff, k=1)
    _pairs = pd.DataFrame({
        "признак 1": corr_diff.index[_iu[0]], "признак 2": corr_diff.columns[_iu[1]],
        "Pearson": corr_pearson.to_numpy()[_iu], "Spearman": corr_spearman.to_numpy()[_iu],
        "Spearman − Pearson": corr_diff.to_numpy()[_iu],
    })
    # исключаем тривиальные пары «признак — его же преобразование»
    _trivial = {frozenset(p) for p in [("total_tracks", "log_total_tracks"), ("album_minutes", "log_album_minutes"),
                                       ("label_count", "label_size_dev"), ("release_year", "year_dist")]}
    _pairs = _pairs[[frozenset((a, b)) not in _trivial for a, b in zip(_pairs["признак 1"], _pairs["признак 2"])]]
    top_diff_pairs = _pairs.reindex(_pairs["Spearman − Pearson"].abs().sort_values(ascending=False).index).head(12)
    top_diff_pairs.round(3).reset_index(drop=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Анализ пар, в которых коэффициенты заметно различаются.** В таблице выше видны три причины расхождения:

    1. **Выброс определяет коэффициент Пирсона** ($|\rho_P| \gg |\rho_S|$). Пара `artist_count` — `is_various_artists`:
       Pearson ≈ 0.88, Spearman ≈ 0.24. Все 1 387 сборников «Various Artists» имеют одно и то же гигантское значение
       `artist_count = 1387`, а у обычных исполнителей — единицы и десятки. Эта точка-выброс «тянет» прямую на себя,
       и Пирсон показывает почти функциональную связь, которой на самом деле нет. Ранги не зависят от того,
       насколько велико крайнее значение, поэтому Спирмен даёт реалистичную оценку. То же для
       `type_compilation` — `artist_count` (0.62 против 0.21).
    2. **Монотонная, но нелинейная связь** ($|\rho_S| \gg |\rho_P|$). Пара `label_count` — `artist_count`:
       Pearson ≈ 0.19, Spearman ≈ 0.71. Оба признака — счётные с тяжёлым правым хвостом. Зависимость монотонная
       (исполнители больших лейблов выпускают больше релизов), но далека от прямой. Аналогично `mean_duration` —
       `album_minutes` (0.19 против 0.52) и `total_tracks` — `type_single` (−0.64 против −0.86): у синглов 1–3 трека,
       а у альбомов — широкий разброс. Ранговая корреляция лучше отражает такое «разделение».
       Сюда же относится целевой признак: `is_karaoke` (−0.26 против −0.42). Популярность сильно скошена,
       и правый хвост популярных альбомов ослабляет Пирсона.
    3. **Немонотонная связь**: коэффициенты расходятся вплоть до разных знаков. `release_year` и целевой признак:
       Pearson ≈ −0.15, Spearman ≈ 0.03. U-образная зависимость: положительная и отрицательная ветви
       компенсируют друг друга. Сюда же относятся пары «исходный признак — его немонотонное преобразование»
       (`release_year` — `year_dist`: −0.48 против +0.62). Такие случаи и требуют линеаризации.

    Кроме того, у признаков с большим числом **совпадающих значений** (бинарные флаги, `feat_share` с 70% нулей)
    ранги содержат много связок, и коэффициенты тоже могут расходиться.

    **Мультиколлинеарность.** Сильно связаны: `total_tracks` и `album_minutes` (Pearson 0.93; после логарифмирования 0.96),
    `type_single` и `log_total_tracks` (−0.90), `type_compilation` и `is_various_artists` (0.70), `artist_te` и `is_major_label` (0.52),
    `label_count` и `is_karaoke` (0.43: караоке-фабрики — самые «большие» лейблы), а также исходные признаки и их
    линеаризованные версии. В модель стоит включать один признак из сильно скоррелированной группы
    (или использовать регуляризацию).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Связь признаков с целевым значением
    """)
    return


@app.cell
def _(ALL_FEATURES, FEATURE_GROUP_OF, TARGET, corr_pearson, corr_spearman, pd):
    target_corr = pd.DataFrame({
        "группа": [FEATURE_GROUP_OF[f] for f in ALL_FEATURES],
        "Pearson": corr_pearson.loc[ALL_FEATURES, TARGET],
        "Spearman": corr_spearman.loc[ALL_FEATURES, TARGET],
    })
    target_corr["Spearman − Pearson"] = target_corr["Spearman"] - target_corr["Pearson"]
    target_corr = target_corr.reindex(target_corr["Spearman"].abs().sort_values(ascending=False).index)
    target_corr.round(3)
    return (target_corr,)


@app.cell
def _(plt, target_corr):
    _tc = target_corr.iloc[::-1]
    _colors = {"исходный": "tab:gray", "кодированный": "tab:blue", "сконструированный": "tab:orange",
               "линеаризованный": "tab:green"}
    _fig, _axes = plt.subplots(1, 2, figsize=(20, 9), sharey=True)
    for _ax, _m in zip(_axes, ["Pearson", "Spearman"]):
        _ax.barh(_tc.index, _tc[_m], color=[_colors[g] for g in _tc["группа"]])
        _ax.axvline(0, color="black", lw=0.8)
        _ax.set_title(f"Корреляция {_m} с album_popularity")
        _ax.grid(axis="x", alpha=0.3)
    _handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in _colors.values()]
    _axes[1].legend(_handles, _colors.keys(), loc="lower right")
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Проверка гипотез

    Гипотеза о предметном признаке считается **подтверждённой**, если знак обоих коэффициентов совпадает с ожидаемым
    и связь не пренебрежимо мала ($|\rho| \ge 0.05$); **частично подтверждённой** — если условие выполнено только
    для одного коэффициента. Линеаризация считается **успешной**, если модуль коэффициента Пирсона вырос.
    """)
    return


@app.cell
def _(HYPOTHESES, LIN_MAP, lin_compare, mo, np, pd, target_corr):
    def _verdict(sign, p, s, thr=0.05):
        _ok = [np.sign(v) == sign and abs(v) >= thr for v in (p, s)]
        return "подтверждена" if all(_ok) else ("частично" if any(_ok) else "не подтверждена")

    hyp_check = pd.DataFrame([{
        "признак": f, "гипотеза": text, "ожидаемый знак": "+" if sign > 0 else "−",
        "Pearson": target_corr.loc[f, "Pearson"], "Spearman": target_corr.loc[f, "Spearman"],
        "вывод": _verdict(sign, target_corr.loc[f, "Pearson"], target_corr.loc[f, "Spearman"]),
    } for f, (sign, text) in HYPOTHESES.items()]).set_index("признак")

    lin_check = pd.DataFrame([{
        "исходный → новый": f"{src} → {dst}",
        "|Pearson| до": abs(lin_compare.loc[src, "Pearson до"]),
        "|Pearson| после": abs(lin_compare.loc[src, "Pearson после"]),
        "вывод": "линеаризация успешна" if abs(lin_compare.loc[src, "Pearson после"]) > abs(lin_compare.loc[src, "Pearson до"])
                 else "не улучшила",
    } for src, (dst, _) in LIN_MAP.items()]).set_index("исходный → новый")

    mo.vstack([mo.md("**Предметные признаки (2.3)**"), hyp_check.round(3),
               mo.md("**Линеаризованные признаки (2.4)**"), lin_check.round(3)])
    return


@app.cell(hide_code=True)
def _(mo, target_corr):
    _tc = target_corr
    _f = lambda name, m="Spearman": f"{_tc.loc[name, m]:+.3f}"
    mo.md(f"""
    **Выводы по корреляционному анализу.**

    1. **Сильнейший признак — `artist_te`** (Pearson {_f('artist_te', 'Pearson')}, Spearman {_f('artist_te')}): популярность
       определяется прежде всего исполнителем. Связь почти линейна (коэффициенты близки), как и ожидалось
       от target encoding.
    2. **Предметные признаки оказались полезны.** `is_major_label` ({_f('is_major_label')}) и `is_karaoke` ({_f('is_karaoke')})
       входят в число сильнейших. По силе они сопоставимы с исходными признаками или сильнее, хотя построены
       только из текста названий и лейблов. `is_special_edition` и `is_friday` слабее, но знак совпадает с ожидаемым.
       `feat_share` по Спирмену ({_f('feat_share')}) заметно сильнее, чем по Пирсону ({_f('feat_share', 'Pearson')}).
       Главное — сам факт наличия коллабораций, а не их доля: связь ступенчатая, а не линейная.
       `album_minutes` фактически повторяет информацию `total_tracks`.
    3. **Линеаризация подтвердилась для всех четырёх признаков.** Наибольший выигрыш дали немонотонные преобразования:
       `label_size_dev` (колокол → монотонная связь) и `year_dist` (U-форма → монотонная связь). Логарифм для
       `total_tracks`/`album_minutes` увеличил Пирсона при неизменном Спирмене. Так и должно быть
       при монотонном преобразовании.
    4. **Расхождения Пирсона и Спирмена** указывают на нелинейность, выбросы или много совпадающих значений.
       Если $|\\rho_S| \\gg |\\rho_P|$, связь монотонная, но нелинейная, и помогает логарифм или ранговое преобразование.
       Если оба коэффициента близки к нулю при видимой зависимости на графике, связь немонотонная.
    5. **Слабые признаки:** `month_sin`/`month_cos`, `mean_duration`, `artist_count`, `is_various_artists`
       почти не связаны с популярностью (|ρ| ≤ 0.05). Это кандидаты на исключение. Исходный `label_count` тоже слаб,
       но его преобразованная версия `label_size_dev` полезна.
    6. Корреляция измеряет только **парную** связь. Признак со слабой корреляцией может быть полезен в сочетании с другими,
       а сильно скоррелированные признаки (например, `log_total_tracks` и `type_single`) частично дублируют друг друга.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.7 Масштабирование признаков

    Выбираем три числовых признака, **существенно различающихся по масштабу и форме распределения**:

    - `release_year`: значения около 2000, диапазон ≈ 1929–2023, распределение с левым хвостом;
    - `album_minutes`: от долей минуты до сотен минут, сильная правая асимметрия;
    - `label_count`: от 1 до сотен, очень тяжёлый правый хвост.

    Методы: **StandardScaler** (z-оценка), **MinMaxScaler** (в [0; 1]), **RobustScaler** (медиана и IQR).

    По материалам лекции данные **сначала делим** на обучающую (80%) и тестовую (20%) выборки.
    Параметры масштабирования (среднее, std, min, max, медиана, квартили) вычисляем **только на обучающей**
    и применяем к обеим выборкам. Так информация из тестовой выборки не «просочится» в обучение.
    """)
    return


@app.cell
def _(
    MinMaxScaler,
    RANDOM_STATE,
    RobustScaler,
    StandardScaler,
    data_final,
    pd,
    train_test_split,
):
    SCALE_FEATURES = ["release_year", "album_minutes", "label_count"]
    X_tr_raw, X_te_raw = train_test_split(data_final[SCALE_FEATURES], test_size=0.2, random_state=RANDOM_STATE)

    SCALERS = {"StandardScaler": StandardScaler, "MinMaxScaler": MinMaxScaler, "RobustScaler": RobustScaler}
    scaled = {"Исходные": (X_tr_raw, X_te_raw)}
    for _name, _cls in SCALERS.items():
        _sc = _cls().fit(X_tr_raw)
        scaled[_name] = (pd.DataFrame(_sc.transform(X_tr_raw), columns=SCALE_FEATURES, index=X_tr_raw.index),
                         pd.DataFrame(_sc.transform(X_te_raw), columns=SCALE_FEATURES, index=X_te_raw.index))
    X_tr_raw.describe().T.assign(range=lambda d: d["max"] - d["min"]).round(2)
    return SCALE_FEATURES, scaled


@app.cell
def _(SCALE_FEATURES, plt, scaled, sns):
    _fig, _axes = plt.subplots(1, 4, figsize=(24, 5))
    for _ax, (_name, (_tr, _)) in zip(_axes, scaled.items()):
        for _f in SCALE_FEATURES:
            sns.kdeplot(_tr[_f], ax=_ax, label=_f, fill=True, alpha=0.2, cut=0)
        _ax.set_title(_name)
        _ax.set_xlabel("значение")
        _ax.legend()
    plt.suptitle("Три признака на одной оси: до и после масштабирования (обучающая выборка)", fontsize=14)
    plt.tight_layout()
    _fig
    return


@app.cell
def _(SCALE_FEATURES, plt, scaled, sns):
    _fig, _axes = plt.subplots(len(SCALE_FEATURES), 4, figsize=(24, 5 * len(SCALE_FEATURES)))
    for _i, _f in enumerate(SCALE_FEATURES):
        for _j, (_name, (_tr, _te)) in enumerate(scaled.items()):
            _ax = _axes[_i, _j]
            sns.kdeplot(_tr[_f], ax=_ax, label="обучающая", cut=0)
            sns.kdeplot(_te[_f], ax=_ax, label="тестовая", linestyle="--", cut=0)
            _ax.set_title(f"{_f} — {_name}")
            _ax.legend()
    plt.suptitle("Плотности распределения по каждому признаку: обучающая и тестовая выборки", fontsize=14, y=1.0)
    plt.tight_layout()
    _fig
    return


@app.cell
def _(SCALE_FEATURES, pd, scaled):
    def _stats(s):
        return {"mean": s.mean(), "std": s.std(), "min": s.min(), "median": s.median(), "max": s.max(),
                "IQR": s.quantile(0.75) - s.quantile(0.25), "skew": s.skew()}

    _rows = {}
    for _f in SCALE_FEATURES:
        for _name, (_tr, _te) in scaled.items():
            _r = _stats(_tr[_f])
            _r.update({"test mean": _te[_f].mean(), "test std": _te[_f].std(),
                       "test min": _te[_f].min(), "test max": _te[_f].max()})
            _rows[(_f, _name)] = _r
    scale_stats = pd.DataFrame(_rows).T
    scale_stats.index.names = ["признак", "метод"]
    scale_stats.round(3)
    return (scale_stats,)


@app.cell(hide_code=True)
def _(mo, scale_stats):
    _s = scale_stats
    _g = lambda f, m, c: _s.loc[(f, m), c]
    mo.md(f"""
    **Сравнение методов масштабирования.**

    Общие наблюдения (видны на графиках и в таблице):

    - **Форма распределения не меняется** ни одним из методов: асимметрия (`skew`) одинакова до и после.
      Все три метода — линейные преобразования $x' = (x - a)/b$, они только сдвигают и растягивают ось.
    - До масштабирования признаки несопоставимы: на общей оси `release_year` (≈ 2000) и `label_count` (1–856)
      лежат в разных областях. В метрических алгоритмах признак с большим разбросом доминировал бы в расстоянии.
    - Параметры взяты с обучающей выборки, поэтому на тестовой среднее не равно ровно 0 (StandardScaler),
      а значения могут выходить за [0; 1] (MinMaxScaler).

    **`release_year`.** StandardScaler даёт mean = 0, std = 1, но из-за левого хвоста минимум уходит до
    {_g('release_year', 'StandardScaler', 'min'):.1f}. MinMaxScaler сжимает почти все релизы в узкую полосу у единицы
    (медиана {_g('release_year', 'MinMaxScaler', 'median'):.2f}): редкие старые альбомы растягивают диапазон.
    RobustScaler центрирует по медиане, хвост доходит до {_g('release_year', 'RobustScaler', 'min'):.1f} IQR.
    Для этого признака наиболее сбалансирован **StandardScaler**.

    **`album_minutes`.** Правая асимметрия (skew ≈ {_g('album_minutes', 'Исходные', 'skew'):.1f}) сохраняется после любого метода.
    MinMaxScaler сжимает основную массу около нуля (медиана {_g('album_minutes', 'MinMaxScaler', 'median'):.3f}),
    RobustScaler оставляет длинный хвост до {_g('album_minutes', 'RobustScaler', 'max'):.0f}. Масштабирование само по себе
    не решает проблему хвоста: сначала нужен логарифм (`log_album_minutes`, раздел 2.4).

    **`label_count`.** Самый тяжёлый хвост. После MinMaxScaler медиана равна {_g('label_count', 'MinMaxScaler', 'median'):.4f}:
    практически все значения неразличимы около 0, потому что максимум определяется несколькими гигантскими лейблами.
    RobustScaler (медиана и IQR не зависят от хвоста) сохраняет различия в «основной массе», но хвост достигает
    {_g('label_count', 'RobustScaler', 'max'):.0f}. StandardScaler занимает промежуточное положение
    (max ≈ {_g('label_count', 'StandardScaler', 'max'):.1f}). Для этого признака из «чистых» скейлеров лучше RobustScaler,
    а ещё лучше — логарифмирование перед масштабированием.

    **Вывод:** MinMaxScaler чувствителен к экстремальным значениям и плохо подходит для асимметричных признаков.
    RobustScaler устойчив к выбросам. StandardScaler корректен, когда распределение близко к симметричному.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Выбор способа масштабирования для итогового набора

    **Выбран StandardScaler, применённый после логарифмирования асимметричных признаков.** Обоснование:

    1. Анализ показал, что главная проблема — **тяжёлые хвосты счётных признаков**, а не выбор скейлера.
       Поэтому `total_tracks`, `album_minutes`, `label_count` и `artist_count` берём в логарифмированном виде:
       асимметрия падает до < 1 (см. таблицу ниже, столбец «skew до»). Экстремальные выбросы обработаны ещё в ЛР3.
       Оставшаяся асимметрия (`release_year`, `year_dist`, `mean_duration`, `artist_te`) — это реальные редкие объекты
       (старые альбомы, длинные треки, популярные исполнители), а не ошибки, и их значения ограничены.
       В таких условиях std не определяется единичными гигантскими значениями, как было у исходного `label_count`.
    2. StandardScaler приводит **все** признаки к одинаковой дисперсии (std = 1). Это стандартное требование для линейных
       моделей с регуляризацией (штраф одинаково действует на все коэффициенты), kNN, SVM, PCA (метод оперирует
       дисперсиями) и градиентного спуска.
    3. RobustScaler — разумная альтернатива, если данные не логарифмировать: он не даёт хвосту сжать основную массу.
       MinMaxScaler отвергнут: он зависит от двух крайних значений, сжимает основную массу данных и не гарантирует
       диапазон [0; 1] на новых данных.

    **Что масштабируем:** непрерывные и счётные признаки. **Что не масштабируем:** бинарные флаги и one-hot столбцы
    (уже в {0, 1}, при масштабировании теряется интерпретация «есть/нет»), а также `month_sin`/`month_cos` (уже в [−1; 1]).
    Деревьям решений и ансамблям на их основе масштабирование не требуется.
    """)
    return


@app.cell
def _(RANDOM_STATE, StandardScaler, TARGET, data_final, np, pd, train_test_split):
    final_set = data_final.copy()
    final_set["log_label_count"] = np.log(final_set["label_count"])
    final_set["log_artist_count"] = np.log(final_set["artist_count"])

    SCALE_FINAL = ["log_total_tracks", "log_album_minutes", "release_year", "year_dist", "mean_duration",
                   "feat_share", "log_label_count", "label_size_dev", "log_artist_count", "artist_te"]
    NO_SCALE_FINAL = ["type_compilation", "type_single", "is_karaoke", "is_major_label", "is_friday",
                      "is_special_edition", "is_various_artists", "month_sin", "month_cos"]

    _tr, _te = train_test_split(final_set[SCALE_FINAL + NO_SCALE_FINAL + [TARGET]], test_size=0.2,
                                random_state=RANDOM_STATE)
    final_scaler = StandardScaler().fit(_tr[SCALE_FINAL])
    final_train, final_test = _tr.copy(), _te.copy()
    final_train[SCALE_FINAL] = final_scaler.transform(_tr[SCALE_FINAL])
    final_test[SCALE_FINAL] = final_scaler.transform(_te[SCALE_FINAL])

    pd.DataFrame({
        "skew до": _tr[SCALE_FINAL].skew(),
        "train mean": final_train[SCALE_FINAL].mean(), "train std": final_train[SCALE_FINAL].std(),
        "test mean": final_test[SCALE_FINAL].mean(), "test std": final_test[SCALE_FINAL].std(),
        "train min": final_train[SCALE_FINAL].min(), "train max": final_train[SCALE_FINAL].max(),
    }).round(3)
    return SCALE_FINAL, final_train


@app.cell
def _(SCALE_FINAL, final_train, plt, sns):
    _fig, _ax = plt.subplots(figsize=(18, 6))
    sns.boxplot(data=final_train[SCALE_FINAL], ax=_ax, fliersize=1)
    _ax.set_title("Итоговые масштабированные признаки (обучающая выборка, StandardScaler)")
    _ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(final_train, mo):
    mo.md(f"""
    Итоговый набор после масштабирования: обучающая выборка — **{final_train.shape[0]}** объектов,
    признаков — **{final_train.shape[1] - 1}**. На обучающей выборке масштабированные признаки имеют mean = 0 и std = 1.
    На тестовой значения близки к ним, но не совпадают точно: параметры взяты с обучающей выборки.
    Все признаки теперь в сопоставимом масштабе (основная масса в пределах ±3).

    ## Общие выводы

    1. **Первичный анализ.** Набор Spotify (438 971 трек, 78 172 альбома) содержит числовые, категориальные
       с высокой кардинальностью, бинарные признаки, дату и текст. Целевой признак `album_popularity` задан
       на уровне альбома, поэтому анализ проведён на таблице альбомов. Популярность сильно скошена: 30% нулей.
    2. **Кодирование** выбрано по типу и кардинальности: one-hot для `album_type`, count encoding для `label`,
       count и out-of-fold target encoding для `artist_id`, sin/cos для месяца. Показано, что наивный target encoding
       даёт утечку целевой переменной.
    3. **Предметные признаки** подтвердили большинство гипотез. Особенно ценны `is_karaoke` и `is_major_label`,
       извлечённые из текста. Они сопоставимы по силе с исходными признаками или сильнее.
    4. **Линеаризация** улучшила линейную связь для всех четырёх выбранных признаков. Наибольший эффект дали
       преобразования немонотонных зависимостей (колокол и U-образная форма).
    5. **Корреляционный анализ** показал, что расхождение Пирсона и Спирмена — индикатор нелинейности, выбросов
       или немонотонности. Главный фактор популярности — исполнитель (`artist_te`).
    6. **Масштабирование** не меняет форму распределения. Для асимметричных признаков нужно сначала логарифмирование,
       затем StandardScaler, обученный только на обучающей выборке.
    """)
    return


if __name__ == "__main__":
    app.run()
