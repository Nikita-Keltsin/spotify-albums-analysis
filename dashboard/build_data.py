"""Пересборка данных дашборда из ноутбука ЛР4.

Запуск (из папки dashboard/):
    python build_data.py
    python build_data.py --notebook ../lab4_eda.py --out assets/data.js

Скрипт выполняет все ячейки marimo-ноутбука lab4_eda.py (ему нужен
spotify_albums_2023_clean.csv рядом с ноутбуком), собирает агрегаты для
графиков и таблиц и записывает их в assets/data.js как window.DASH_DATA.
"""
import argparse
import datetime
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TARGET = "album_popularity"


def load_notebook(path: Path):
    """Импортирует marimo-ноутбук и выполняет его, возвращая словарь переменных."""
    os.chdir(path.parent)  # ноутбук ищет CSV рядом с собой
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("lab4_eda", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _, defs = module.app.run()
    return defs


def verdict(sign, p, s, thr=0.05):
    ok = [np.sign(v) == sign and abs(v) >= thr for v in (p, s)]
    return "подтверждена" if all(ok) else ("частично" if any(ok) else "не подтверждена")


def build(d):
    alb, dom, df = d["albums"], d["dom"], d["df_tracks"]
    y = alb[TARGET]
    out = {}

    out["kpi"] = [{
        "tracks": int(len(df)), "albums": int(len(alb)),
        "artists": int(df["artist_id"].nunique()), "labels": int(df["label"].nunique()),
        "median_popularity": float(y.median()), "mean_popularity": round(float(y.mean()), 1),
        "zero_share": round(float((y == 0).mean()), 4),
        "single_share": round(float((alb["album_type"] == "single").mean()), 4),
    }]

    # Распределение популярности: интервалы по 5, последний 95–100 включительно
    edges = list(range(0, 100, 5)) + [101]
    counts, _ = np.histogram(y, bins=edges)
    out["pop_hist"] = [{"bin_start": a, "bin_label": f"{a}–{a + 4}" if a < 95 else "95–100", "albums": int(c)}
                       for a, c in zip(edges[:-1], counts)]

    g = alb.groupby("album_type")[TARGET].agg(["size", "mean", "median"]).sort_values("mean")
    out["by_type"] = [{"album_type": t, "albums": int(r["size"]), "mean_popularity": round(r["mean"], 1),
                       "median_popularity": float(r["median"])} for t, r in g.iterrows()]

    gy = alb.groupby("release_year")[TARGET].agg(["size", "mean"])
    out["by_year"] = [{"year": int(t), "albums": int(r["size"]), "mean_popularity": round(r["mean"], 1)}
                      for t, r in gy.iterrows() if t >= 1960]

    tl = (alb.groupby("label").agg(albums=(TARGET, "size"), mean_popularity=(TARGET, "mean"))
          .sort_values("albums", ascending=False).head(15))
    out["top_labels"] = [{"label": l, "albums": int(r.albums), "mean_popularity": round(r.mean_popularity, 1),
                          "is_karaoke": int(dom.loc[alb["label"] == l, "is_karaoke"].mean() > 0.5)}
                         for l, r in tl.iterrows()]

    names = {"is_karaoke": "Караоке и каверы", "is_major_label": "Мейджор-лейбл",
             "is_friday": "Релиз в пятницу", "is_special_edition": "Переиздание (deluxe, remaster)"}
    out["flags"] = [{"feature": f, "name": n, "share": round(dom[f].mean(), 4),
                     "popularity_with": round(y[dom[f] == 1].mean(), 1),
                     "popularity_without": round(y[dom[f] == 0].mean(), 1)} for f, n in names.items()]

    bins = pd.cut(dom["feat_share"], [-0.01, 0, 0.25, 0.5, 0.75, 0.99, 1],
                  labels=["0", "до 25%", "25–50%", "50–75%", "75–99%", "100%"])
    fs = alb.groupby(bins, observed=True)[TARGET].agg(["size", "mean"])
    out["feat_bins"] = [{"feat_share_bin": str(i), "albums": int(r["size"]), "mean_popularity": round(r["mean"], 1)}
                        for i, r in fs.iterrows()]

    tc = d["target_corr"]
    out["hyp"] = [{"feature": f, "hypothesis": text, "expected": "+" if s > 0 else "−",
                   "pearson": round(float(tc.loc[f, "Pearson"]), 3), "spearman": round(float(tc.loc[f, "Spearman"]), 3),
                   "verdict": verdict(s, tc.loc[f, "Pearson"], tc.loc[f, "Spearman"])}
                  for f, (s, text) in d["HYPOTHESES"].items()]

    out["encoding"] = [
        {"feature": "album_type", "kind": "номинальный", "categories": 3,
         "method": 'One-hot (drop="first")', "reason": "мало категорий, порядка нет"},
        {"feature": "label", "kind": "номинальный", "categories": int(df["label"].nunique()),
         "method": "Count encoding", "reason": "высокая кардинальность; число релизов = размер лейбла"},
        {"feature": "artist_id", "kind": "номинальный", "categories": int(df["artist_id"].nunique()),
         "method": "Count + target encoding out-of-fold", "reason": "исполнитель — главный фактор; out-of-fold убирает утечку"},
        {"feature": "release_date", "kind": "дата", "categories": int(alb["release_date"].nunique()),
         "method": "Год + месяц через sin/cos + is_friday", "reason": "месяц циклический: декабрь и январь рядом"},
        {"feature": "featured_artists", "kind": "многозначный", "categories": int(df["featured_artists"].nunique()),
         "method": "Есть/нет → доля по альбому", "reason": "важен факт коллаборации, а не имена"},
    ]

    rows = []
    for src, (dst, _) in d["LIN_MAP"].items():
        for stage, x in (("до", d["cont_df"][src]), ("после", d["lin"][dst])):
            for r in d["binned_trend"](x, d["cont_df"][TARGET]).itertuples():
                rows.append({"feature": src, "stage": stage, "x": round(float(r.x), 3),
                             "mean_popularity": round(float(r.y), 2), "albums": int(r.n)})
    out["lin_trend"] = rows

    lc = d["lin_compare"].reset_index()
    out["lin_compare"] = [{
        "feature": r["исходный"], "transform": r["преобразование"],
        "pearson_before": round(r["Pearson до"], 3), "pearson_after": round(r["Pearson после"], 3),
        "spearman_before": round(r["Spearman до"], 3), "spearman_after": round(r["Spearman после"], 3),
        "trend_r2_before": round(r["R² линии трендов до"], 2), "trend_r2_after": round(r["R² линии трендов после"], 2),
    } for _, r in lc.iterrows()]

    out["target_corr"] = [{"feature": f, "group": r["группа"], "pearson": round(float(r.Pearson), 3),
                           "spearman": round(float(r.Spearman), 3)} for f, r in tc.iterrows()]

    cp, cs = d["corr_pearson"], d["corr_spearman"]
    cols = list(cp.columns)
    out["corr"] = [{"feature_a": a, "feature_b": b, "order_a": i, "order_b": j,
                    "pearson": round(float(cp.loc[a, b]), 3), "spearman": round(float(cs.loc[a, b]), 3)}
                   for i, a in enumerate(cols) for j, b in enumerate(cols)]

    ss = d["scale_stats"].reset_index()
    out["scale"] = [{
        "feature": r["признак"], "method": r["метод"], "key": f"{r['признак']}|{r['метод']}",
        "mean": round(r["mean"], 3), "std": round(r["std"], 3), "min": round(r["min"], 3),
        "median": round(r["median"], 3), "max": round(r["max"], 3), "iqr": round(r["IQR"], 3),
        "skew": round(r["skew"], 2), "test_mean": round(r["test mean"], 3), "test_std": round(r["test std"], 3),
    } for _, r in ss.iterrows()]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--notebook", default=str(HERE.parent / "lab4_eda.py"), help="путь к lab4_eda.py")
    ap.add_argument("--out", default=str(HERE / "assets" / "data.js"), help="куда записать data.js")
    args = ap.parse_args()

    out_path = Path(args.out).resolve()
    data = build(load_notebook(Path(args.notebook).resolve()))
    stamp = datetime.date.today().isoformat()
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "// Сгенерировано build_data.py из lab4_eda.py " + stamp + ". Не редактируйте вручную.\n"
        "window.DASH_DATA = " + payload + ";\n", encoding="utf-8")
    print(f"Готово: {out_path} ({out_path.stat().st_size / 1024:.0f} КБ, наборов данных: {len(data)})")


if __name__ == "__main__":
    main()
