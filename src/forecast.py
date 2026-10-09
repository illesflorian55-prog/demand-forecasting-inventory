"""
Keresletelőrejelzés: gépi tanulási modell összevetése egyszerű módszerekkel.

Lépések:
  1. Validáció: tanítás a 2024. július előtti célheteken, előrejelzés 2024 második felére.
     A validációs hibákból számoljuk később a biztonsági készletet (inventory.py).
  2. Teszt: tanítás a 2025 előtti célheteken, előrejelzés a 2025-ös hetekre (1–4 hét előre).
  3. Végleges modell: tanítás minden ismert adaton, előrejelzés a következő 4 hétre.

Összehasonlított módszerek:
  - Naiv: a következő hetekben ugyanannyi fogy, mint az utolsó ismert héten.
  - 4 hetes mozgóátlag: az utolsó 4 hét átlaga.
  - Szezonális naiv: az előző év ugyanazon heti eladása.
  - Gépi tanulás: gradient boosting (scikit-learn HistGradientBoostingRegressor) Poisson-veszteséggel.

Futtatás:  python src/forecast.py
Kimenet:   results/model_metrics.csv, results/forecasts_*.csv, results/feature_importance.csv
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from common import (FEATURE_LABELS_HU, FEATURES, RANDOM_STATE, RESULTS, TEST_START,
                    VALIDATION_START, bias, build_dataset, wape, weekly_table)

MODELS_HU = {
    "naive": "Naiv (utolsó hét)",
    "ma4": "4 hetes mozgóátlag",
    "seasonal_naive": "Szezonális naiv (előző év)",
    "ml": "Gépi tanulás (gradient boosting)",
}


def make_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="poisson",            # darabszám jellegű, nemnegatív célváltozó
        learning_rate=0.05,
        max_iter=500,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=1.0,
        categorical_features=[FEATURES.index("category_code")],
        random_state=RANDOM_STATE,
    )


def add_baselines(df: pd.DataFrame) -> pd.DataFrame:
    df["naive"] = df["lag_0"]
    df["ma4"] = df["roll_4"]
    # ha nincs előző évi adat, a mozgóátlag helyettesíti
    df["seasonal_naive"] = df["last_year"].fillna(df["roll_4"])
    return df


def fit_predict(train: pd.DataFrame, test: pd.DataFrame):
    model = make_model()
    model.fit(train[FEATURES], train["y"])
    pred = np.clip(model.predict(test[FEATURES]), 0, None)
    return model, pred


def metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    segments = {
        "Összes hét": df,
        "Akciós hetek": df[df["promo_days"] > 0],
        "Nem akciós hetek": df[df["promo_days"] == 0],
    }
    for seg_name, seg in segments.items():
        for m, label in MODELS_HU.items():
            rows.append({
                "szegmens": seg_name,
                "modell": label,
                "wape": round(wape(seg["y"], seg[m]), 4),
                "torzitas": round(bias(seg["y"], seg[m]), 4),
                "sorok": len(seg),
            })
    return pd.DataFrame(rows)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    w, products = weekly_table()
    data = add_baselines(build_dataset(w))
    known = data[data["y"].notna()]

    keep = ["sku", "category", "origin_week", "target_week", "h", "y", "promo_days",
            "naive", "ma4", "seasonal_naive", "ml"]

    # 1. validáció: 2024 második fele
    tr = known[known["target_week"] < VALIDATION_START]
    va = known[(known["target_week"] >= VALIDATION_START) & (known["target_week"] < TEST_START)].copy()
    _, va["ml"] = fit_predict(tr, va)
    va[keep].to_csv(RESULTS / "forecasts_validation.csv", index=False, float_format="%.3f")

    # 2. teszt: 2025
    tr = known[known["target_week"] < TEST_START]
    te = known[known["target_week"] >= TEST_START].copy()
    model, te["ml"] = fit_predict(tr, te)
    te[keep].to_csv(RESULTS / "forecasts_test.csv", index=False, float_format="%.3f")

    metrics = metrics_table(te)
    metrics.to_csv(RESULTS / "model_metrics.csv", index=False)

    by_h = []
    for h, g in te.groupby("h"):
        for m, label in MODELS_HU.items():
            by_h.append({"horizont_het": h, "modell": label, "wape": round(wape(g["y"], g[m]), 4)})
    pd.DataFrame(by_h).to_csv(RESULTS / "model_metrics_by_horizon.csv", index=False)

    # jellemzők fontossága: mennyit romlik a teszthiba, ha egy jellemző értékeit összekeverjük
    sample = te.sample(n=min(6000, len(te)), random_state=RANDOM_STATE)
    imp = permutation_importance(model, sample[FEATURES], sample["y"], n_repeats=5,
                                 random_state=RANDOM_STATE, scoring="neg_mean_absolute_error")
    fi = pd.DataFrame({
        "jellemzo": FEATURES,
        "jellemzo_hu": [FEATURE_LABELS_HU[f] for f in FEATURES],
        "fontossag_mae": imp.importances_mean.round(4),
    }).sort_values("fontossag_mae", ascending=False)
    fi.to_csv(RESULTS / "feature_importance.csv", index=False)

    # 3. végleges modell: előrejelzés a következő 4 hétre az utolsó ismert hétről
    last_origin = w.loc[w["quantity"].notna(), "week"].max()
    future = data[(data["origin_week"] == last_origin) & data["y"].isna()].copy()
    _, future["ml"] = fit_predict(known, future)
    future = future.merge(products[["sku", "product_name"]], on="sku")
    future[["sku", "product_name", "category", "origin_week", "target_week", "h",
            "promo_days", "discount", "ml"]].rename(columns={"ml": "forecast"}).to_csv(
        RESULTS / "forecast_next_4_weeks.csv", index=False, float_format="%.2f")

    # összefoglaló a konzolra
    pd.set_option("display.width", 140)
    print("Tesztidőszak: 2025 (1–4 hetes előrejelzések)")
    print(metrics.pivot(index="modell", columns="szegmens", values="wape").round(3))
    print("\nTorzítás (összes hét):")
    print(metrics[metrics["szegmens"] == "Összes hét"][["modell", "torzitas"]].to_string(index=False))
    print("\nLegfontosabb jellemzők:")
    print(fi.head(8)[["jellemzo_hu", "fontossag_mae"]].to_string(index=False))


if __name__ == "__main__":
    main()
