"""
Közös segédfüggvények: adatbetöltés, heti aggregálás és a modellezési tábla felépítése.

Az előrejelzés logikája ("direkt több horizontú előrejelzés"):
  - az előrejelzés készítésének hete az ún. origó (t), addig ismerjük a tényleges eladásokat;
  - a cél a t+h hét eladása, ahol h = 1..4 hét;
  - a jellemzők két csoportja:
      * az origóig ismert múltbeli adatok (utolsó hetek eladásai, mozgóátlagok),
      * a célhétre előre ismert adatok (akciós naptár, munkaszüneti napok, naptári hét,
        az előző év ugyanazon heti eladása).
  Az akciókat a cég előre megtervezi, ezért ezek jogosan használhatók jellemzőként.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"
IMAGES = ROOT / "images"

HORIZONS = [1, 2, 3, 4]
TEST_START = pd.Timestamp("2024-12-30")        # a 2025-ös hetek: tesztidőszak
VALIDATION_START = pd.Timestamp("2024-07-01")  # 2024 második fele: validáció (biztonsági készlethez)
RANDOM_STATE = 42

FEATURES = [
    "h", "category_code",
    # múltbeli adatok az origóig
    "lag_0", "lag_1", "lag_2", "lag_3", "roll_4", "roll_8", "roll_13", "base_level",
    "promo_days_origin",
    # a célhétre előre ismert adatok
    "promo_days", "discount", "promo_prev_week", "holiday_days",
    "week_of_year", "month", "last_year", "last_year_promo_days",
]

FEATURE_LABELS_HU = {
    "h": "Előrejelzési távolság (hét)",
    "category_code": "Termékkategória",
    "lag_0": "Eladás az utolsó ismert héten",
    "lag_1": "Eladás 1 héttel korábban",
    "lag_2": "Eladás 2 héttel korábban",
    "lag_3": "Eladás 3 héttel korábban",
    "roll_4": "4 hetes mozgóátlag",
    "roll_8": "8 hetes mozgóátlag",
    "roll_13": "13 hetes mozgóátlag",
    "base_level": "Alapszint (8 hetes medián)",
    "promo_days_origin": "Akció az utolsó ismert héten",
    "promo_days": "Akciós napok a célhéten",
    "discount": "Akciós kedvezmény mértéke",
    "promo_prev_week": "Akció az előző héten",
    "holiday_days": "Munkaszüneti napok a célhéten",
    "week_of_year": "Naptári hét",
    "month": "Hónap",
    "last_year": "Előző év azonos heti eladása",
    "last_year_promo_days": "Akció az előző év azonos hetén",
}


def load_raw():
    products = pd.read_csv(RAW / "products.csv")
    daily = pd.read_csv(RAW / "daily_sales.csv", parse_dates=["date"])
    promos = pd.read_csv(RAW / "promotions.csv", parse_dates=["start_date", "end_date"])
    holidays = pd.read_csv(RAW / "holidays.csv", parse_dates=["date"])
    return products, daily, promos, holidays


def weekly_table(extra_weeks: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Heti tábla termékenként: eladott darab + az adott hét akciós és ünnepnapi adatai.

    Az utolsó tényhét után `extra_weeks` jövőbeli hét is szerepel (eladás nélkül),
    hogy a következő hetekre is lehessen előrejelzést készíteni.
    """
    products, daily, promos, holidays = load_raw()

    daily["week"] = daily["date"] - pd.to_timedelta(daily["date"].dt.weekday, unit="D")
    sales = daily.groupby(["sku", "week"], as_index=False)["quantity"].sum()

    last_week = sales["week"].max()
    weeks = pd.date_range(sales["week"].min(), last_week + pd.Timedelta(weeks=extra_weeks), freq="W-MON")
    grid = pd.MultiIndex.from_product([products["sku"], weeks], names=["sku", "week"]).to_frame(index=False)
    w = grid.merge(sales, on=["sku", "week"], how="left")
    w.loc[w["week"] > last_week, "quantity"] = np.nan  # jövőbeli hetek

    # akciós napok és kedvezmény hetente
    promo_days = []
    for pr in promos.itertuples():
        for d in pd.date_range(pr.start_date, pr.end_date, freq="D"):
            promo_days.append((pr.sku, d - pd.Timedelta(days=d.weekday()), pr.discount_pct))
    pdays = pd.DataFrame(promo_days, columns=["sku", "week", "discount"])
    pweek = pdays.groupby(["sku", "week"]).agg(promo_days=("discount", "size"), discount=("discount", "max"))
    w = w.merge(pweek.reset_index(), on=["sku", "week"], how="left")
    w[["promo_days", "discount"]] = w[["promo_days", "discount"]].fillna(0)

    # munkaszüneti napok hetente (hétköznapra vagy szombatra eső ünnepek)
    holidays["week"] = holidays["date"] - pd.to_timedelta(holidays["date"].dt.weekday, unit="D")
    hweek = holidays[holidays["date"].dt.weekday < 6].groupby("week").size().rename("holiday_days")
    w = w.merge(hweek.reset_index(), on="week", how="left")
    w["holiday_days"] = w["holiday_days"].fillna(0)

    w = w.merge(products[["sku", "category"]], on="sku")
    w["category_code"] = w["category"].astype("category").cat.codes
    w = w.sort_values(["sku", "week"]).reset_index(drop=True)
    w["week_idx"] = w.groupby("sku").cumcount()
    return w, products


def build_dataset(w: pd.DataFrame) -> pd.DataFrame:
    """Modellezési tábla: minden (termék, origó, horizont) hármas egy sor."""
    w = w.copy()
    g = w.groupby("sku")["quantity"]
    # origón ismert múltbeli jellemzők
    for k in range(4):
        w[f"lag_{k}"] = g.shift(k)
    for k in (4, 8, 13):
        w[f"roll_{k}"] = g.transform(lambda s, k=k: s.rolling(k).mean())
    w["base_level"] = g.transform(lambda s: s.rolling(8).median())
    w["promo_days_origin"] = w["promo_days"]

    origin_cols = ["sku", "week", "week_idx", "category", "category_code",
                   "lag_0", "lag_1", "lag_2", "lag_3", "roll_4", "roll_8", "roll_13",
                   "base_level", "promo_days_origin"]
    origin = w[origin_cols].rename(columns={"week": "origin_week", "week_idx": "origin_idx"})
    origin = origin[origin["roll_13"].notna()]  # legalább 13 hét előzmény kell

    # célhétre előre ismert jellemzők
    t = w[["sku", "week", "week_idx", "quantity", "promo_days", "discount", "holiday_days"]].copy()
    t["promo_prev_week"] = t.groupby("sku")["promo_days"].shift(1).fillna(0)
    t["week_of_year"] = t["week"].dt.isocalendar().week.astype(int)
    t["month"] = (t["week"] + pd.Timedelta(days=3)).dt.month
    t["last_year"] = t.groupby("sku")["quantity"].shift(52)
    t["last_year_promo_days"] = t.groupby("sku")["promo_days"].shift(52)
    t = t.rename(columns={"week": "target_week", "week_idx": "target_idx", "quantity": "y"})

    parts = []
    for h in HORIZONS:
        o = origin.copy()
        o["h"] = h
        o["target_idx"] = o["origin_idx"] + h
        parts.append(o.merge(t, on=["sku", "target_idx"], how="inner"))
    return pd.concat(parts, ignore_index=True)


def wape(y, f) -> float:
    """Súlyozott átlagos abszolút százalékos hiba: sum|y - f| / sum(y)."""
    y, f = np.asarray(y, float), np.asarray(f, float)
    return float(np.abs(y - f).sum() / y.sum())


def bias(y, f) -> float:
    """Torzítás: (sum(f) - sum(y)) / sum(y); pozitív = túlbecslés."""
    y, f = np.asarray(y, float), np.asarray(f, float)
    return float((f.sum() - y.sum()) / y.sum())
