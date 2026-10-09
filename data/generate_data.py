"""
Szintetikus napi értékesítési adatkészlet generálása keresletelőrejelzéshez.

Fiktív cég: Zöldág Drogéria Kft. – drogérialánc központi raktára, amely a
boltokat és a webáruházat szolgálja ki.

Az adatok NEM valósak: rögzített véletlenmaggal (seed) generált, reprodukálható
mintaadatok. A kereslet valószerű tényezőkből áll össze:
  - termékenkénti alapszint és éves trend,
  - kategóriánkénti szezonalitás (pl. napvédelem nyáron, ajándékszettek decemberben),
  - a hét napjainak hatása (péntek–szombat csúcs, vasárnap mélypont),
  - akciók (10–33% kedvezmény, termékenként eltérő árrugalmassággal)
    és az akció utáni visszaesés (a vevők előre bevásároltak),
  - munkaszüneti napok (alacsony forgalom, előtte megugró vásárlás),
  - véletlen ingadozás (negatív binomiális eloszlás).

Futtatás:  python data/generate_data.py
Kimenet:   data/raw/products.csv, daily_sales.csv, promotions.csv, holidays.csv
"""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
START = date(2023, 1, 2)           # hétfő
END = date(2025, 12, 28)           # vasárnap -> 156 teljes hét
PROMO_PLAN_END = date(2026, 2, 1)  # az akciós naptár előre ismert, 2026 januárjára is

OUT_DIR = Path(__file__).resolve().parent / "raw"
rng = np.random.default_rng(SEED)

# --------------------------------------------------------------------------
# Termékek
# --------------------------------------------------------------------------
# kategória: (havi szezonális szorzók jan..dec, árszint Ft, "felhalmozható"-e)
CATEGORIES = {
    "Hajápolás": ([1.0, 0.95, 1.0, 1.0, 1.05, 1.05, 1.05, 1.0, 1.0, 1.0, 1.05, 1.15], 1600, True),
    "Testápolás": ([0.9, 0.9, 0.95, 1.0, 1.05, 1.05, 1.0, 0.95, 0.95, 1.0, 1.2, 1.6], 1900, True),
    "Napvédelem": ([0.15, 0.2, 0.35, 0.7, 1.5, 2.6, 3.0, 2.2, 0.6, 0.25, 0.15, 0.2], 3900, False),
    "Háztartás és tisztítás": ([0.95, 0.95, 1.3, 1.35, 1.1, 1.0, 0.95, 0.95, 1.0, 1.0, 1.0, 1.15], 1300, True),
    "Babaápolás": ([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 2700, True),
    "Szájhigiénia": ([1.0, 0.95, 1.0, 1.0, 1.0, 1.0, 1.0, 1.05, 1.15, 1.0, 1.0, 1.05], 1100, True),
}

PRODUCT_NAMES = {
    "Hajápolás": ["Sampon normál hajra 400 ml", "Sampon korpásodás ellen 250 ml",
                  "Hajbalzsam 200 ml", "Hajpakolás 300 ml", "Hajlakk 250 ml",
                  "Hajfesték barna", "Szárazsampon 200 ml"],
    "Testápolás": ["Tusfürdő 250 ml", "Testápoló 400 ml", "Golyós dezodor 50 ml",
                   "Spray dezodor 150 ml", "Kézkrém 75 ml", "Ajándékszett tusfürdő+testápoló",
                   "Folyékony szappan 500 ml"],
    "Napvédelem": ["Naptej SPF30 200 ml", "Naptej SPF50 200 ml", "Napozás utáni testápoló",
                   "Gyermek naptej SPF50+", "Napvédő arckrém SPF50"],
    "Háztartás és tisztítás": ["Mosógél 2 l", "Mosókapszula (30 db)", "Öblítő 1,5 l",
                               "Mosogatószer 1 l", "Mosogatógép-tabletta (40 db)",
                               "Általános tisztítószer 750 ml", "Fürdőszobai tisztító 500 ml",
                               "WC-tisztító gél 750 ml"],
    "Babaápolás": ["Pelenka 4-es méret (50 db)", "Pelenka 5-ös méret (44 db)",
                   "Popsitörlő (3×64 db)", "Babasampon 300 ml", "Popsikrém 100 ml",
                   "Babafürdető 400 ml"],
    "Szájhigiénia": ["Fogkrém fehérítő 75 ml", "Fogkrém érzékeny fogra 75 ml",
                     "Fogkefe közepes", "Szájvíz 500 ml", "Fogselyem 50 m",
                     "Elektromos fogkefe pótfej (2 db)", "Gyermek fogkrém 50 ml"],
}

products = []
pid = 1
for cat, (season, price_level, _) in CATEGORIES.items():
    for name in PRODUCT_NAMES[cat]:
        price = round(price_level * rng.lognormal(0, 0.35), -1)
        products.append({
            "sku": f"SKU{pid:03d}",
            "product_name": name,
            "category": cat,
            "unit_price": price,
            "unit_cost": round(price * rng.uniform(0.58, 0.72), -1),
            "lead_time_weeks": int(rng.choice([1, 2, 3], p=[0.3, 0.5, 0.2])),
            # rejtett paraméterek a generáláshoz (nem kerülnek a CSV-be)
            "x_base": float(rng.lognormal(np.log(9), 0.6)),
            "x_growth": float(rng.normal(0.06, 0.06)),
            "x_elasticity": float(rng.uniform(2.0, 5.0)),
        })
        pid += 1
prod = pd.DataFrame(products)

# --------------------------------------------------------------------------
# Naptár: munkaszüneti napok
# --------------------------------------------------------------------------
EASTER = {2023: date(2023, 4, 9), 2024: date(2024, 3, 31),
          2025: date(2025, 4, 20), 2026: date(2026, 4, 5)}
holidays = []
for y in range(2023, 2027):
    e = EASTER[y]
    holidays += [
        (date(y, 1, 1), "Újév"), (date(y, 3, 15), "Nemzeti ünnep"),
        (e - timedelta(days=2), "Nagypéntek"), (e + timedelta(days=1), "Húsvéthétfő"),
        (date(y, 5, 1), "A munka ünnepe"), (e + timedelta(days=50), "Pünkösdhétfő"),
        (date(y, 8, 20), "Államalapítás ünnepe"), (date(y, 10, 23), "Nemzeti ünnep"),
        (date(y, 11, 1), "Mindenszentek"), (date(y, 12, 24), "Szenteste"),
        (date(y, 12, 25), "Karácsony"), (date(y, 12, 26), "Karácsony"),
    ]
hol = pd.DataFrame(holidays, columns=["date", "holiday_name"])
hol["date"] = pd.to_datetime(hol["date"])
hol_set = set(hol["date"])

# --------------------------------------------------------------------------
# Akciós naptár: termékenként évi ~5 akció, hétfőn kezdődik, 1 vagy 2 hétig tart
# --------------------------------------------------------------------------
promo_rows = []
mondays = pd.date_range(START, PROMO_PLAN_END, freq="W-MON")
for p in prod.itertuples():
    busy = np.zeros(len(mondays), dtype=bool)
    n_promos = rng.poisson(5 * len(mondays) / 52)
    for _ in range(n_promos):
        for _attempt in range(20):
            i = int(rng.integers(0, len(mondays) - 2))
            weeks = int(rng.choice([1, 2], p=[0.6, 0.4]))
            # akciók között legalább 3 hét szünet
            if not busy[max(0, i - 3): i + weeks + 3].any():
                busy[i: i + weeks] = True
                promo_rows.append({
                    "sku": p.sku,
                    "start_date": mondays[i],
                    "end_date": mondays[i] + timedelta(days=7 * weeks - 1),
                    "discount_pct": float(rng.choice([0.10, 0.15, 0.20, 0.25, 0.30, 0.33])),
                })
                break
promos = pd.DataFrame(promo_rows).sort_values(["start_date", "sku"]).reset_index(drop=True)
promos.insert(0, "promo_id", [f"PR{i:04d}" for i in range(1, len(promos) + 1)])

# --------------------------------------------------------------------------
# Napi kereslet
# --------------------------------------------------------------------------
days = pd.date_range(START, END, freq="D")
n_days = len(days)
doy = days.dayofyear.to_numpy()
years_since = (days - pd.Timestamp(START)).days.to_numpy() / 365.25
WEEKDAY = np.array([1.0, 0.95, 0.95, 1.0, 1.2, 1.3, 0.6])  # hétfő..vasárnap
weekday_mult = WEEKDAY[days.weekday]

is_hol = days.isin(hol_set)
pre_hol = pd.DatetimeIndex(days + pd.Timedelta(days=1)).isin(hol_set) & ~is_hol
holiday_mult = np.where(is_hol, 0.15, np.where(pre_hol, 1.5, 1.0))

# havi szorzók simítása: hónap közepén a megadott érték, köztük lineáris átmenet
mid_doy = np.array([15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349])


def seasonal_curve(monthly):
    x = np.concatenate([mid_doy - 365, mid_doy, mid_doy + 365])
    y = np.tile(monthly, 3)
    return np.interp(doy, x, y)


day_index = {d: i for i, d in enumerate(days)}
sales = []
for p in prod.itertuples():
    season, _, stockable = CATEGORIES[p.category]
    mu = (p.x_base * seasonal_curve(np.array(season)) * weekday_mult * holiday_mult
          * (1 + p.x_growth) ** years_since)

    promo_mult = np.ones(n_days)
    for pr in promos[promos["sku"] == p.sku].itertuples():
        s, e = pr.start_date, pr.end_date
        if s > days[-1]:
            continue
        i0, i1 = day_index.get(s, None), day_index.get(min(e, days[-1]), None)
        if i0 is None:
            continue
        promo_mult[i0: i1 + 1] *= np.exp(p.x_elasticity * pr.discount_pct)
        # akció utáni visszaesés a felhalmozható termékeknél (a vevők előre vásároltak)
        if stockable:
            promo_mult[i1 + 1: i1 + 15] *= 0.8
    mu = mu * promo_mult

    r = 4.0  # negatív binomiális szóródás
    qty = rng.negative_binomial(r, r / (r + mu))
    sales.append(pd.DataFrame({"date": days, "sku": p.sku, "quantity": qty}))

daily = pd.concat(sales, ignore_index=True)

# --------------------------------------------------------------------------
# Mentés
# --------------------------------------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
prod[[c for c in prod.columns if not c.startswith("x_")]].to_csv(OUT_DIR / "products.csv", index=False)
daily.to_csv(OUT_DIR / "daily_sales.csv", index=False, date_format="%Y-%m-%d")
promos.to_csv(OUT_DIR / "promotions.csv", index=False, date_format="%Y-%m-%d")
hol.sort_values("date").to_csv(OUT_DIR / "holidays.csv", index=False, date_format="%Y-%m-%d")

print(f"Termékek: {len(prod)}, napi sorok: {len(daily):,}, akciók: {len(promos)}")
print(f"Időszak: {START} – {END}, összes eladott darab: {daily['quantity'].sum():,}")
