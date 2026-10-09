"""
Készlettervezés az előrejelzésekre építve.

1. ABC–XYZ elemzés: ABC = a termék súlya az árbevételben, XYZ = a heti kereslet ingadozása.
2. Biztonsági készlet: z × σ, ahol σ a beszerzési átfutási idő + 1 hét alatti összesített
   előrejelzési hiba szórása a 2024. második félévi validációs időszakban; z = 1,645 (95%).
3. Szimuláció 2025-re (heti felülvizsgálat, "feltöltési szintig rendelés"):
   minden hét elején rendelünk annyit, hogy a készlet + a beérkező rendelések elérjék
   a következő (átfutási idő + 1) hét előrejelzett keresletét + a biztonsági készletet.
   A rendelés az átfutási idő után érkezik; a ki nem szolgálható kereslet elveszett eladás.
   Három szabályt hasonlítunk össze:
     A) 4 hetes mozgóátlag + 1 hétnyi biztonsági készlet (gyakori ökölszabály)
     B) 4 hetes mozgóátlag + statisztikai biztonsági készlet
     C) gépi tanulási előrejelzés + statisztikai biztonsági készlet
4. Rendelési javaslat a 2025. december 29-i hétre a C) szabály szerint.

Előfeltétel: python src/forecast.py
Futtatás:    python src/inventory.py
Kimenet:     results/abc_xyz.csv, results/inventory_simulation.csv, results/reorder_plan.csv
"""

import numpy as np
import pandas as pd

from common import RAW, RESULTS, TEST_START

Z_95 = 1.645
WARMUP_WEEKS = 4  # a szimuláció első heteit nem értékeljük (beállási időszak)

POLICIES = {
    "A": ("4 hetes mozgóátlag + 1 hét biztonsági készlet", "ma4", "rule"),
    "B": ("4 hetes mozgóátlag + statisztikai biztonsági készlet", "ma4", "stat"),
    "C": ("Gépi tanulás + statisztikai biztonsági készlet", "ml", "stat"),
}


def abc_xyz(test: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    weekly = test[test["h"] == 1][["sku", "target_week", "y"]]
    stats = weekly.groupby("sku")["y"].agg(["sum", "mean", "std"]).reset_index()
    stats = stats.merge(products[["sku", "product_name", "category", "unit_price"]], on="sku")
    stats["revenue"] = stats["sum"] * stats["unit_price"]
    stats = stats.sort_values("revenue", ascending=False)
    stats["cum_share"] = stats["revenue"].cumsum() / stats["revenue"].sum()
    stats["abc"] = np.where(stats["cum_share"] <= 0.80, "A", np.where(stats["cum_share"] <= 0.95, "B", "C"))
    stats["cv"] = stats["std"] / stats["mean"]
    stats["xyz"] = np.where(stats["cv"] < 0.5, "X", np.where(stats["cv"] < 1.0, "Y", "Z"))
    out = stats[["sku", "product_name", "category", "revenue", "cum_share", "abc", "cv", "xyz"]]
    return out.rename(columns={"revenue": "arbevetel_2025", "cum_share": "kumulalt_reszarany",
                               "cv": "relativ_szoras"})


def lead_time_errors(val: pd.DataFrame, products: pd.DataFrame, model: str) -> pd.Series:
    """Termékenként az (átfutási idő + 1) heti összesített előrejelzési hiba szórása."""
    lt = products.set_index("sku")["lead_time_weeks"]
    sigmas = {}
    for sku, g in val.groupby("sku"):
        cover = int(lt[sku]) + 1
        g = g[g["h"] <= cover]
        cum = g.groupby("origin_week").agg(n=("h", "size"), err=("y", "sum"), f=(model, "sum"))
        cum = cum[cum["n"] == cover]
        sigmas[sku] = float((cum["err"] - cum["f"]).std())
    return pd.Series(sigmas)


def forecast_lookup(test: pd.DataFrame, model: str) -> dict:
    """{(sku, origin_week): [1, 2, 3, 4 hetes előrejelzés]} – hiányzó horizontot az utolsóval pótolunk."""
    out = {}
    for (sku, origin), g in test.groupby(["sku", "origin_week"]):
        vals = g.sort_values("h")[model].tolist()
        out[(sku, origin)] = vals + [vals[-1]] * (4 - len(vals))
    return out


def simulate(test, products, sigma, model, ss_type):
    weeks = sorted(test.loc[test["target_week"] >= TEST_START, "target_week"].unique())
    demand = test[test["h"] == 1].set_index(["sku", "target_week"])["y"]
    fc = forecast_lookup(test, model)
    rows, end_state = [], {}

    for p in products.itertuples():
        L = int(p.lead_time_weeks)
        d = np.array([demand[(p.sku, wk)] for wk in weeks])
        on_hand = d[: L + 1].sum()  # azonos kezdőkészlet minden szabálynál
        arrivals = np.zeros(len(weeks) + L + 1)
        for i, wk in enumerate(weeks):
            on_hand += arrivals[i]
            f = fc[(p.sku, wk - pd.Timedelta(weeks=1))][: L + 1]
            ss = f[0] if ss_type == "rule" else Z_95 * sigma[p.sku]
            target_level = sum(f) + ss
            position = on_hand + arrivals[i + 1: i + L].sum()
            order = max(0.0, np.ceil(target_level - position))
            arrivals[i + L] += order
            sold = min(on_hand, d[i])
            on_hand -= sold
            rows.append({"sku": p.sku, "week": wk, "demand": d[i], "sold": sold,
                         "lost": d[i] - sold, "end_stock": on_hand, "order_qty": order,
                         "unit_price": p.unit_price, "unit_cost": p.unit_cost})
        end_state[p.sku] = {"on_hand": on_hand, "pipeline": arrivals[len(weeks): len(weeks) + L].sum()}
    return pd.DataFrame(rows), end_state


def summarize(sim: pd.DataFrame, name: str) -> dict:
    first_eval = sorted(sim["week"].unique())[WARMUP_WEEKS]
    s = sim[sim["week"] >= first_eval]
    stock_value = (s["end_stock"] * s["unit_cost"]).groupby(s["week"]).sum()
    return {
        "szabaly": name,
        "kiszolgalasi_szint": round(s["sold"].sum() / s["demand"].sum(), 4),
        "elveszett_eladas_ft": round((s["lost"] * s["unit_price"]).sum()),
        "atlagos_keszletertek_ft": round(stock_value.mean()),
        "hiany_hetek_aranya": round((s["lost"] > 0).mean(), 4),
        "ertekelt_hetek": s["week"].nunique(),
    }


def main() -> None:
    products = pd.read_csv(RAW / "products.csv")
    test = pd.read_csv(RESULTS / "forecasts_test.csv", parse_dates=["origin_week", "target_week"])
    val = pd.read_csv(RESULTS / "forecasts_validation.csv", parse_dates=["origin_week", "target_week"])

    classes = abc_xyz(test, products)
    classes.to_csv(RESULTS / "abc_xyz.csv", index=False, float_format="%.4f")

    sigmas = {m: lead_time_errors(val, products, m) for m in ("ma4", "ml")}
    summary, states = [], {}
    for key, (name, model, ss_type) in POLICIES.items():
        sim, states[key] = simulate(test, products, sigmas[model], model, ss_type)
        summary.append({"kod": key, **summarize(sim, name)})
    summary = pd.DataFrame(summary)
    summary.to_csv(RESULTS / "inventory_simulation.csv", index=False)

    # rendelési javaslat a következő hétre (C szabály)
    nxt = pd.read_csv(RESULTS / "forecast_next_4_weeks.csv", parse_dates=["target_week"])
    plan = []
    for p in products.itertuples():
        L = int(p.lead_time_weeks)
        f = nxt[nxt["sku"] == p.sku].sort_values("h")["forecast"].to_numpy()
        ss = np.ceil(Z_95 * sigmas["ml"][p.sku])
        level = np.ceil(f[: L + 1].sum() + ss)
        st = states["C"][p.sku]
        order = max(0, level - st["on_hand"] - st["pipeline"])
        plan.append({
            "sku": p.sku, "product_name": p.product_name, "category": p.category,
            "atfutasi_ido_het": L,
            **{f"elorejelzes_{i + 1}_het": round(v, 1) for i, v in enumerate(f)},
            "biztonsagi_keszlet": ss, "feltoltesi_szint": level,
            "aktualis_keszlet": st["on_hand"], "beerkezo_rendelesek": st["pipeline"],
            "javasolt_rendeles_db": order, "rendeles_erteke_ft": order * p.unit_cost,
        })
    plan = pd.DataFrame(plan).merge(classes[["sku", "abc", "xyz"]], on="sku")
    plan.to_csv(RESULTS / "reorder_plan.csv", index=False)

    pd.set_option("display.width", 160)
    print(summary.to_string(index=False))
    print("\nABC–XYZ (termékek száma):")
    print(pd.crosstab(classes["abc"], classes["xyz"]))
    print(f"\nRendelési javaslat: {int((plan['javasolt_rendeles_db'] > 0).sum())} termék, "
          f"{plan['rendeles_erteke_ft'].sum() / 1e6:.2f} millió Ft beszerzési értékben")


if __name__ == "__main__":
    main()
