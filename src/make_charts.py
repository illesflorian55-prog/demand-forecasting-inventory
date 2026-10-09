"""
Diagramok készítése a README-hez.
Előfeltétel: python src/forecast.py és python src/inventory.py

Futtatás:  python src/make_charts.py
Kimenet:   images/*.png
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from common import IMAGES, RAW, RESULTS  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREY = "#a9a8a3"
PROMO = "#f3e3c3"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_2,
    "xtick.color": TEXT_2,
    "ytick.color": TEXT_2,
    "text.color": TEXT,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
})

MODEL_SHORT = {
    "Naiv (utolsó hét)": "Naiv",
    "Szezonális naiv (előző év)": "Szezonális naiv",
    "4 hetes mozgóátlag": "4 hetes mozgóátlag",
    "Gépi tanulás (gradient boosting)": "Gépi tanulás",
}


def title(ax, main: str, sub: str) -> None:
    ax.set_title(main, loc="left", fontsize=13, fontweight="bold", pad=24)
    ax.text(0, 1.02, sub, transform=ax.transAxes, color=TEXT_2, fontsize=9.5)


def pct(x: float) -> str:
    return f"{x * 100:.0f}%".replace(".", ",")


def forecast_vs_actual() -> None:
    test = pd.read_csv(RESULTS / "forecasts_test.csv", parse_dates=["target_week"])
    products = pd.read_csv(RAW / "products.csv").set_index("sku")
    d1 = test[test["h"] == 1]

    skus = ["SKU015", "SKU024"]  # szezonális termék és gyakran akciózott termék
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.2), sharex=True)
    for ax, sku in zip(axes, skus):
        d = d1[d1["sku"] == sku].sort_values("target_week")
        for wk in d.loc[d["promo_days"] > 0, "target_week"]:
            ax.axvspan(wk, wk + pd.Timedelta(days=7), color=PROMO, linewidth=0)
        ax.plot(d["target_week"], d["y"], color=TEXT, linewidth=1.6, label="Tényleges eladás")
        ax.plot(d["target_week"], d["ma4"], color=GREY, linewidth=1.6, linestyle="--",
                label="4 hetes mozgóátlag")
        ax.plot(d["target_week"], d["ml"], color=BLUE, linewidth=2, label="Gépi tanulás")
        ax.set_ylabel("Eladott darab / hét")
        ax.set_ylim(0, d[["y", "ma4", "ml"]].max().max() * 1.12)
        ax.text(0.005, 0.93, f"{products.loc[sku, 'product_name']} ({products.loc[sku, 'category']})",
                transform=ax.transAxes, fontweight="bold", fontsize=10)
    axes[0].legend(frameon=False, loc="upper right", ncol=3)
    axes[1].text(0.995, 0.93, "sárga sáv = akciós hét", transform=axes[1].transAxes,
                 ha="right", color=TEXT_2, fontsize=9)
    title(axes[0], "Előrejelzés és tény 2025-ben (1 héttel előre)",
          "A mozgóátlag késve követi a szezont és az akciókat, a modell előre számol velük")
    fig.tight_layout()
    fig.savefig(IMAGES / "forecast_vs_actual.png", dpi=150)
    plt.close(fig)


def model_comparison() -> None:
    m = pd.read_csv(RESULTS / "model_metrics.csv")
    m["modell_rovid"] = m["modell"].map(MODEL_SHORT)
    order = ["Naiv", "Szezonális naiv", "4 hetes mozgóátlag", "Gépi tanulás"]
    segments = [("Nem akciós hetek", GREY), ("Akciós hetek", ORANGE), ("Összes hét", BLUE)]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    height = 0.26
    for j, (seg, color) in enumerate(segments):
        s = m[m["szegmens"] == seg].set_index("modell_rovid").loc[order]
        ys = [i + (j - 1) * height for i in range(len(order))]
        bars = ax.barh(ys, s["wape"] * 100, height=height * 0.92, color=color, label=seg)
        for bar in bars:
            ax.text(bar.get_width() + 0.6, bar.get_y() + bar.get_height() / 2,
                    f"{bar.get_width():.0f}%", va="center", fontsize=8.5, color=TEXT)
    ax.set_yticks(range(len(order)), order)
    ax.invert_yaxis()
    ax.set_xlabel("Előrejelzési hiba – WAPE (%), kisebb = jobb")
    ax.set_xlim(0, m["wape"].max() * 100 * 1.15)
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower right")

    total = m[m["szegmens"] == "Összes hét"].set_index("modell_rovid")["wape"]
    better = 1 - total["Gépi tanulás"] / total["4 hetes mozgóátlag"]
    title(ax, "Előrejelzési hiba módszerenként (2025, 1–4 héttel előre)",
          f"A gépi tanulási modell hibája {pct(better)}-kal kisebb a 4 hetes mozgóátlagénál")
    fig.tight_layout()
    fig.savefig(IMAGES / "model_comparison.png", dpi=150)
    plt.close(fig)


def feature_importance() -> None:
    fi = pd.read_csv(RESULTS / "feature_importance.csv").head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.barh(fi["jellemzo_hu"], fi["fontossag_mae"], color=BLUE, height=0.6)
    ax.set_xlabel("Ennyivel nő az átlagos abszolút hiba (db/hét), ha a jellemzőt összekeverjük")
    ax.grid(axis="y", visible=False)
    title(ax, "Mire támaszkodik a modell? (permutációs fontosság)",
          "Az alapszint után az akciós naptár a legfontosabb információ")
    fig.tight_layout()
    fig.savefig(IMAGES / "feature_importance.png", dpi=150)
    plt.close(fig)


def inventory_policies() -> None:
    s = pd.read_csv(RESULTS / "inventory_simulation.csv")
    labels = ["A) Mozgóátlag\n+ 1 hét\nbiztonsági készlet", "B) Mozgóátlag\n+ statisztikai\nbiztonsági készlet",
              "C) Gépi tanulás\n+ statisztikai\nbiztonsági készlet"]
    colors = [GREY, GREY, BLUE]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
    b1 = ax1.bar(labels, s["kiszolgalasi_szint"] * 100, color=colors, width=0.6)
    for bar in b1:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{bar.get_height():.1f}%".replace(".", ","), ha="center", fontsize=10, fontweight="bold")
    ax1.set_ylim(80, 100)
    ax1.set_ylabel("Kiszolgálási szint (%)")
    ax1.set_title("Kiszolgált kereslet aránya", loc="left", fontsize=11, fontweight="bold")

    b2 = ax2.bar(labels, s["atlagos_keszletertek_ft"] / 1e6, color=colors, width=0.6)
    for bar in b2:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                 f"{bar.get_height():.1f} M Ft".replace(".", ","), ha="center", fontsize=10, fontweight="bold")
    ax2.set_ylim(0, s["atlagos_keszletertek_ft"].max() / 1e6 * 1.15)
    ax2.set_ylabel("Átlagos készletérték (millió Ft, beszerzési áron)")
    ax2.set_title("Lekötött készlet", loc="left", fontsize=11, fontweight="bold")
    for ax in (ax1, ax2):
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", labelsize=8.5)

    a, b, c = (s.set_index("kod").loc[k] for k in "ABC")
    less_stock = 1 - c["atlagos_keszletertek_ft"] / a["atlagos_keszletertek_ft"]
    more_service = (c["kiszolgalasi_szint"] - a["kiszolgalasi_szint"]) * 100
    fig.suptitle("Készletszimuláció 2025-re: három rendelési szabály", x=0.01, y=0.985, ha="left",
                 fontsize=13, fontweight="bold")
    fig.text(0.01, 0.905, f"A gépi tanulásra épülő szabály {pct(less_stock)}-kal kevesebb készlettel "
             f"{more_service:.1f} százalékponttal magasabb kiszolgálási szintet ér el, mint az ökölszabály".replace(".", ","),
             ha="left", color=TEXT_2, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(IMAGES / "inventory_policies.png", dpi=150)
    plt.close(fig)


def main() -> None:
    IMAGES.mkdir(exist_ok=True)
    forecast_vs_actual()
    model_comparison()
    feature_importance()
    inventory_policies()
    print("Diagramok elkészültek az images/ mappában.")


if __name__ == "__main__":
    main()
