"""Build the two-watershed (Lower Hafren, Oak Creek) figures in paper/FiguresR2/.

Inputs : each site's data CSV, figs/df_metrics.csv and per-site PNGs written by
         Plynlimon/Postprocess-UQ.py and OakCreek/Postprocess-UQ.py.
Outputs: Fig-obs.png, Fig-performances.png, Fig-performances-op.png,
         Fig-performances-op-kge.png, Fig-timeseries-best.png,
         Fig-timeseries-best-op.png, Fig-mdn-params.png.

Usage (from the repo root, after running both Postprocess-UQ.py scripts):
    python make_combined_figures.py
"""
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from DiffFlowTransport.utils.plot import plot_performance_traintest_sites, plot_performance_obs_op

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "paper" / "FiguresR2"
SITES = {"Lower Hafren Watershed": ROOT / "Plynlimon", "Oak Creek Watershed": ROOT / "OakCreek"}

MDN_TYPES = ['MDN$_{\\text{LSTM}}$', 'MDN$_{Q}$', 'MDN$_{Q,\\text{LSTM}}$']
GAMMA_LABEL = '$\\Gamma_\\text{dynamic}$'


def performance_figures():
    df_by_site = {
        site: pd.read_csv(d / "figs" / "df_metrics.csv").rename(columns={"model-type": "model_type"})
        for site, d in SITES.items()
    }
    subs = {site: df[(df["varn"] == "C_Q") & (df["with-sm"])] for site, df in df_by_site.items()}
    plot_performance_traintest_sites(subs, MDN_TYPES, GAMMA_LABEL)
    save("Fig-performances.png")

    plot_performance_obs_op(df_by_site, MDN_TYPES + [GAMMA_LABEL], ylims={"nse": (-0.6, 1)})
    save("Fig-performances-op.png")

    plot_performance_obs_op(df_by_site, MDN_TYPES + [GAMMA_LABEL], metrics=("kge", "alpha", "beta", "cc"))
    save("Fig-performances-op-kge.png")


def data_figure():
    """Forcing and tracer data at both sites, with the model periods
    shaded. C_Q is drawn as dots because it is sampled intermittently (mostly NaN)."""
    rc = {"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "serif"],
          "font.size": 15}
    spin, train, test, cyclic = "#e9e9e9", "#fdf1df", "#ddeef5", "#f4f4f4"
    hafren = pd.read_csv(ROOT / "Plynlimon" / "data-agg.csv", index_col=0, parse_dates=True)
    oak = pd.read_csv(ROOT / "OakCreek" / "data_withDiffusion_withSpinup-correctedZeroFlow.csv", index_col=0, parse_dates=True)
    sites = [
        ("(a) Lower Hafren", hafren,
         [("J", "$J$\n[mm/d]"), ("Cl mg/l", "$C_J$\n[mg/L]"), ("Q", "$Q$\n[mm/d]"), ("Q Cl mg/l", "$C_Q$\n[mg/L]"),
          ("ET", "$ET$\n[mm/d]"), ("Ta", "$T_a$\n[$^\\circ$C]"), ("Rn", "$R_n$\n[W/m$^2$]")],
         [("Spin-up", "1983-05-03", "1988-12-31", spin), ("Training", "1989-01-01", "1998-12-31", train),
          ("Test", "1999-01-01", "2008-12-31", test)]),
        ("(b) Oak Creek", oak.assign(Tair=oak["Tair"] - 273.15),
         [("P2", "$J$\n[mm/d]"), ("J", "Rain\n[mm/d]"), ("C_P2", "$C_J$\n[mol/mol]"), ("Q", "$Q$\n[mm/d]"),
          ("C_Q", "$C_Q$\n[mol/mol]"), ("ET", "$ET$\n[mm/d]"), ("Tair", "$T_a$\n[$^\\circ$C]"),
          ("SWIN", "$SW_{in}$\n[W/m$^2$]")],
         [("Cyclic ATS spin-up\n(pre-transient)", oak.index[0], "2012-09-30", cyclic),
          ("Spin-up", "2012-10-01", "2014-12-31", spin), ("Training", "2015-10-01", "2020-09-30", train),
          ("Test", "2021-10-01", "2023-12-28", test)]),
    ]
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(20, 17))
        for subfig, (title, df, panels, periods) in zip(fig.subfigures(1, 2, wspace=0.04), sites):
            axes = subfig.subplots(len(panels), 1, sharex=True)
            for ax, (col, label) in zip(axes, panels):
                for _, start, end, color in periods:
                    ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color=color, zorder=0, lw=0)
                if col in ("Q Cl mg/l", "C_Q"):
                    ax.plot(df.index, df[col], ".", color="k", ms=2)
                else:
                    ax.plot(df.index, df[col], "-", color="k", lw=0.6)
                ax.set_ylabel(label, rotation=0, ha="right", va="center")
            for name, start, end, _ in periods:
                mid = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
                axes[0].text(mid, 1.08, name, transform=axes[0].get_xaxis_transform(), ha="center", va="bottom", fontsize=13)
            axes[-1].set_xlabel("Date")
            subfig.suptitle(title, fontsize=18)
    save("Fig-obs.png")


def save(fname):
    plt.savefig(OUT / fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {OUT / fname}")


def stack_site_pngs(fname, side_by_side=False, label_h=60, gap=60):
    """Place each site's PNG one below the other (default) or side by side,
    each with an (a)/(b) label above its top-left corner."""
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
    except OSError:
        font = ImageFont.load_default()
    imgs = [Image.open(d / "figs" / fname).convert("RGB") for d in SITES.values()]
    if side_by_side:
        width = sum(img.width for img in imgs) + gap * (len(imgs) - 1)
        height = label_h + max(img.height for img in imgs)
    else:
        width = max(img.width for img in imgs)
        height = sum(label_h + img.height for img in imgs)
    combined = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(combined)
    x = y = 0
    for k, img in enumerate(imgs):
        draw.text((x + 10, y + 10), f"({chr(97 + k)})", fill="black", font=font)
        combined.paste(img, (x, y + label_h))
        if side_by_side:
            x += img.width + gap
        else:
            y += label_h + img.height
    combined.save(OUT / fname)
    print(f"Wrote {OUT / fname}")


if __name__ == "__main__":
    data_figure()
    performance_figures()
    for fname in ["Fig-timeseries-best.png", "Fig-timeseries-best-op.png"]:
        stack_site_pngs(fname, side_by_side=True)
    stack_site_pngs("Fig-mdn-params.png")
