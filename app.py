import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import yaml
import os
import zipfile
from io import BytesIO

st.set_page_config(page_title="Plate Analyzer (Samples + QC, Recipe Driven)", layout="wide")
st.title("Plate Analyzer — Samples & QC (Recipe Driven)")


# ------------------------------------------------------------
# Load plate recipes
# ------------------------------------------------------------
def load_recipes(folder="Plate Recipes"):
    recipes = {}
    if not os.path.exists(folder):
        return recipes
    for file in os.listdir(folder):
        if file.lower().endswith((".yml", ".yaml")):
            path = os.path.join(folder, file)
            with open(path, "r") as f:
                cfg = yaml.safe_load(f)
                name = cfg.get("name", file)
                recipes[name] = cfg
    return recipes


recipes = load_recipes()
if not recipes:
    st.error("No recipe files found in 'Plate Recipes'. Add YAML recipe files and reload.")
    st.stop()

recipe_name = st.selectbox("Plate recipe", list(recipes.keys()))
recipe = recipes[recipe_name]


# ------------------------------------------------------------
# CSV utilities
# ------------------------------------------------------------
def read_csv_flex(f):
    for enc in ["utf-8-sig", "latin1", "cp1252", "utf-16"]:
        try:
            return pd.read_csv(f, encoding=enc), enc
        except Exception:
            f.seek(0)
    f.seek(0)
    return pd.read_csv(f, sep=None, engine="python", encoding_errors="replace"), "auto"


def parse_row_col(well_id: str):
    if not isinstance(well_id, str) or not well_id:
        return np.nan, np.nan
    token = well_id.split("_")[-1]
    m = re.match(r"^\s*([A-Za-z])\s*0*?(\d+)\s*$", token)
    if not m:
        m = re.search(r"([A-Za-z])\s*0*?(\d+)", token)
    if not m:
        return np.nan, np.nan
    return m.group(1).upper(), int(m.group(2))


def prepare_df(df, active_probes):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    required = ["Well Id", "Probe Id", "Concentration"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        return None

    df = df[df["Probe Id"].astype(str).isin(active_probes)]
    if df.empty:
        st.error("All rows removed after filtering inactive probes.")
        return None

    parsed = df["Well Id"].apply(parse_row_col).apply(pd.Series)
    parsed.columns = ["Row", "Col"]
    df = pd.concat([df, parsed], axis=1)

    df["Concentration"] = pd.to_numeric(df["Concentration"], errors="coerce")
    return df


# ------------------------------------------------------------
# Figure builder: QC deviation vs Beaker (NO averaging)
# ------------------------------------------------------------
def build_qc_figure(qc_view):
    if qc_view.empty:
        return None

    row_order = {chr(ord("A") + i): i + 1 for i in range(26)}
    plot_df = qc_view.copy()
    plot_df["BeakerIndex"] = plot_df["Row"].map(row_order)

    fig, ax = plt.subplots()

    unique_exp = sorted(plot_df["Expected"].dropna().unique())
    all_beakers = sorted(plot_df["BeakerIndex"].dropna().unique())

    for exp in unique_exp:
        sub = plot_df[np.isclose(plot_df["Expected"], exp)].copy()
        if sub.empty:
            continue

        # preserve run order within expected concentration
        sub = sub.sort_values("RunOrder")

        ax.plot(
            sub["BeakerIndex"],
            sub["Pct_Deviation"],
            marker="o",
            linestyle="-",
            label=f"{exp} mmol/L"
        )

        # annotate every QC point
        for _, r in sub.iterrows():
            ax.annotate(
                f"{r['Pct_Deviation']:.1f}%",
                (r["BeakerIndex"], r["Pct_Deviation"]),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=8,
            )

    ax.set_xticks(all_beakers)
    ax.set_xticklabels([f"Beaker {int(b)}" for b in all_beakers])

    ax.axhline(0, linestyle="--")
    ax.axhline(2, linestyle=":")
    ax.axhline(5, linestyle=":")
    ax.axhline(-2, linestyle=":")
    ax.axhline(-5, linestyle=":")

    ax.set_xlabel("Beaker")
    ax.set_ylabel("% deviation")
    ax.set_title("QC % deviation by beaker")
    ax.set_ylim(-12, 12)

    # summary box unchanged (still uses all raw QC points)
    metrics = (
        qc_view.groupby("Expected")["Pct_Deviation"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("Expected")
    )

    summary_lines = []
    for _, row in metrics.iterrows():
        std_str = "N/A" if pd.isna(row["std"]) else f"{row['std']:.2f}%"
        summary_lines.append(
            f"{row['Expected']:.1f} mmol/L: mean {row['mean']:.2f}%, sd {std_str} (n={int(row['count'])})"
        )

    ax.text(
        0.02,
        0.98,
        "\n".join(summary_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

    ax.legend()
    fig.tight_layout()
    return fig
