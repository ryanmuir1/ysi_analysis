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
# Figure builder: QC deviation vs Beaker
# ------------------------------------------------------------
def build_qc_figure(qc_view):
    if qc_view.empty:
        return None

    # Map rows A,B,C... to beaker index 1,2,3...
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

        # preserve run order so lines make sense
        sub = sub.sort_values("RunOrder")

        ax.plot(
            sub["BeakerIndex"],
            sub["Pct_Deviation"],
            marker="o",
            linestyle="-",
            label=f"{exp} mmol/L"
        )

        # label every QC measurement (no averaging)
        for _, r in sub.iterrows():
            ax.annotate(
                f"{r['Pct_Deviation']:.1f}%",
                (r["BeakerIndex"], r["Pct_Deviation"]),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=8,
            )

    # X axis as beakers
    ax.set_xticks(all_beakers)
    ax.set_xticklabels([f"Beaker {int(b)}" for b in all_beakers])

    # Spec lines
    ax.axhline(0, linestyle="--")
    ax.axhline(2, linestyle=":")
    ax.axhline(5, linestyle=":")
    ax.axhline(-2, linestyle=":")
    ax.axhline(-5, linestyle=":")

    ax.set_xlabel("Beaker")
    ax.set_ylabel("% deviation")
    ax.set_title("QC % deviation by beaker")
    ax.set_ylim(-12, 12)

    # Summary box (still uses all raw QC points)
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



# ------------------------------------------------------------
# Upload
# ------------------------------------------------------------
uploaded = st.file_uploader("Upload BioAnalysis CSV", type=["csv"])
if not uploaded:
    st.info("Upload a CSV to begin.")
    st.stop()

df_raw, enc = read_csv_flex(uploaded)
df = prepare_df(df_raw, active_probes=recipe["active_probes"])
if df is None:
    st.stop()

sample_cols = recipe["sample_columns"]
qc_cols = recipe["qc_columns"]
expected_vals = np.array(recipe["qc_expected_concentrations"], dtype=float)


# ------------------------------------------------------------
# QC calculations (run order only used for table now)
# ------------------------------------------------------------
qc = df[df["Col"].isin(qc_cols)].copy()
qc_view = pd.DataFrame()
plate_fail = False

if not qc.empty:
    if "Local Completion Time" in qc.columns:
        try:
            qc["_time"] = pd.to_datetime(qc["Local Completion Time"])
        except Exception:
            qc["_time"] = pd.NaT
    else:
        qc["_time"] = pd.NaT

    if qc["_time"].isna().all():
        qc["RunOrder"] = np.arange(1, len(qc) + 1)
    else:
        qc = qc.sort_values("_time").reset_index(drop=True)
        qc["RunOrder"] = np.arange(1, len(qc) + 1)

    def closest_expected(x):
        if pd.isna(x):
            return np.nan
        idx = np.abs(expected_vals - x).argmin()
        return float(expected_vals[idx])

    qc["Expected"] = qc["Concentration"].apply(closest_expected)
    qc["Pct_Deviation"] = (qc["Concentration"] - qc["Expected"]) / qc["Expected"] * 100

    qc_view = qc[[
        "RunOrder", "Local Completion Time", "Row", "Col", "Well Id",
        "Probe Id", "Concentration", "Expected", "Pct_Deviation"
    ]]

    plate_fail = (qc_view["Pct_Deviation"].abs() > 5).any()


# ------------------------------------------------------------
# Sample averaging
# ------------------------------------------------------------
samples = df[df["Col"].isin(sample_cols)].copy()
if samples.empty:
    avg_samples = pd.DataFrame()
else:
    avg_samples = (
        samples.groupby(["Row", "Col"], as_index=False)
        .agg(
            Avg_Concentration=("Concentration", "mean"),
            N_Measurements=("Concentration", "count")
        )
    )

    # map row -> beaker name from YAML
    avg_samples["Beaker"] = avg_samples["Row"].map(recipe["row_to_beaker"])

    row_order = {chr(ord("A") + i): i for i in range(26)}
    avg_samples["RowIndex"] = avg_samples["Row"].map(row_order)
    avg_samples = avg_samples.sort_values(["RowIndex", "Col"]).reset_index(drop=True)
    avg_samples = avg_samples.drop(columns=["RowIndex"])


# ------------------------------------------------------------
# Build figure (beaker-based)
# ------------------------------------------------------------
fig = build_qc_figure(qc_view)


# ------------------------------------------------------------
# TOP: banner + download package
# ------------------------------------------------------------
if plate_fail:
    st.error("PLATE AT RISK — at least one QC point outside ±5 percent.")
else:
    st.success("PLATE ACCEPTED — all QC points within ±5 percent.")

st.subheader("Download Package")

code = st.text_input("Enter 3-digit lot code (for LN-AX00XXX)", max_chars=3)
valid_code = bool(code) and code.isdigit() and len(code) == 3

if code and not valid_code:
    st.error("Code must be exactly 3 digits (e.g. 123).")

if valid_code and not qc_view.empty and not avg_samples.empty and fig is not None:
    prefix = f"LN-AX00{code}"
    status = "FAIL" if plate_fail else "PASS"

    mem_zip = BytesIO()
    with zipfile.ZipFile(mem_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        # 1) raw data (as uploaded; safer to use df_raw)
        raw_name = f"{prefix}_Raw_{status}.csv"
        z.writestr(raw_name, df_raw.to_csv(index=False))

        # 2) processed / averaged samples
        proc_name = f"{prefix}_Processed_{status}.csv"
        z.writestr(proc_name, avg_samples.to_csv(index=False))

        # 3) QC table
        qc_name = f"{prefix}_QC_{status}.csv"
        z.writestr(qc_name, qc_view.to_csv(index=False))

        # 4) QC plot PNG (beaker-based)
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        plot_name = f"{prefix}_QCplot_{status}.png"
        z.writestr(plot_name, buf.read())

    mem_zip.seek(0)

    st.download_button(
        "Download Package",
        data=mem_zip.getvalue(),
        file_name=f"{prefix}_PACK_{status}.zip",
        mime="application/zip",
    )
else:
    st.download_button(
        "Download Package",
        data=b"",
        file_name="package.zip",
        mime="application/zip",
        disabled=True,
    )


# ------------------------------------------------------------
# Plot + tables
# ------------------------------------------------------------
st.subheader("QC % deviation by beaker")
if fig is not None:
    st.pyplot(fig)
else:
    st.info("No QC data available to plot.")

st.subheader("Averaged samples")
st.dataframe(avg_samples, use_container_width=True)

st.subheader("QC measurements (raw)")
st.dataframe(qc_view, use_container_width=True)
