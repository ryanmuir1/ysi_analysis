import io
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
# Utility functions
# ------------------------------------------------------------
def read_csv_flex(f):
    for enc in ["utf-8-sig", "latin1", "cp1252", "utf-16"]:
        try:
            return pd.read_csv(f, encoding=enc), enc
        except Exception:
            f.seek(0)
    f.seek(0)
    return pd.read_csv(f, sep=None, engine="python", encoding_errors="replace"), "auto-sep"


def parse_row_col(well_id: str):
    if not isinstance(well_id, str) or not well_id:
        return np.nan, np.nan
    token = well_id
    if "_" in token:
        token = token.split("_")[-1]
    m = re.match(r'^\s*([A-Za-z])\s*0*?(\d+)\s*$', token)
    if not m:
        m = re.search(r'([A-Za-z])\s*0*?(\d+)', token)
    if m:
        row = m.group(1).upper()
        col = int(m.group(2))
        return row, col
    return np.nan, np.nan


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
# Main
# ------------------------------------------------------------
uploaded = st.file_uploader("Upload BioAnalysis CSV", type=["csv"])

if not uploaded:
    st.info("Upload a CSV to begin.")
    st.stop()

df, enc = read_csv_flex(uploaded)
st.caption(f"Detected encoding: {enc} — rows: {df.shape[0]}, cols: {df.shape[1]}")
df = prepare_df(df, active_probes=recipe["active_probes"])
if df is None:
    st.stop()


sample_cols = recipe["sample_columns"]
qc_cols = recipe["qc_columns"]
expected_vals = np.array(recipe["qc_expected_concentrations"], dtype=float)


# ------------------------------------------------------------
# Samples
# ------------------------------------------------------------
samples = df[df["Col"].isin(sample_cols)].copy()

if samples.empty:
    avg_samples = pd.DataFrame()
    st.info("No sample rows found for configured sample columns.")
else:
    avg_samples = (
        samples.groupby(["Row", "Col"], as_index=False)
        .agg(
            Avg_Concentration=("Concentration", "mean"),
            N_Measurements=("Concentration", "count")
        )
    )

    # Map row → beaker
    avg_samples["Beaker"] = avg_samples["Row"].map(recipe["row_to_beaker"])

    # Sort properly by row, then col
    row_order = {chr(ord("A") + i): i for i in range(26)}
    avg_samples["RowIndex"] = avg_samples["Row"].map(row_order)
    avg_samples = avg_samples.sort_values(["RowIndex", "Col"]).reset_index(drop=True)
    avg_samples = avg_samples.drop(columns=["RowIndex"])

st.subheader(f"Averaged Samples (Columns {sample_cols})")
st.dataframe(avg_samples, use_container_width=True)


# ------------------------------------------------------------
# QC
# ------------------------------------------------------------
qc_view = pd.DataFrame()
qc = df[df["Col"].isin(qc_cols)].copy()

if qc.empty:
    st.info("No QC rows for configured QC columns.")
else:
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

    st.subheader(f"QC Measurements (Columns {qc_cols})")
    st.dataframe(qc_view, use_container_width=True)


# ------------------------------------------------------------
# PASS / FAIL banner
# ------------------------------------------------------------
plate_fail = False
if not qc_view.empty:
    plate_fail = (qc_view["Pct_Deviation"].abs() > 5).any()

if plate_fail:
    st.error("PLATE AT RISK — at least one QC point outside ±5 percent.")
else:
    st.success("PLATE ACCEPTED — all QC points within ±5 percent.")


# ------------------------------------------------------------
# QC plot
# ------------------------------------------------------------
st.subheader("QC % Deviation Over Run")

y_min, y_max = st.slider(
    "Y-axis range",
    min_value=-50.0,
    max_value=50.0,
    value=(-12.0, 12.0),
    step=0.5
)

fig = None
if not qc_view.empty:
    fig, ax = plt.subplots()
    for exp in sorted(np.unique(qc_view["Expected"].dropna())):
        subset = qc_view[np.isclose(qc_view["Expected"], exp)]
        if not subset.empty:
            ax.plot(subset["RunOrder"], subset["Pct_Deviation"], marker="o", label=f"{exp} mmol/L")

    ax.axhline(0, linestyle="--")
    ax.axhline(2, linestyle=":")
    ax.axhline(5, linestyle=":")
    ax.axhline(-2, linestyle=":")
    ax.axhline(-5, linestyle=":")

    ax.set_xlabel("Run order")
    ax.set_ylabel("% deviation")
    ax.set_ylim([y_min, y_max])
    ax.legend()

    st.pyplot(fig)


# ------------------------------------------------------------
# Package download
# ------------------------------------------------------------
st.subheader("Download Package")

pkg_trigger = st.button("Download Package")

if pkg_trigger:
    code = st.text_input("Enter 3-digit code (e.g. 123):", max_chars=3)
    if code and len(code) == 3 and code.isdigit():

        prefix = f"LN-AX00{code}"
        status = "FAIL" if plate_fail else "PASS"

        mem_zip = BytesIO()
        with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as z:

            # 1. Raw file
            raw_name = f"{prefix}_Raw_{status}.csv"
            z.writestr(raw_name, df.to_csv(index=False))

            # 2. Processed samples
            samp_name = f"{prefix}_Processed_{status}.csv"
            z.writestr(samp_name, avg_samples.to_csv(index=False))

            # 3. QC table
            qc_name = f"{prefix}_QC_{status}.csv"
            z.writestr(qc_name, qc_view.to_csv(index=False))

            # 4. QC Plot PNG
            plot_name = f"{prefix}_QCplot_{status}.png"
            if fig is not None:
                buf = BytesIO()
                fig.savefig(buf, format="png")
                buf.seek(0)
                z.writestr(plot_name, buf.read())

        mem_zip.seek(0)
        st.download_button(
            "Download ZIP Package",
            data=mem_zip.getvalue(),
            file_name=f"{prefix}_PACK_{status}.zip",
            mime="application/zip"
        )

    elif code:
        st.error("Code must be exactly 3 digits.")


