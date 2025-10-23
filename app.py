
import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Plate Analyzer (Samples + QC)", layout="wide")
st.title("Plate Analyzer — Samples & QC")

uploaded = st.file_uploader("Upload the BioAnalysis CSV", type=["csv"])

def read_csv_flex(f):
    # try several encodings
    for enc in ["utf-8-sig", "latin1", "cp1252", "utf-16"]:
        try:
            return pd.read_csv(f, encoding=enc), enc
        except Exception:
            f.seek(0)
            continue
    f.seek(0)
    # last resort
    return pd.read_csv(f, sep=None, engine="python", encoding_errors="replace"), "auto-sep, replace-errors"

def parse_row_col(well_id: str):
    """
    Handles formats like 'A1', 'A01', 'H12' and 'P96_A01' (ignore the 'P96_' prefix).
    Returns (Row, Col) or (np.nan, np.nan) if unparsable.
    """
    if not isinstance(well_id, str) or not well_id:
        return np.nan, np.nan
    token = well_id
    if "_" in token:
        token = token.split("_")[-1]  # take the last segment after underscore
    m = re.match(r'^\s*([A-Za-z])\s*0*?(\d+)\s*$', token)
    if not m:
        # fallback: letter + digits anywhere
        m = re.search(r'([A-Za-z])\s*0*?(\d+)', token)
    if m:
        row = m.group(1).upper()
        try:
            col = int(m.group(2))
        except Exception:
            col = np.nan
        return row, col
    return np.nan, np.nan

def prepare_df(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    required = ["Well Id", "Probe Id", "Concentration"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        return None

    # Derive Row/Col correctly
    parsed = df["Well Id"].apply(parse_row_col).apply(pd.Series)
    parsed.columns = ["Row", "Col"]
    df = pd.concat([df, parsed], axis=1)
    df["Concentration"] = pd.to_numeric(df["Concentration"], errors="coerce")
    return df

if uploaded:
    df, enc = read_csv_flex(uploaded)
    st.caption(f"Detected encoding: **{enc}** — rows: {df.shape[0]}, cols: {df.shape[1]}")
    df = prepare_df(df)
    if df is not None:
        # ---- Samples (wells 1–9), averaged over probes ----
        samples = df[df["Col"].between(1, 9, inclusive="both")].copy()
        avg_samples = (
            samples.groupby(["Row", "Col"], as_index=False)
            .agg(Avg_Concentration=("Concentration", "mean"),
                 N_Measurements=("Concentration", "count"))
            .sort_values("Avg_Concentration", ascending=True)
            .reset_index(drop=True)
        )
        st.subheader("Averaged Samples (Wells 1–9) — sorted ascending by Avg_Concentration")
        st.dataframe(avg_samples, use_container_width=True)

        # ---- QC (wells 11 & 12), per-probe, in run order ----
        qc = df[df["Col"].isin([11, 12])].copy()

        # order by time if available
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

        expected_vals = np.array([2.8, 11.1, 22.2], dtype=float)
        def closest_expected(x):
            if pd.isna(x):
                return np.nan
            idx = np.abs(expected_vals - x).argmin()
            return float(expected_vals[idx])

        qc["Expected"] = qc["Concentration"].apply(closest_expected)
        qc["Pct_Deviation"] = (qc["Concentration"] - qc["Expected"]) / qc["Expected"] * 100.0

        qc_view = qc[["RunOrder", "Local Completion Time", "Row", "Col", "Well Id", "Probe Id", "Concentration", "Expected", "Pct_Deviation"]].copy()

        st.subheader("QC Measurements (Wells 11 & 12) — in run order")
        st.dataframe(qc_view, use_container_width=True)

        # ---- Metrics: average % deviation per standard ----
        st.subheader("QC Metrics")
        metrics = (qc_view
                   .groupby("Expected")["Pct_Deviation"]
                   .agg(["mean", "count"])
                   .reset_index()
                   .sort_values("Expected"))
        col_metrics = st.columns(len(metrics))
        for i, row in metrics.iterrows():
            with col_metrics[i]:
                st.metric(label=f"{row['Expected']} mmol/L — mean % dev (n={int(row['count'])})",
                          value=f"{row['mean']:.2f}%")

        # ---- Plot controls ----
        st.subheader("QC % Deviation over Run")
        default_range = (-12.0, 12.0)
        y_min, y_max = st.slider("Y-axis range (% deviation)",
                                 min_value=-50.0, max_value=50.0,
                                 value=default_range, step=0.5)

        # ---- Plot ----
        fig, ax = plt.subplots()
        for exp in sorted(expected_vals):
            subset = qc_view[np.isclose(qc_view["Expected"], exp)]
            if not subset.empty:
                ax.plot(subset["RunOrder"], subset["Pct_Deviation"], marker="o", label=f"{exp} mmol/L")

        # Baseline and spec lines
        ax.axhline(0, linestyle="--")           # centerline (0%)
        ax.axhline(2, linestyle=":")            # USL1 +2%
        ax.axhline(5, linestyle=":")            # USL2 +5%
        ax.axhline(-2, linestyle=":")           # LSL1 -2%
        ax.axhline(-5, linestyle=":")           # LSL2 -5%

        ax.set_xlabel("Run order")
        ax.set_ylabel("% deviation from expected")
        ax.set_title("QC deviation over run")
        ax.set_ylim([y_min, y_max])
        ax.legend()
        st.pyplot(fig)

        # Downloads
        st.download_button(
            "Download averaged samples CSV",
            data=avg_samples.to_csv(index=False).encode("utf-8"),
            file_name="averaged_samples_by_well.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download QC table CSV",
            data=qc_view.to_csv(index=False).encode("utf-8"),
            file_name="qc_measurements_with_deviation.csv",
            mime="text/csv",
        )
else:
    st.info("Upload a CSV to begin.")
