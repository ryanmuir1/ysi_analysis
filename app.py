
import io
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

def parse_df(df):
    df = df.copy()
    # normalize columns
    df.columns = [c.strip() for c in df.columns]
    # required
    req = ["Well Id", "Probe Id", "Concentration"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        return None
    # derive row & col
    df["Row"] = df["Well Id"].astype(str).str[0].str.upper()
    df["Col"] = (
        df["Well Id"].astype(str).str.extract(r'(\d+)$')[0].astype(float)
    )
    # coerce concentration
    df["Concentration"] = pd.to_numeric(df["Concentration"], errors="coerce")
    return df

if uploaded:
    df, enc = read_csv_flex(uploaded)
    st.caption(f"Detected encoding: **{enc}** — rows: {df.shape[0]}, cols: {df.shape[1]}")
    df = parse_df(df)
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

        # ---- Plot ----
        st.subheader("QC % Deviation over Run")
        fig, ax = plt.subplots()
        for exp in sorted(expected_vals):
            subset = qc_view[np.isclose(qc_view["Expected"], exp)]
            if not subset.empty:
                ax.plot(subset["RunOrder"], subset["Pct_Deviation"], marker="o", label=f"{exp} mmol/L")
        ax.set_xlabel("Run order")
        ax.set_ylabel("% deviation from expected")
        ax.set_title("QC deviation over run")
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
