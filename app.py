import io
import csv
import re
from typing import List, Dict
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Well Averages", layout="wide")

st.title("Well Glucose Averager")
st.caption("Upload your exported text/CSV file(s). Each row must look like: "
           "`8/27/2025,10:08,AX173-A01,1A,Glucose,2.93,mmol/L,1B,Glucose,2.87,mmol/L,1.0`")

uploaded_files = st.file_uploader(
    "Drop one or more files here (txt or csv).",
    type=["txt", "csv"],
    accept_multiple_files=True,
    help="Each line must contain batch-well like AX173-A01 and two glucose values for 1A and 1B."
)

def parse_lines(text: str) -> List[Dict]:
    records: List[Dict] = []
    reader = csv.reader(io.StringIO(text))
    for row in reader:
        if not row:
            continue
        # Be forgiving about stray spaces
        row = [cell.strip() for cell in row]
        # Extract batch and well from the 3rd column like "AX173-A01"
        bw = row[2] if len(row) > 2 else ""
        if "-" in bw:
            batch_id, well_id = bw.split("-", 1)
        else:
            m = re.search(r"([A-Za-z0-9]+)-([A-H]\d{2})", ",".join(row))
            if not m:
                continue
            batch_id, well_id = m.group(1), m.group(2)

        # Find the two glucose numeric values.
        def parse_float_safe(x):
            try:
                return float(x)
            except Exception:
                return None

        v1 = parse_float_safe(row[5] if len(row) > 5 else None)
        v2 = parse_float_safe(row[9] if len(row) > 9 else None)

        if v1 is None or v2 is None:
            nums = []
            for i, token in enumerate(row):
                if token.lower() == "glucose" and i+1 < len(row):
                    f = parse_float_safe(row[i+1])
                    if f is not None:
                        nums.append(f)
            if len(nums) >= 2:
                v1, v2 = nums[0], nums[1]
            else:
                floats = [parse_float_safe(t) for t in row]
                floats = [f for f in floats if f is not None]
                if len(floats) >= 2:
                    v1, v2 = floats[0], floats[1]
                else:
                    continue

        avg = (v1 + v2) / 2.0

        row_letter = well_id[0] if well_id else ""
        try:
            col_num = int(well_id[1:])
        except Exception:
            col_num = None

        records.append({
            "batchID": batch_id,
            "wellID": well_id,
            "row": row_letter,
            "col": col_num,
            "avg_concentration_mmol_L": avg
        })
    return records

all_records: List[Dict] = []

if uploaded_files:
    for uf in uploaded_files:
        content = uf.read().decode("utf-8", errors="ignore")
        recs = parse_lines(content)
        all_records.extend(recs)

sample_text = st.text_area(
    "Or paste raw lines here (optional):",
    value="",
    height=150,
    placeholder="Paste lines like: 8/27/2025,10:08,AX173-A01,1A,Glucose,2.93,mmol/L,1B,Glucose,2.87,mmol/L,1.0"
)

if sample_text.strip():
    all_records.extend(parse_lines(sample_text))

if not all_records:
    st.info("Upload a file or paste lines above to see results.")
    st.stop()

df = pd.DataFrame(all_records)

# Sort by row (A..H), then by ascending avg concentration, then by column number.
df.sort_values(by=["row", "avg_concentration_mmol_L", "col"], ascending=[True, True, True], inplace=True)

# Format output
out = df[["batchID", "wellID", "avg_concentration_mmol_L"]].rename(
    columns={"avg_concentration_mmol_L": "average concentration (mmol/L)"}
)
out.reset_index(drop=True, inplace=True)

st.subheader("Averaged Results (sorted by row, then ascending concentration)")
st.dataframe(out, use_container_width=True)

# Row filter
with st.expander("Filter by row letter"):
    rows = sorted(df["row"].dropna().unique().tolist())
    selected = st.multiselect("Rows", rows, default=rows)
    filtered = out[out["wellID"].str[0].isin(selected)] if selected else out
    st.dataframe(filtered, use_container_width=True)

# Download
csv_bytes = out.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", data=csv_bytes, file_name="well_averages.csv", mime="text/csv")
