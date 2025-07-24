import streamlit as st
import pandas as pd
import io

st.title("Well ID Concentration Analyzer")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")

    if "Well Id" not in df.columns or "Concentration" not in df.columns:
        st.error("The CSV must contain 'Well Id' and 'Concentration' columns.")
    else:
        # Step 1: Average concentration per Well Id
        avg_conc = df.groupby("Well Id")["Concentration"].mean().reset_index(name="Avg Concentration")

        # Step 2: Extract 5th character (row letter)
        avg_conc["Row"] = avg_conc["Well Id"].str[4]

        # Step 3: Sort by Row, then Concentration
        sorted_df = avg_conc.sort_values(by=["Row", "Avg Concentration"])

        st.success("Data processed successfully.")
        st.dataframe(sorted_df)

        # Prepare CSV for download
        csv_buffer = io.StringIO()
        sorted_df.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode()

        st.download_button(
            label="📥 Download Processed CSV",
            data=csv_bytes,
            file_name="sorted_concentration_by_row.csv",
            mime="text/csv"
        )
