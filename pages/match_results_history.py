import streamlit as st
import pandas as pd


# ---------- Table of match results ----------
st.write(
    """
### Ladder Match Results History
The following table shows all recorded first-to-8 match results from the Toronto Tennis City Singles ladder.
    """
)

OUTPUT_FILE = "./data/cleaned_ttc_scores.csv"
df = read_data = pd.read_csv(OUTPUT_FILE)
st.dataframe(
    # .sort_values(by="Date", ascending=False)
    df.astype({"Date": "string", "IsDraw": "string"}),
    use_container_width=True,
)
