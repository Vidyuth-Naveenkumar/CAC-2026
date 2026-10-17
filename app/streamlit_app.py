from pathlib import Path

import pandas as pd
import streamlit as st

df = pd.read_parquet(Path(__file__).parent / "data" / "risk_scores_app.parquet")
st.write(f"Loaded {len(df)} intersections")
