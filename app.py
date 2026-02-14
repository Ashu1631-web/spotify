import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    file_path = os.path.join(os.path.dirname(__file__), "spotify.csv")
    df = pd.read_csv(file_path)
    return df

df = load_data()

# Rename first column as user
df.rename(columns={"Unnamed: 0": "user"}, inplace=True)

# All song columns
song_columns = df.columns[1:]

# -------------------------------
# Streamlit Config
# -------------------------------
st.set_page_config(page_title="Spotify Dashboard", layout="wide")

st.title("🎧 Spotify User Listening Dashboard")

# -------------------------------
# Refresh Button
# -------------------------------
if st.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

st.markdown("---")

# -------------------------------
# Select User
# -------------------------------
user_list = df["user"].unique()
selected_user = st.selectbox("👤 Select User:", user_list)

# Filter user row
user_row = df[df["user"] == selected_user]

# Convert values to numeric
user_row[song_columns] = user_row[song_columns].apply(
    pd.to_numeric, errors="coerce"
)

# -------------------------------
# Dynamic KPI Cards
# -------------------------------
st.subheader("📌 User Listening KPIs")

# Total songs listened (non-zero)
total_songs_listened = int((user_row[song_columns] > 0).sum(axis=1).values[0])

# Total plays
total_plays = int(user_row
