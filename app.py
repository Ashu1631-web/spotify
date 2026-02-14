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

# Rename first column as User
df.rename(columns={"Unnamed: 0": "user"}, inplace=True)

# Song Columns
song_columns = df.columns[1:]

st.set_page_config(page_title="Spotify Dashboard", layout="wide")

# -------------------------------
# Title
# -------------------------------
st.title("🎧 Spotify User Listening Dashboard")

# -------------------------------
# 🔄 Refresh Button (Added Properly)
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

# Convert values to numeric (Fix)
user_row[song_columns] = user_row[song_columns].apply(
    pd.to_numeric, errors="coerce"
)

# -------------------------------
# Dynamic KPI Cards
# -------------------------------
st.subheader("📌 User Listening KPIs")

total_songs_listened = int((user_row[song_columns] > 0).sum(axis=1).values[0])
total_plays = int(user_row[song_columns].sum(axis=1).values[0])

most_played_song = user_row[song_columns].T.sort_values(
    by=user_row.index[0], ascending=False
).index[0]

col1, col2, col3 = st.columns(3)

col1.metric("🎵 Songs Listened", total_songs_listened)
col2.metric("▶ Total Plays", total_plays)
col3.metric("🔥 Most Played Song", most_played_song)

st.markdown("---")

# -------------------------------
# Dynamic Top Songs Graph
# -------------------------------
st.subheader(f"📊 Top 10 Songs Played by {selected_user}")

user_song_counts = user_row[song_columns].T
user_song_counts.columns = ["plays"]

top_user_songs = user_song_counts.sort_values(
    by="plays", ascending=False
).head(10)

# Different Colors for Each Bar
colors = plt.cm.tab20(np.linspace(0, 1, len(top_user_songs)))

fig = plt.figure(figsize=(10, 5))
plt.bar(top_user_songs.index, top_user_songs["plays"], color=colors)

plt.xticks(rotation=45)
plt.xlabel("Songs")
plt.ylabel("Plays")

st.pyplot(fig)

st.markdown("---")

# -------------------------------
# Recommendation System
# -------------------------------
st.subheader("🤖 Recommended Songs For User")

def reco
