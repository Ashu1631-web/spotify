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
    return pd.read_csv(file_path)

df = load_data()

# Rename first column as user
df.rename(columns={"Unnamed: 0": "user"}, inplace=True)

# Song columns
song_columns = df.columns[1:]

# -------------------------------
# Page Config
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

user_row = df[df["user"] == selected_user]

# Convert values to numeric
user_row[song_columns] = user_row[song_columns].apply(pd.to_numeric, errors="coerce")

# -------------------------------
# KPI Section
# -------------------------------
st.subheader("📌 User KPIs")

total_songs_listened = int((user_row[song_columns] > 0).sum(axis=1).values[0])

total_plays = int(user_row[song_columns].sum(axis=1).values[0])

most_played_song = (
    user_row[song_columns]
    .T.sort_values(by=user_row.index[0], ascending=False)
    .index[0]
)

col1, col2, col3 = st.columns(3)

col1.metric("🎵 Songs Listened", total_songs_listened)
col2.metric("▶ Total Plays", total_plays)
col3.metric("🔥 Most Played Song", most_played_song)

st.markdown("---")

# -------------------------------
# Top 10 Songs Graph
# -------------------------------
st.subheader(f"📊 Top 10 Songs Played by {selected_user}")

user_song_counts = user_row[song_columns].T
user_song_counts.columns = ["plays"]

top_user_songs = user_song_counts.sort_values(by="plays", ascending=False).head(10)

colors = plt.cm.tab20(np.linspace(0, 1, len(top_user_songs)))

fig = plt.figure(figsize=(10, 5))
plt.bar(top_user_songs.index, top_user_songs["plays"], color=colors)
plt.xticks(rotation=45)
plt.xlabel("Songs")
plt.ylabel("Plays")

st.pyplot(fig)

st.markdown("---")

# -------------------------------
# Recommendation Section
# -------------------------------
st.subheader("🤖 Recommended Songs")

def recommend_songs(user_name, n=5):
    user_data = df[df["user"] == user_name]

    listens = user_data[song_columns].T
    listens.columns = ["plays"]

    return listens.sort_values(by="plays", ascending=False).head(n)

if st.button("🎶 Show Recommendations"):
    rec = recommend_songs(selected_user)

    st.subheader("✅ Recommended Songs For You:")

    for song, row in rec.iterrows():
        st.write(f"🎧 {song} → Plays: {int(row['plays'])}")
