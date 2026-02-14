import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
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

st.set_page_config(page_title="Spotify User Recommendation", layout="wide")

# -------------------------------
# Title
# -------------------------------
st.title("🎧 Spotify User-Based Recommendation Dashboard")

# -------------------------------
# KPI Cards
# -------------------------------
col1, col2, col3 = st.columns(3)

col1.metric("👤 Total Users", df["user"].nunique())
col2.metric("🎵 Total Songs", df.shape[1] - 1)
col3.metric("📊 Total Records", df.shape[0])

st.markdown("---")

# -------------------------------
# Top Listening Songs (Overall)
# -------------------------------
st.subheader("🔥 Top 10 Most Played Songs Overall")

song_columns = df.columns[1:]  # all song columns

top_songs = df[song_columns].sum().sort_values(ascending=False).head(10)

fig = plt.figure()
plt.bar(top_songs.index, top_songs.values)
plt.xticks(rotation=45)
plt.xlabel("Songs")
plt.ylabel("Total Plays")

st.pyplot(fig)

st.markdown("---")

# -------------------------------
# User Recommendation System
# -------------------------------
st.subheader("🤖 Recommend Songs for a User")

# Select User
user_list = df["user"].unique()
selected_user = st.selectbox("Select User:", user_list)

# Recommend Function
def recommend_songs(user_name, n=5):
    user_row = df[df["user"] == user_name]

    # Get listening history
    listens = user_row[song_columns].T
    listens.columns = ["plays"]

    # Recommend most played songs by that user
    top_user_songs = listens.sort_values(by="plays", ascending=False).head(n)

    return top_user_songs

if st.button("Recommend Songs"):
    st.subheader(f"🎶 Top Recommendations for {selected_user}")

    rec = recommend_songs(selected_user)

    for song, row in rec.iterrows():
        st.write(f"✅ {song}  → Plays: {row['plays']}")
