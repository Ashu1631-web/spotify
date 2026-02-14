import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# -------------------------------
# Load Dataset CSV
# -------------------------------
@st.cache_data
def load_data():
    file_path = os.path.join(os.path.dirname(__file__), "spotify.csv")
    df = pd.read_csv(file_path)
    return df

df = load_data()

# -------------------------------
# Auto Rename Columns (Fix)
# -------------------------------
df.columns = df.columns.str.lower().str.strip()

# Rename if dataset has song_1, user_1, genre_1
if "song_1" in df.columns:
    df.rename(columns={"song_1": "song"}, inplace=True)

if "user_1" in df.columns:
    df.rename(columns={"user_1": "artist"}, inplace=True)

if "genre_1" in df.columns:
    df.rename(columns={"genre_1": "genre"}, inplace=True)

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(page_title="Spotify Dashboard", layout="wide")

# -------------------------------
# Title
# -------------------------------
st.title("🎧 Spotify Recommendation Dashboard")

# -------------------------------
# KPI Cards
# -------------------------------
st.subheader("📌 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

col1.metric("🎵 Total Songs", df["song"].nunique())
col2.metric("👤 Total Users/Artists", df["artist"].nunique())
col3.metric("🎼 Total Genres", df["genre"].nunique())

# Popularity Optional
if "popularity" in df.columns:
    col4.metric("⭐ Avg Popularity", round(df["popularity"].mean(), 2))
else:
    col4.metric("⭐ Popularity", "Not Available")

st.markdown("---")

# -------------------------------
# Top Listening Users/Artists Graph
# -------------------------------
st.subheader("📊 Top 10 Listening Users/Artists")

top_artists = df["artist"].value_counts().head(10)

fig1 = plt.figure()
plt.bar(top_artists.index, top_artists.values)
plt.xticks(rotation=45)
plt.xlabel("User / Artist")
plt.ylabel("Count")

st.pyplot(fig1)

st.markdown("---")

# -------------------------------
# Top Songs Graph
# -------------------------------
st.subheader("🎶 Top 10 Most Played Songs")

top_songs = df["song"].value_counts().head(10)

fig2 = plt.figure()
plt.bar(top_songs.index, top_songs.values)
plt.xticks(rotation=45)
plt.xlabel("Songs")
plt.ylabel("Count")

st.pyplot(fig2)

st.markdown("---")

# -------------------------------
# Recommendation Engine
# -------------------------------
st.subheader("🤖 Music Recommendation System")

# Create Tags
df["tags"] = df["artist"].astype(str) + " " + df["genre"].astype(str)

# Vectorization
cv = CountVectorizer(max_features=5000)
vectors = cv.fit_transform(df["tags"]).toarray()

# Similarity Matrix
similarity = cosine_similarity(vectors)

# Recommend Function
def recommend(song_name):
    if song_name not in df["song"].values:
        return ["❌ Song not found in dataset"]

    index = df[df["song"] == song_name].index[0]

    distances = sorted(
        list(enumerate(similarity[index])),
        reverse=True,
        key=lambda x: x[1]
    )

    recommendations = []
    for i in distances[1:6]:
        recommendations.append(df.iloc[i[0]]["song"])

    return recommendations

# Dropdown UI
song_list = df["song"].unique()
selected_song = st.selectbox("🎵 Select a Song:", song_list)

if st.button("Recommend Songs"):
    st.subheader("✅ Recommended Songs For You:")

    results = recommend(selected_song)
    for song in results:
        st.write("🎧", song)
