import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    file_path = os.path.join(os.path.dirname(__file__), "spotify.csv")
    return pd.read_csv(file_path)

df = load_data()

# -------------------------------
# Show Columns (Debug)
# -------------------------------
st.write("✅ Columns Found:", df.columns.tolist())

# -------------------------------
# Auto Detect Columns
# -------------------------------
def find_column(possible_names):
    for col in df.columns:
        if col.lower().strip() in possible_names:
            return col
    return None

song_col = find_column(["song", "song_1", "track", "track_name", "title"])
artist_col = find_column(["artist", "artists", "user", "user_1", "singer"])
genre_col = find_column(["genre", "genre_1", "category", "type"])

# -------------------------------
# Check Required Columns
# -------------------------------
if song_col is None or artist_col is None or genre_col is None:
    st.error("❌ Dataset columns not matching!")
    st.info("Please rename columns like: song, artist, genre")
    st.stop()

# Rename to standard
df = df.rename(columns={
    song_col: "song",
    artist_col: "artist",
    genre_col: "genre"
})

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Spotify Dashboard", layout="wide")

st.title("🎧 Spotify Recommendation Dashboard")

# -------------------------------
# KPI Cards
# -------------------------------
st.subheader("📌 KPI Overview")

c1, c2, c3 = st.columns(3)

c1.metric("🎵 Total Songs", df["song"].nunique())
c2.metric("👤 Total Users/Artists", df["artist"].nunique())
c3.metric("🎼 Total Genres", df["genre"].nunique())

st.markdown("---")

# -------------------------------
# Top Artists Graph
# -------------------------------
st.subheader("📊 Top Listening Users/Artists")

top_artists = df["artist"].value_counts().head(10)

fig = plt.figure()
plt.bar(top_artists.index, top_artists.values)
plt.xticks(rotation=45)
st.pyplot(fig)

st.markdown("---")

# -------------------------------
# Recommendation Engine
# -------------------------------
st.subheader("🤖 Recommendation System")

df["tags"] = df["artist"].astype(str) + " " + df["genre"].astype(str)

cv = CountVectorizer(max_features=5000)
vectors = cv.fit_transform(df["tags"]).toarray()

similarity = cosine_similarity(vectors)

def recommend(song_name):
    index = df[df["song"] == song_name].index[0]
    distances = sorted(list(enumerate(similarity[index])),
                       reverse=True,
                       key=lambda x: x[1])

    rec = []
    for i in distances[1:6]:
        rec.append(df.iloc[i[0]]["song"])
    return rec

selected_song = st.selectbox("🎵 Select Song:", df["song"].unique())

if st.button("Recommend"):
    st.subheader("✅ Recommended Songs:")
    for s in recommend(selected_song):
        st.write("🎧", s)
