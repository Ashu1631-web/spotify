import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# -------------------------------
# Load XLS Dataset Correctly
# -------------------------------
@st.cache_data
def load_data():
    file_path = os.path.join(os.path.dirname(__file__), "spotify.xls")
    df = pd.read_excel(file_path, engine="xlrd")  # ✅ FIX for .xls
    return df

df = load_data()

st.set_page_config(page_title="Spotify Recommendation Dashboard", layout="wide")

# -------------------------------
# Title
# -------------------------------
st.title("🎧 Spotify Recommendation Dashboard")

# -------------------------------
# KPI Cards
# -------------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("🎵 Total Songs", df["song"].nunique())
col2.metric("🎤 Total Artists", df["artist"].nunique())
col3.metric("🎼 Total Genres", df["genre"].nunique())

if "popularity" in df.columns:
    col4.metric("⭐ Avg Popularity", round(df["popularity"].mean(), 2))
else:
    col4.metric("⭐ Popularity", "Not Available")

st.markdown("---")

# -------------------------------
# Top Artists Graph
# -------------------------------
st.subheader("📊 Top 10 Artists (Most Songs)")

top_artists = df["artist"].value_counts().head(10)

fig = plt.figure()
plt.bar(top_artists.index, top_artists.values)
plt.xticks(rotation=45)
plt.xlabel("Artist")
plt.ylabel("Number of Songs")

st.pyplot(fig)

st.markdown("---")

# -------------------------------
# Recommendation Engine
# -------------------------------
st.subheader("🎶 Music Recommendation System")

df["tags"] = df["artist"] + " " + df["genre"]

cv = CountVectorizer(max_features=5000)
vectors = cv.fit_transform(df["tags"]).toarray()

similarity = cosine_similarity(vectors)

def recommend(song_name):
    if song_name not in df["song"].values:
        return ["❌ Song not found in dataset"]

    index = df[df["song"] == song_name].index[0]
    distances = sorted(list(enumerate(similarity[index])),
                       reverse=True,
                       key=lambda x: x[1])

    recommendations = []
    for i in distances[1:6]:
        recommendations.append(df.iloc[i[0]]["song"])

    return recommendations

song_list = df["song"].values
selected_song = st.selectbox("Select a Song:", song_list)

if st.button("Recommend Songs"):
    st.subheader("✅ Recommended Songs For You:")

    results = recommend(selected_song)
    for song in results:
        st.write("🎵", song)
