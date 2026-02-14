import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Music Recommendation Dashboard",
    page_icon="🎧",
    layout="wide"
)

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("spotify.csv")  # Your dataset file
    return df

df = load_data()

# -------------------------------
# Feature Engineering
# -------------------------------
df["tags"] = df["artist"] + " " + df["genre"]

# Vectorization
cv = CountVectorizer(max_features=5000)
vectors = cv.fit_transform(df["tags"]).toarray()

# Similarity Matrix
similarity = cosine_similarity(vectors)

# -------------------------------
# Recommendation Function
# -------------------------------
def recommend(song_name):
    if song_name not in df["song"].values:
        return []

    index = df[df["song"] == song_name].index[0]
    distances = sorted(
        list(enumerate(similarity[index])),
        reverse=True,
        key=lambda x: x[1]
    )

    recommendations = []
    for i in distances[1:6]:
        recommendations.append(df.iloc[i[0]])

    return recommendations


# -------------------------------
# Sidebar UI
# -------------------------------
st.sidebar.title("🎵 Dashboard Controls")
st.sidebar.write("Select a song and get recommendations instantly!")

song_list = df["song"].values
selected_song = st.sidebar.selectbox("🎧 Choose a Song", song_list)

st.sidebar.markdown("---")
st.sidebar.info("💡 Tip: Select any song to discover similar tracks!")

# -------------------------------
# Main Dashboard UI
# -------------------------------
st.title("🎧 Music Recommendation Engine")
st.markdown(
    """
    <h4 style='color:gray;'>
    Discover songs similar to your favorite tracks with AI-powered recommendations.
    </h4>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# Display Selected Song Details
song_data = df[df["song"] == selected_song].iloc[0]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🎵 Song", song_data["song"])

with col2:
    st.metric("🎤 Artist", song_data["artist"])

with col3:
    st.metric("🎼 Genre", song_data["genre"])

st.markdown("---")

# Recommendation Button
if st.sidebar.button("🚀 Recommend Songs"):

    st.subheader("✨ Recommended Songs For You")

    results = recommend(selected_song)

    if results:

        cols = st.columns(5)

        for idx, song in enumerate(results):
            with cols[idx]:
                st.markdown(
                    f"""
                    <div style="
                        background-color:#1e1e1e;
                        padding:15px;
                        border-radius:15px;
                        text-align:center;
                        box-shadow: 0px 0px 10px rgba(0,0,0,0.3);
                    ">
                        <h4 style="color:white;">🎵 {song['song']}</h4>
                        <p style="color:lightgray;">🎤 {song['artist']}</p>
                        <p style="color:skyblue;">🎼 {song['genre']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    else:
        st.error("❌ Song not found in dataset!")

# Footer
st.markdown("---")
st.markdown(
    "<center>🚀 Built with Streamlit | Music Recommendation Dashboard</center>",
    unsafe_allow_html=True
)
