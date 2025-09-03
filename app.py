import os
import pickle
import gdown
import streamlit as st
import pandas as pd
import numpy as np

# ---------------------------
# Google Drive file ID for similarity.pkl
# ---------------------------
FILE_ID = "1RjvW0G9u6tnNZ4SWK8Y4N4YkEQyzLNfn"
SIMILARITY_FILE = "similarity.pkl"

# ---------------------------
# Download similarity.pkl if missing
# ---------------------------
def download_similarity():
    if not os.path.exists(SIMILARITY_FILE):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        try:
            with st.spinner("⬇️ Downloading similarity matrix from Google Drive..."):
                gdown.download(url, SIMILARITY_FILE, quiet=False)
        except Exception as e:
            st.error(f"Failed to download similarity.pkl: {e}")
            st.stop()

@st.cache_resource
def load_similarity():
    download_similarity()
    with open(SIMILARITY_FILE, "rb") as f:
        return pickle.load(f)

# ---------------------------
# Load movies data (from repo)
# ---------------------------
movie_dict = pickle.load(open('movies.pkl', 'rb'))
movies = pd.DataFrame(movie_dict)

# ---------------------------
# Load similarity
# ---------------------------
similarity = load_similarity()

# ---------------------------
# Prepare for recommendations
# ---------------------------
# Case 1: similarity is a dict of movie -> list of recommended movies
if isinstance(similarity, dict):
    def recommend(movie):
        return similarity.get(movie, [])

# Case 2: similarity is a 2D NumPy array
elif isinstance(similarity, (np.ndarray, list)):
    if isinstance(similarity, list):
        similarity = np.array(similarity)
    movie_index = {title: idx for idx, title in enumerate(movies['title'].values)}

    def recommend(movie):
        idx = movie_index.get(movie)
        if idx is None:
            return []
        distances = similarity[idx]
        recommended_indices = distances.argsort()[::-1][1:6]  # top 5 excluding self
        return movies['title'].iloc[recommended_indices].values.tolist()

else:
    st.error("Unknown format for similarity.pkl!")
    st.stop()

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🎬 Movie Recommender System")
st.markdown("""
### Welcome to my Movie Recommender System  
Created as part of a Data Analytics & AI/ML project to enhance user experience through personalized recommendations.
""")

selected_movie_name = st.selectbox('Select a movie:', movies['title'].values)

if st.button('Recommend'):
    recommendations = recommend(selected_movie_name)
    if not recommendations:
        st.warning("No recommendations found for this movie.")
    else:
        st.markdown("#### Recommended Movies:")
        for i in recommendations:
            st.write(i)

# ---------------------------
# Footer
# ---------------------------
st.markdown("""
---
**Developed by V.Shreyan**  
📧 shreyansharma76@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/vshreyan) | [GitHub](https://github.com/shreyan-77)
""")
