import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load and prepare data (same as before)
@st.cache_data
def load_data():
    df = pd.read_csv('data/movies.csv')
    df['combined'] = df['genres'] + " " + df['overview']
    return df

@st.cache_resource
def build_similarity(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

def recommend_movies(movie_title, df, cosine_sim, top_n=5):
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    if movie_title not in indices:
        return []
    idx = indices[movie_title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    recommendations = df['title'].iloc[movie_indices].tolist()
    scores = [round(s[1], 3) for s in sim_scores]
    return list(zip(recommendations, scores))

# --- Streamlit UI ---
st.set_page_config(page_title="Movie Recommender", page_icon="🎬")
st.title("🎬 Movie Recommendation System")
st.markdown("Select a movie you like, and I'll suggest similar movies.")

# Load data
df = load_data()
cosine_sim = build_similarity(df)

# Dropdown to select a movie
movie_list = df['title'].tolist()
selected_movie = st.selectbox("Choose a movie:", movie_list)

# Number of recommendations
top_n = st.slider("Number of recommendations:", min_value=1, max_value=10, value=5)

if st.button("Recommend"):
    recs = recommend_movies(selected_movie, df, cosine_sim, top_n=top_n)
    if recs:
        st.subheader(f"Movies similar to **{selected_movie}**:")
        for i, (title, score) in enumerate(recs, 1):
            st.write(f"{i}. **{title}**  (similarity: {score})")
    else:
        st.error("Movie not found. Please try another.")