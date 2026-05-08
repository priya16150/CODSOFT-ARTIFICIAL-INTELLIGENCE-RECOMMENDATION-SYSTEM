import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------
# 1. Load and prepare data
# ------------------------------
def load_data(filepath='data/movies.csv'):
    df = pd.read_csv(filepath)
    # Combine genres and overview into a single string for each movie
    df['combined'] = df['genres'] + " " + df['overview']
    return df

# ------------------------------
# 2. Build TF-IDF matrix and similarity
# ------------------------------
def build_similarity_matrix(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

# ------------------------------
# 3. Recommendation function
# ------------------------------
def recommend_movies(movie_title, df, cosine_sim, top_n=5):
    # Get index of the movie
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    
    if movie_title not in indices:
        print(f"Movie '{movie_title}' not found in the dataset.")
        return []
    
    idx = indices[movie_title]
    # Get similarity scores for all movies with this one
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort by similarity (descending) and skip the first (itself)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    # Get movie indices
    movie_indices = [i[0] for i in sim_scores]
    # Return titles and similarity scores
    recommendations = df['title'].iloc[movie_indices].tolist()
    scores = [round(s[1], 3) for s in sim_scores]
    return list(zip(recommendations, scores))

# ------------------------------
# 4. Main - interactive demo
# ------------------------------
if __name__ == "__main__":
    # Load data and build similarity matrix
    df = load_data()
    cosine_sim = build_similarity_matrix(df)
    
    print("\n🎬 **Content-Based Movie Recommender** 🎬\n")
    print("Available movies:")
    print("\n".join(f" - {title}" for title in df['title']))
    
    while True:
        movie = input("\nEnter a movie title you like (or 'quit' to exit): ").strip()
        if movie.lower() == 'quit':
            break
        recs = recommend_movies(movie, df, cosine_sim, top_n=5)
        if recs:
            print(f"\n🔍 Movies similar to '{movie}':")
            for i, (title, score) in enumerate(recs, 1):
                print(f"  {i}. {title} (similarity: {score})")
        else:
            print("Please try another movie from the list.")