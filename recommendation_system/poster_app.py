from flask import Flask, request, jsonify, render_template_string
import csv
import math
import requests

app = Flask(__name__)

OMDB_API_KEY = "cbf107ba"  

def fetch_poster_url(title):
    """Get poster URL from OMDb API (fast, reliable)"""
    url = f"http://www.omdbapi.com/?t={title}&apikey={OMDB_API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        if data.get("Response") == "True":
            poster = data.get("Poster")
            if poster and poster != "N/A":
                return poster
        return None
    except Exception as e:
        print(f"Error fetching poster for {title}: {e}")
        return None

def load_movies():
    movies = []
    try:
        with open('data/movies.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                movies.append({
                    'id': row['movieId'],
                    'title': row['title'],
                    'genres': row['genres'],
                    'overview': row['overview']
                })
    except FileNotFoundError:
        movies = [
            {'id': '1', 'title': 'The Dark Knight', 'genres': 'Action|Crime|Drama', 'overview': 'Batman fights the Joker.'},
            {'id': '2', 'title': 'Inception', 'genres': 'Action|Sci-Fi|Thriller', 'overview': 'Dream heist.'},
            {'id': '3', 'title': 'The Matrix', 'genres': 'Action|Sci-Fi', 'overview': 'Reality is a simulation.'},
            {'id': '4', 'title': 'Interstellar', 'genres': 'Adventure|Drama|Sci-Fi', 'overview': 'Space travel to save humanity.'},
            {'id': '5', 'title': 'The Godfather', 'genres': 'Crime|Drama', 'overview': 'Mafia family drama.'},
            {'id': '6', 'title': 'Pulp Fiction', 'genres': 'Crime|Drama', 'overview': 'Interlocking crime stories.'},
            {'id': '7', 'title': 'Forrest Gump', 'genres': 'Drama|Romance', 'overview': 'A simple man influences history.'},
            {'id': '8', 'title': 'Titanic', 'genres': 'Drama|Romance', 'overview': 'Love story on a sinking ship.'},
            {'id': '9', 'title': 'Avatar', 'genres': 'Action|Adventure|Sci-Fi', 'overview': 'Alien planet conflict.'},
            {'id': '10', 'title': 'The Avengers', 'genres': 'Action|Adventure|Sci-Fi', 'overview': 'Superheroes assemble.'},
            {'id': '11', 'title': 'Joker', 'genres': 'Crime|Drama|Thriller', 'overview': 'Origin of the Joker.'},
            {'id': '12', 'title': 'Gladiator', 'genres': 'Action|Adventure|Drama', 'overview': 'Revenge in ancient Rome.'},
        ]
    return movies

movies = load_movies()

def get_word_counts(text):
    words = text.lower().split()
    stopwords = {'a', 'an', 'and', 'the', 'of', 'to', 'in', 'for', 'on', 'with', 'by', 'is', 'are', 'was', 'were'}
    words = [w for w in words if w not in stopwords and len(w) > 2]
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    return freq

def compute_tfidf(movies):
    docs = [f"{m['genres']} {m['overview']}" for m in movies]
    word_docs = [get_word_counts(doc) for doc in docs]
    doc_freq = {}
    for wc in word_docs:
        for word in wc:
            doc_freq[word] = doc_freq.get(word, 0) + 1
    N = len(movies)
    tfidf_vectors = []
    for wc in word_docs:
        vec = {}
        for word, tf in wc.items():
            idf = math.log((N + 1) / (doc_freq.get(word, 1) + 1)) + 1
            vec[word] = tf * idf
        tfidf_vectors.append(vec)
    return tfidf_vectors

def cosine_sim(vec1, vec2):
    words = set(vec1.keys()) | set(vec2.keys())
    dot = sum(vec1.get(w,0) * vec2.get(w,0) for w in words)
    norm1 = math.sqrt(sum(v*v for v in vec1.values()))
    norm2 = math.sqrt(sum(v*v for v in vec2.values()))
    return dot/(norm1*norm2) if norm1 and norm2 else 0

tfidf_vectors = compute_tfidf(movies)

def recommend_movies(title, top_n=5):
    try:
        idx = next(i for i,m in enumerate(movies) if m['title'].lower() == title.lower())
    except StopIteration:
        return []
    sim_scores = [(i, cosine_sim(tfidf_vectors[idx], tfidf_vectors[i])) for i in range(len(movies)) if i != idx]
    sim_scores.sort(key=lambda x: x[1], reverse=True)
    results = []
    for i, score in sim_scores[:top_n]:
        poster = fetch_poster_url(movies[i]['title'])
        results.append({
            'title': movies[i]['title'],
            'similarity': round(score, 3),
            'poster': poster if poster else ''
        })
    return results

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Movie Recommender</title>
    <style>
        body { font-family: Arial; background: #1e1e2f; color: white; margin: 0; padding: 20px; }
        h1 { text-align: center; }
        .search-container { text-align: center; margin: 20px; }
        #searchInput {
            width: 60%;
            padding: 12px;
            font-size: 16px;
            border-radius: 30px;
            border: none;
            outline: none;
            text-align: center;
        }
        .grid { display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; margin-top: 30px; }
        .movie-card {
            background: #2d2d44;
            border-radius: 12px;
            width: 180px;
            cursor: pointer;
            transition: 0.2s;
            text-align: center;
            padding: 10px;
        }
        .movie-card:hover { transform: scale(1.03); background: #3d3d5c; }
        .movie-poster {
            width: 150px;
            height: 225px;
            background: #555;
            border-radius: 8px;
            overflow: hidden;
            margin: 0 auto;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .movie-poster img { width: 100%; height: 100%; object-fit: cover; }
        .movie-poster span { color: white; text-align: center; padding: 10px; font-size: 14px; }
        .movie-title { margin-top: 10px; font-size: 14px; font-weight: bold; }
        .rec-section { margin-top: 50px; }
        .rec-grid { display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; }
        button { background: #ff5722; color: white; border: none; padding: 8px 16px; border-radius: 20px; cursor: pointer; margin: 10px; }
        .container { max-width: 1200px; margin: auto; }
        .no-results { text-align: center; width: 100%; margin-top: 50px; font-size: 18px; }
    </style>
</head>
<body>
<div class="container">
    <h1>🎬 Movie Recommender</h1>
    <div class="search-container">
        <input type="text" id="searchInput" placeholder="🔍 Search for a movie..." onkeyup="filterMovies()">
    </div>
    <div class="grid" id="movieGrid"></div>
    <div id="recommendations" class="rec-section" style="display:none">
        <h2>🎯 You might also like:</h2>
        <div class="rec-grid" id="recGrid"></div>
        <button onclick="clearRecs()">Clear recommendations</button>
    </div>
</div>

<script>
let allMovies = {{ movies | tojson }};

function displayMovies(moviesList) {
    let grid = document.getElementById('movieGrid');
    if (!moviesList.length) {
        grid.innerHTML = '<div class="no-results">✨ No movies found. Try a different search.</div>';
        return;
    }
    grid.innerHTML = '';
    moviesList.forEach(movie => {
        let card = document.createElement('div');
        card.className = 'movie-card';
        card.setAttribute('onclick', `getRecommendations('${movie.title.replace(/'/g, "\\'")}')`);
        let posterHtml = movie.poster ? `<img src="${movie.poster}" alt="${movie.title}">` : `<span>${movie.title.slice(0,20)}</span>`;
        card.innerHTML = `
            <div class="movie-poster">${posterHtml}</div>
            <div class="movie-title">${movie.title}</div>
        `;
        grid.appendChild(card);
    });
}

function filterMovies() {
    let query = document.getElementById('searchInput').value.toLowerCase();
    let filtered = allMovies.filter(movie => movie.title.toLowerCase().includes(query));
    displayMovies(filtered);
}

function getRecommendations(title) {
    fetch('/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: title })
    })
    .then(response => response.json())
    .then(data => {
        let recDiv = document.getElementById('recGrid');
        recDiv.innerHTML = '';
        data.forEach(rec => {
            let card = document.createElement('div');
            card.className = 'movie-card';
            let posterHtml = rec.poster ? `<img src="${rec.poster}" alt="${rec.title}">` : `<span>${rec.title.slice(0,20)}</span>`;
            card.innerHTML = `
                <div class="movie-poster">${posterHtml}</div>
                <div class="movie-title">${rec.title}</div>
                <div style="font-size:11px; color:#aaa;">similarity: ${rec.similarity}</div>
            `;
            recDiv.appendChild(card);
        });
        document.getElementById('recommendations').style.display = 'block';
        window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
    });
}

function clearRecs() {
    document.getElementById('recommendations').style.display = 'none';
}

// Initial load
displayMovies(allMovies);
</script>
</body>
</html>
"""

@app.route('/')
def home():
    movies_data = []
    for m in movies:
        poster = fetch_poster_url(m['title'])
        movies_data.append({'title': m['title'], 'poster': poster if poster else ''})
    return render_template_string(HTML_TEMPLATE, movies=movies_data)

@app.route('/recommend', methods=['POST'])
def recommend_api():
    data = request.get_json()
    title = data.get('title')
    recs = recommend_movies(title, top_n=5)
    return jsonify(recs)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
