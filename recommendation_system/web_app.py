from flask import Flask, request, render_template_string
import csv
import math

app = Flask(__name__)

def load_movies():
    movies = []
    with open('data/movies.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            movies.append({
                'id': row['movieId'],
                'title': row['title'],
                'genres': row['genres'],
                'overview': row['overview']
            })
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
    dot = 0
    norm1 = 0
    norm2 = 0
    for w in words:
        v1 = vec1.get(w, 0)
        v2 = vec2.get(w, 0)
        dot += v1 * v2
        norm1 += v1 * v1
        norm2 += v2 * v2
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot / (math.sqrt(norm1) * math.sqrt(norm2))

tfidf_vectors = compute_tfidf(movies)

def recommend_movies(title, top_n=5):
    idx = None
    for i, m in enumerate(movies):
        if m['title'].lower() == title.lower():
            idx = i
            break
    if idx is None:
        return []
    
    sim_scores = []
    for i, vec in enumerate(tfidf_vectors):
        if i == idx:
            continue
        sim = cosine_sim(tfidf_vectors[idx], vec)
        sim_scores.append((i, sim))
    
    sim_scores.sort(key=lambda x: x[1], reverse=True)
    top_indices = [i for i, _ in sim_scores[:top_n]]
    results = [(movies[i]['title'], round(sim_scores[j][1], 3)) for j, i in enumerate(top_indices)]
    return results

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Movie Recommender</title>
    <style>
        body { font-family: Arial; margin: 40px; background: #f5f5f5; }
        .container { max-width: 600px; margin: auto; background: white; padding: 20px; border-radius: 10px; }
        select, input, button { padding: 8px; margin: 5px; width: 100%; }
        button { background: #007bff; color: white; border: none; cursor: pointer; }
        .rec-item { background: #e9ecef; margin: 5px; padding: 10px; border-radius: 5px; }
    </style>
</head>
<body>
<div class="container">
    <h2>🎬 Movie Recommendation System</h2>
    <form method="POST">
        <label>Select a movie you like:</label>
        <select name="movie">
            {% for movie in movies %}
                <option value="{{ movie.title }}">{{ movie.title }}</option>
            {% endfor %}
        </select>
        <br>
        <label>Number of recommendations:</label>
        <input type="number" name="top_n" value="5" min="1" max="10">
        <br>
        <button type="submit">Recommend</button>
    </form>
    {% if recommendations %}
        <h3>Movies similar to "{{ selected_movie }}":</h3>
        {% for title, score in recommendations %}
            <div class="rec-item">{{ title }} (similarity: {{ score }})</div>
        {% endfor %}
    {% endif %}
</div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        selected = request.form['movie']
        top_n = int(request.form.get('top_n', 5))
        recs = recommend_movies(selected, top_n)
        return render_template_string(HTML_TEMPLATE, movies=movies, recommendations=recs, selected_movie=selected)
    return render_template_string(HTML_TEMPLATE, movies=movies, recommendations=None, selected_movie=None)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
