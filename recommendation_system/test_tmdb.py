import requests


API_KEY ="94b0c519cb16e7225666f1fed52d6144"

def search_movie(movie_title):
    """Searches for a movie on TMDb and prints its first poster URL."""
    search_url = "https://api.themoviedb.org/3/search/movie"
    params = {
        'api_key': API_KEY,
        'query': movie_title,
        'language': 'en-US'
    }

    response = requests.get(search_url, params=params)
    data = response.json()
    
    print(f"HTTP Status Code: {response.status_code}")

    if response.status_code == 200 and data['results']:
        first_movie = data['results'][0]
        poster_path = first_movie.get('poster_path')
        
        if poster_path:
            poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
            print(f"✅ Movie Found: {first_movie['title']} (Release: {first_movie.get('release_date', 'N/A')})")
            print(f"🎬 Poster URL: {poster_url}")
            return poster_url
        else:
            print("Poster not available for this movie.")
    else:
        print("❌ Movie not found or API error.")

if __name__ == "__main__":
    search_movie("Inception")
