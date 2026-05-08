import requests
import csv
import os
import time

API_KEY = "94b0c519cb16e7225666f1fed52d6144"
POSTER_FOLDER = "static/posters"

if not os.path.exists(POSTER_FOLDER):
    os.makedirs(POSTER_FOLDER)
else:
    print(f"Folder '{POSTER_FOLDER}' already exists, will overwrite missing posters.")

def load_movie_titles():
    titles = []
    with open('data/movies.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            titles.append(row['title'])
    return titles

def download_poster(title, retries=3):
    for attempt in range(retries):
        try:
            search_url = "https://api.themoviedb.org/3/search/movie"
            params = {"api_key": API_KEY, "query": title}
            resp = requests.get(search_url, params=params, timeout=10)
            data = resp.json()
            
            if data.get("results") and data["results"][0].get("poster_path"):
                poster_path = data["results"][0]["poster_path"]
                img_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
                
                img_data = requests.get(img_url, timeout=15).content
                safe_title = title.replace("/", "_").replace("\\", "_").replace(":", "_")
                filepath = os.path.join(POSTER_FOLDER, safe_title + ".jpg")
                with open(filepath, 'wb') as f:
                    f.write(img_data)
                print(f"✅ Downloaded: {title}")
                return True
            else:
                print(f"❌ No poster found: {title}")
                return False
        except Exception as e:
            print(f"⚠️ Attempt {attempt+1} failed for {title}: {e}")
            time.sleep(2) 
    print(f"❌ Failed to download {title} after {retries} attempts")
    return False

titles = load_movie_titles()
print(f"Found {len(titles)} movies. Starting download...")
for t in titles:
    download_poster(t)
print("\n🎉 Done. Check 'static/posters' folder.")
