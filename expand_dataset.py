import pandas as pd
import ast
import json
import requests
import time
import os
import concurrent.futures

def load_base_dataset():
    print("Loading base CSV datasets...")
    # Load raw CSVs, handle issues
    movies = pd.read_csv("movies.csv", on_bad_lines="skip")
    credits = pd.read_csv("movies2.csv", on_bad_lines="skip")
    
    # Cast and clean IDs
    movies["id"] = pd.to_numeric(movies["id"], errors="coerce")
    movies = movies.dropna(subset=["id"])
    movies["id"] = movies["id"].astype(int)
    
    credits["movie_id"] = pd.to_numeric(credits["movie_id"], errors="coerce")
    credits = credits.dropna(subset=["movie_id"])
    credits["movie_id"] = credits["movie_id"].astype(int)
    
    # Remove credits duplicates
    credits = credits.drop_duplicates(subset=["movie_id"])
    
    # Merge on movie_id
    base_df = movies.merge(credits, left_on="id", right_on="movie_id", suffixes=("", "_credit"))
    
    # Keep necessary columns
    keep_cols = ["movie_id", "title", "overview", "genres", "keywords", "cast", "crew", "vote_average", "popularity"]
    base_df = base_df[keep_cols]
    
    print(f"Base dataset loaded: {len(base_df)} unique movies.")
    return base_df

def get_api_key():
    # Attempt to load from Streamlit secrets, then .streamlit/secrets.toml, then environment
    api_key = None
    try:
        # Check secrets.toml directly to be safe
        if os.path.exists(".streamlit/secrets.toml"):
            with open(".streamlit/secrets.toml", "r") as f:
                for line in f:
                    if "TMDB_API_KEY" in line:
                        api_key = line.split("=")[1].replace('"', "").replace("'", "").strip()
                        break
    except Exception:
        pass
    
    if not api_key:
        api_key = os.getenv("TMDB_API_KEY")
    
    return api_key

def fetch_movie_with_retry(movie_id, api_key, retries=3):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&append_to_response=credits,keywords"
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                time.sleep(1.5 + attempt * 2)
            else:
                time.sleep(0.5)
        except Exception:
            time.sleep(1)
    return None

def fetch_movies_list(endpoint_url, params, max_pages=50):
    movie_list = []
    for page in range(1, max_pages + 1):
        params["page"] = page
        try:
            r = requests.get(endpoint_url, params=params, timeout=5)
            if r.status_code == 200:
                data = r.json()
                results = data.get("results", [])
                if not results:
                    break
                for m in results:
                    if m.get("id"):
                        movie_list.append({
                            "id": int(m["id"]),
                            "popularity": float(m.get("popularity", 0)),
                            "title": m.get("title", "")
                        })
            elif r.status_code == 429:
                time.sleep(2)
                # Retry once
                r = requests.get(endpoint_url, params=params, timeout=5)
                if r.status_code == 200:
                    results = r.json().get("results", [])
                    for m in results:
                        if m.get("id"):
                            movie_list.append({
                                "id": int(m["id"]),
                                "popularity": float(m.get("popularity", 0)),
                                "title": m.get("title", "")
                            })
        except Exception as e:
            print(f"Error fetching page {page} of list: {e}")
            
        if page % 20 == 0:
            print(f"  Fetched page {page}/{max_pages}...")
            
    return movie_list

def collect_new_movie_ids(api_key):
    print("Collecting movie IDs from popular, top rated and genre discover lists...")
    all_discovered = {}
    
    # 1. Fetch Popular Movies
    print("Fetching Popular Movies list...")
    pop_movies = fetch_movies_list("https://api.themoviedb.org/3/movie/popular", {"api_key": api_key}, max_pages=60)
    for m in pop_movies:
        all_discovered[m["id"]] = m["popularity"]
        
    # 2. Fetch Top Rated Movies
    print("Fetching Top Rated Movies list...")
    tr_movies = fetch_movies_list("https://api.themoviedb.org/3/movie/top_rated", {"api_key": api_key}, max_pages=60)
    for m in tr_movies:
        all_discovered[m["id"]] = max(all_discovered.get(m["id"], 0), m["popularity"])
        
    # 3. Fetch Genre Discover Lists
    genres_url = f"https://api.themoviedb.org/3/genre/movie/list?api_key={api_key}"
    try:
        gr = requests.get(genres_url, timeout=5)
        genres = gr.json().get("genres", [])
    except Exception:
        genres = []
        
    print(f"Found {len(genres)} TMDB genres. Fetching discover pages for each...")
    for idx, g in enumerate(genres):
        g_id = g["id"]
        g_name = g["name"]
        print(f"Fetching movies for genre {g_name} ({idx+1}/{len(genres)})...")
        # Discover movies for this genre
        g_movies = fetch_movies_list(
            "https://api.themoviedb.org/3/discover/movie",
            {"api_key": api_key, "with_genres": g_id, "sort_by": "popularity.desc"},
            max_pages=20
        )
        for m in g_movies:
            all_discovered[m["id"]] = max(all_discovered.get(m["id"], 0), m["popularity"])
            
    print(f"Total unique movie IDs discovered from lists: {len(all_discovered)}")
    return all_discovered

def parse_tmdb_genres(genres_list):
    return [g["name"] for g in genres_list if "name" in g] if genres_list else []

def parse_tmdb_keywords(keywords_data):
    if not keywords_data:
        return []
    # Can be a list of keywords or a dict with keywords list (depending on details vs list responses)
    if isinstance(keywords_data, list):
        return [k["name"] for k in keywords_data if "name" in k]
    elif isinstance(keywords_data, dict):
        kw_list = keywords_data.get("keywords", [])
        return [k["name"] for k in kw_list if "name" in k]
    return []

def parse_tmdb_cast(cast_list):
    # Get top 3 actors
    if not cast_list:
        return []
    return [c["name"] for c in cast_list[:3] if "name" in c]

def parse_tmdb_crew(crew_list):
    # Get directors
    if not crew_list:
        return []
    return [c["name"] for c in crew_list if c.get("job") == "Director" and "name" in c]

def clean_tag_list(lst):
    if not lst or not isinstance(lst, list):
        return []
    # Replace spaces inside strings
    return [item.replace(" ", "") for item in lst if item]

def stem_word(word):
    if len(word) <= 3:
        return word
    # Simple Porter-like suffix stripper
    if word.endswith('ing'):
        return word[:-3]
    if word.endswith('ly'):
        return word[:-2]
    if word.endswith('ed'):
        return word[:-2]
    if word.endswith('es') and not word.endswith('aes') and not word.endswith('ees'):
        return word[:-2]
    if word.endswith('s') and not word.endswith('ss') and not word.endswith('us') and not word.endswith('as'):
        return word[:-1]
    return word

def main():
    api_key = get_api_key()
    print(f"Using TMDB API key: {api_key[:4]}...{api_key[-4:]}")
    
    # Load base dataset
    base_df = load_base_dataset()
    base_ids = set(base_df["movie_id"].tolist())
    
    # Collect IDs from lists
    discovered_ids_pop = collect_new_movie_ids(api_key)
    
    # Filter for new IDs only
    new_ids = [mid for mid in discovered_ids_pop.keys() if mid not in base_ids]
    print(f"Found {len(new_ids)} movie IDs not present in the base dataset.")
    
    # Sort new IDs by popularity to fetch the most prominent movies first
    new_ids.sort(key=lambda mid: discovered_ids_pop[mid], reverse=True)
    
    # We want the total dataset to grow to 10,000+ movies
    # We need: 10,200 - len(base_df)
    target_new_count = max(5500, 10200 - len(base_df))
    print(f"Targeting details fetch for the top {target_new_count} new movies...")
    ids_to_fetch = new_ids[:target_new_count]
    
    new_movies_data = []
    
    print(f"Starting parallel fetch for {len(ids_to_fetch)} movies using ThreadPoolExecutor...")
    start_time = time.time()
    
    # Run fetch in parallel using threads
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=25) as executor:
        # Submit tasks
        future_to_id = {executor.submit(fetch_movie_with_retry, mid, api_key): mid for mid in ids_to_fetch}
        for future in concurrent.futures.as_completed(future_to_id):
            mid = future_to_id[future]
            try:
                data = future.result()
                if data:
                    new_movies_data.append(data)
            except Exception as e:
                print(f"Exception fetching details for ID {mid}: {e}")
            
            completed += 1
            if completed % 100 == 0 or completed == len(ids_to_fetch):
                elapsed = time.time() - start_time
                speed = completed / elapsed
                rem_time = (len(ids_to_fetch) - completed) / speed if speed > 0 else 0
                print(f"  Fetched {completed}/{len(ids_to_fetch)} details... Elapsed: {elapsed:.1f}s, Rem: {rem_time:.1f}s")
                
    print(f"Finished details fetch. Successfully retrieved details for {len(new_movies_data)} movies.")
    
    # Build dataframe for new movies
    rows = []
    for m in new_movies_data:
        mid = m.get("id")
        title = m.get("title", "")
        overview = m.get("overview", "")
        vote_average = m.get("vote_average", 0.0)
        popularity = m.get("popularity", 0.0)
        
        # Raw formats matching Kaggle dataset
        genres_raw = json.dumps(m.get("genres", []))
        keywords_raw = json.dumps(m.get("keywords", {}).get("keywords", []))
        cast_raw = json.dumps(m.get("credits", {}).get("cast", []))
        crew_raw = json.dumps(m.get("credits", {}).get("crew", []))
        
        rows.append({
            "movie_id": mid,
            "title": title,
            "overview": overview,
            "genres": genres_raw,
            "keywords": keywords_raw,
            "cast": cast_raw,
            "crew": crew_raw,
            "vote_average": vote_average,
            "popularity": popularity
        })
        
    new_movies_df = pd.DataFrame(rows)
    
    # Combine datasets
    print("Merging new movies with base dataset...")
    combined_raw_df = pd.concat([base_df, new_movies_df], ignore_index=True)
    combined_raw_df = combined_raw_df.drop_duplicates(subset=["movie_id"])
    
    print(f"Total raw dataset size: {len(combined_raw_df)} movies.")
    combined_raw_df.to_csv("expanded_movies_raw.csv", index=False)
    print("Saved raw expanded dataset to 'expanded_movies_raw.csv'.")
    
    # Preprocess the entire raw dataset
    print("Preprocessing the combined dataset...")
    
    # Handle missing values
    combined_raw_df["overview"] = combined_raw_df["overview"].fillna("")
    combined_raw_df["genres"] = combined_raw_df["genres"].fillna("[]")
    combined_raw_df["keywords"] = combined_raw_df["keywords"].fillna("[]")
    combined_raw_df["cast"] = combined_raw_df["cast"].fillna("[]")
    combined_raw_df["crew"] = combined_raw_df["crew"].fillna("[]")
    combined_raw_df["vote_average"] = pd.to_numeric(combined_raw_df["vote_average"], errors="coerce").fillna(0.0)
    combined_raw_df["popularity"] = pd.to_numeric(combined_raw_df["popularity"], errors="coerce").fillna(0.0)
    
    # Parsing functions
    def safe_convert_genres(text):
        try:
            data = ast.literal_eval(text) if isinstance(text, str) else text
            if isinstance(data, list):
                return parse_tmdb_genres(data)
        except Exception:
            try:
                data = json.loads(text)
                if isinstance(data, list):
                    return parse_tmdb_genres(data)
            except Exception:
                pass
        return []

    def safe_convert_keywords(text):
        try:
            data = ast.literal_eval(text) if isinstance(text, str) else text
            if isinstance(data, list):
                return [k["name"] for k in data if "name" in k]
        except Exception:
            try:
                data = json.loads(text)
                if isinstance(data, list):
                    return [k["name"] for k in data if "name" in k]
            except Exception:
                pass
        return []

    def safe_convert_cast(text):
        try:
            data = ast.literal_eval(text) if isinstance(text, str) else text
            if isinstance(data, list):
                return parse_tmdb_cast(data)
        except Exception:
            try:
                data = json.loads(text)
                if isinstance(data, list):
                    return parse_tmdb_cast(data)
            except Exception:
                pass
        return []

    def safe_convert_crew(text):
        try:
            data = ast.literal_eval(text) if isinstance(text, str) else text
            if isinstance(data, list):
                return parse_tmdb_crew(data)
        except Exception:
            try:
                data = json.loads(text)
                if isinstance(data, list):
                    return parse_tmdb_crew(data)
            except Exception:
                pass
        return []
        
    print("  Parsing text fields...")
    genres_parsed = combined_raw_df["genres"].apply(safe_convert_genres)
    keywords_parsed = combined_raw_df["keywords"].apply(safe_convert_keywords)
    cast_parsed = combined_raw_df["cast"].apply(safe_convert_cast)
    crew_parsed = combined_raw_df["crew"].apply(safe_convert_crew)
    
    # Clean spaces
    genres_cleaned = genres_parsed.apply(clean_tag_list)
    keywords_cleaned = keywords_parsed.apply(clean_tag_list)
    cast_cleaned = cast_parsed.apply(clean_tag_list)
    crew_cleaned = crew_parsed.apply(clean_tag_list)
    
    # Tokenize overview
    overview_tokens = combined_raw_df["overview"].apply(lambda x: str(x).split())
    
    # Combine columns into tags
    print("  Combining fields and building weighted tags...")
    tags_series = overview_tokens + (genres_cleaned * 3) + keywords_cleaned + cast_cleaned + crew_cleaned
    
    # Convert tags to lowercase and apply custom stemming
    print("  Applying lowercase and custom suffix stemming...")
    def process_tags(tokens):
        stemmed_tokens = [stem_word(word.lower()) for word in tokens]
        return " ".join(stemmed_tokens)
        
    processed_tags = tags_series.apply(process_tags)
    
    # Build the final processed DataFrame
    processed_df = pd.DataFrame({
        "movie_id": combined_raw_df["movie_id"],
        "title": combined_raw_df["title"],
        "overview": combined_raw_df["overview"],
        "genres": genres_parsed.apply(json.dumps),
        "keywords": keywords_parsed.apply(json.dumps),
        "cast": cast_parsed.apply(json.dumps),
        "crew": crew_parsed.apply(json.dumps),
        "vote_average": combined_raw_df["vote_average"],
        "popularity": combined_raw_df["popularity"],
        "tags": processed_tags,
        "title_lower": combined_raw_df["title"].str.lower()
    })
    
    # Verify dataset size
    print(f"Final processed dataset row count: {len(processed_df)}")
    if len(processed_df) >= 10000:
        print("Success! Dataset is over 10,000 movies.")
    else:
        print(f"Warning: Dataset size is {len(processed_df)}, which is below 10,000.")
        
    processed_df.to_csv("expanded_movies.csv", index=False)
    print("Saved preprocessed expanded dataset to 'expanded_movies.csv'.")
    print("Dynamic dataset expansion and pre-processing completed successfully!")

if __name__ == "__main__":
    main()
