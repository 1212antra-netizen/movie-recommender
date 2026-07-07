"""
Movie Recommender Pro — Hybrid content-based recommendation system
with TMDB dataset expansion, genre filtering, and Netflix-style UI.
"""

from __future__ import annotations

import ast
import concurrent.futures
import json
import os
import re
import warnings
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
MIN_DATASET_SIZE = 10_000          # Target size when manually expanding
DATA_CACHE = APP_DIR / "expanded_movies.csv"
RAW_DATA_CACHE = APP_DIR / "expanded_movies_raw.csv"
LOCAL_MOVIES = APP_DIR / "movies.csv"
LOCAL_CREDITS = APP_DIR / "movies2.csv"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
NUM_RECOMMENDATIONS = 5

PLACEHOLDER_POSTER = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' width='300' height='450' viewBox='0 0 300 450'>"
    "<rect width='100%' height='100%' fill='%23181818'/>"
    "<rect x='10' y='10' width='280' height='430' fill='none' stroke='%23333333'"
    " stroke-width='2' stroke-dasharray='6,6'/>"
    "<circle cx='150' cy='180' r='50' fill='%23E50914' opacity='0.8'/>"
    "<text x='50%' y='280' font-family='system-ui,sans-serif' font-weight='700'"
    " font-size='18' fill='%23FFFFFF' text-anchor='middle'>NO POSTER</text>"
    "</svg>"
)

TITLE_STOP_WORDS = frozenset(
    {"the", "a", "of", "and", "in", "on", "to", "for", "with", "at", "by", "from", "an", "part", "ii", "iii", "iv", "2", "3", "4"}
)

# ---------------------------------------------------------------------------
# Page config & styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Movie Recommender Pro",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
.stApp { background-color: #111111; color: #FFFFFF; }
.header-container {
    text-align: center; padding: 28px 0 18px;
    background: linear-gradient(180deg, rgba(229,9,20,0.15) 0%, rgba(17,17,17,0) 100%);
    border-radius: 12px; margin-bottom: 20px;
}
.app-title { color: #E50914; font-size: 2.8rem; font-weight: 800; margin: 0; letter-spacing: -0.5px; }
.app-subtitle { color: #aaaaaa; font-size: 1.05rem; font-weight: 300; margin-top: 6px; }
.stSlider label, .stSelectbox label, .stMultiSelect label, .stRadio label {
    color: #E50914 !important; font-weight: 600 !important;
}
div.stButton > button:first-child {
    background-color: #E50914 !important; color: white !important;
    font-size: 1.05rem !important; font-weight: 700 !important;
    border-radius: 8px !important; border: none !important;
    padding: 12px 24px !important; width: 100% !important;
    box-shadow: 0 4px 15px rgba(229, 9, 20, 0.35) !important;
}
div.stButton > button:first-child:hover {
    background-color: #ff1f2f !important; transform: translateY(-2px);
}
.movie-card {
    background: #181818; border-radius: 10px; overflow: hidden;
    border: 1px solid #2a2a2a; text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.movie-card:hover {
    transform: translateY(-8px) scale(1.02);
    box-shadow: 0 12px 28px rgba(229,9,20,0.35); border-color: #E50914;
}
.movie-poster {
    width: 100%; aspect-ratio: 2/3; object-fit: cover;
    display: block; border-bottom: 3px solid #E50914; background: #222;
}
.movie-info { padding: 10px 8px; }
.movie-title {
    color: #fff; font-size: 0.92rem; font-weight: 700;
    min-height: 44px; line-height: 1.25;
    display: flex; align-items: center; justify-content: center;
}
.movie-meta {
    display: flex; justify-content: space-between; font-size: 0.78rem;
    color: #8c8c8c; margin-top: 4px;
}
.movie-rating { color: #ffb400; font-weight: 600; }
.category-header {
    border-left: 5px solid #E50914; padding-left: 12px;
    margin: 28px 0 16px; font-weight: 700; font-size: 1.4rem;
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def get_api_key() -> str | None:
    """Resolve TMDB API key from Streamlit secrets or environment."""
    try:
        if "TMDB_API_KEY" in st.secrets:
            return st.secrets["TMDB_API_KEY"]
    except Exception:
        pass
    return os.environ.get("TMDB_API_KEY")


def stem_word(word: str) -> str:
    """Lightweight suffix stemmer for tag normalization."""
    if len(word) <= 3:
        return word
    for suffix in ("ing", "ly", "ed"):
        if word.endswith(suffix):
            return word[: -len(suffix)]
    if word.endswith("es") and not word.endswith(("aes", "ees")):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "as")):
        return word[:-1]
    return word


def clean_tag_list(items: list[str]) -> list[str]:
    return [item.replace(" ", "") for item in items if item]


def parse_json_names(text: Any, name_key: str = "name", limit: int | None = None) -> list[str]:
    """Parse TMDB/Kaggle JSON list fields into plain name lists."""
    if isinstance(text, list):
        data = text
    elif isinstance(text, str) and text.strip():
        try:
            data = ast.literal_eval(text)
        except Exception:
            try:
                data = json.loads(text)
            except Exception:
                return []
    else:
        return []

    names = [item[name_key] for item in data if isinstance(item, dict) and name_key in item]
    return names[:limit] if limit else names


def parse_directors(text: Any) -> list[str]:
    if isinstance(text, list):
        data = text
    elif isinstance(text, str) and text.strip():
        try:
            data = ast.literal_eval(text)
        except Exception:
            try:
                data = json.loads(text)
            except Exception:
                return []
    else:
        return []
    return [item["name"] for item in data if isinstance(item, dict) and item.get("job") == "Director"]


def normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", title.lower().strip())


def is_franchise_duplicate(title_a: str, title_b: str) -> bool:
    """Detect franchise/sequel overlap to improve recommendation diversity."""
    a = normalize_title(title_a).replace(":", "").replace("-", "")
    b = normalize_title(title_b).replace(":", "").replace("-", "")

    if a == b or (len(a) > 4 and len(b) > 4 and (a.startswith(b) or b.startswith(a))):
        return True

    common = set(a.split()) & set(b.split()) - TITLE_STOP_WORDS
    return len(common) >= 2


def hybrid_score(similarity: float, popularity: float, rating: float) -> float:
    """Hybrid ranking: 60% similarity + 20% popularity + 20% rating."""
    pop_norm = min(float(popularity) / 100.0, 1.0)
    rating_norm = min(float(rating) / 10.0, 1.0)
    return (similarity * 0.6) + (pop_norm * 0.2) + (rating_norm * 0.2)


# ---------------------------------------------------------------------------
# Data loading & TMDB expansion
# ---------------------------------------------------------------------------


def _load_local_base() -> pd.DataFrame:
    """Load and merge local movies.csv (+ movies2.csv credits when needed)."""
    if not LOCAL_MOVIES.exists():
        raise FileNotFoundError(
            f"Missing dataset. Commit `{DATA_CACHE.name}` to your repo for deployment, "
            f"or place `{LOCAL_MOVIES.name}` locally for development."
        )

    movies = pd.read_csv(LOCAL_MOVIES, on_bad_lines="skip")
    required = {"movie_id", "title", "overview", "genres", "keywords", "cast", "crew", "vote_average", "popularity"}

    if required.issubset(movies.columns):
        base = movies[list(required)].copy()
        base["movie_id"] = pd.to_numeric(base["movie_id"], errors="coerce")
    else:
        if not LOCAL_CREDITS.exists():
            raise FileNotFoundError(f"Missing credits file: {LOCAL_CREDITS.name}")
        credits = pd.read_csv(LOCAL_CREDITS, on_bad_lines="skip")
        movies["id"] = pd.to_numeric(movies["id"], errors="coerce")
        movies = movies.dropna(subset=["id"]).astype({"id": int})
        credits["movie_id"] = pd.to_numeric(credits["movie_id"], errors="coerce")
        credits = credits.dropna(subset=["movie_id"]).astype({"movie_id": int})
        credits = credits.drop_duplicates(subset=["movie_id"])
        merged = movies.merge(credits, left_on="id", right_on="movie_id", how="inner")
        base = merged[
            ["movie_id", "title", "overview", "genres", "keywords", "cast", "crew", "vote_average", "popularity"]
        ].copy()

    base = base.dropna(subset=["movie_id"]).drop_duplicates(subset=["movie_id"])
    base["movie_id"] = base["movie_id"].astype(int)
    return base


def _fetch_tmdb_page(url: str, params: dict, page: int) -> list[dict]:
    params = {**params, "page": page}
    try:
        response = requests.get(url, params=params, timeout=8)
        if response.status_code == 429:
            time.sleep(2)
            response = requests.get(url, params=params, timeout=8)
        if response.status_code != 200:
            return []
        return response.json().get("results", [])
    except Exception:
        return []


def _collect_movie_ids(api_key: str, max_pages: int = 40) -> dict[int, float]:
    """Discover movie IDs from popular, top-rated, and genre endpoints."""
    discovered: dict[int, float] = {}
    endpoints = [
        ("https://api.themoviedb.org/3/movie/popular", {}),
        ("https://api.themoviedb.org/3/movie/top_rated", {}),
    ]

    for url, extra in endpoints:
        for page in range(1, max_pages + 1):
            results = _fetch_tmdb_page(url, {"api_key": api_key, **extra}, page)
            if not results:
                break
            for item in results:
                mid = item.get("id")
                if mid:
                    discovered[int(mid)] = max(discovered.get(int(mid), 0.0), float(item.get("popularity", 0)))

    try:
        genre_resp = requests.get(
            f"https://api.themoviedb.org/3/genre/movie/list?api_key={api_key}",
            timeout=8,
        )
        genres = genre_resp.json().get("genres", []) if genre_resp.status_code == 200 else []
    except Exception:
        genres = []

    for genre in genres:
        for page in range(1, 25):
            results = _fetch_tmdb_page(
                "https://api.themoviedb.org/3/discover/movie",
                {"api_key": api_key, "with_genres": genre["id"], "sort_by": "popularity.desc"},
                page,
            )
            if not results:
                break
            for item in results:
                mid = item.get("id")
                if mid:
                    discovered[int(mid)] = max(discovered.get(int(mid), 0.0), float(item.get("popularity", 0)))

    return discovered


def _fetch_movie_details(movie_id: int, api_key: str) -> dict | None:
    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {"api_key": api_key, "append_to_response": "credits,keywords"}
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=8)
            if response.status_code == 200:
                return response.json()
            if response.status_code == 429:
                time.sleep(1.5 + attempt)
        except Exception:
            time.sleep(1)
    return None


def fetch_tmdb_data(base_df: pd.DataFrame, api_key: str, target_size: int = MIN_DATASET_SIZE) -> pd.DataFrame:
    """
    Expand the local dataset using TMDB API.
    Fetches popular, top-rated, and genre-discover lists across multiple pages.
    """
    existing_ids = set(base_df["movie_id"].tolist())
    discovered = _collect_movie_ids(api_key, max_pages=50)

    new_ids = [mid for mid in discovered if mid not in existing_ids]
    new_ids.sort(key=lambda mid: discovered[mid], reverse=True)

    needed = max(0, target_size - len(base_df))
    ids_to_fetch = new_ids[: max(needed + 500, 5500)]

    rows: list[dict] = []
    progress = st.progress(0.0, text="Fetching movies from TMDB…")

    def _worker(mid: int) -> dict | None:
        data = _fetch_movie_details(mid, api_key)
        if not data:
            return None
        return {
            "movie_id": int(data["id"]),
            "title": data.get("title", ""),
            "overview": data.get("overview", ""),
            "genres": json.dumps(data.get("genres", [])),
            "keywords": json.dumps(data.get("keywords", {}).get("keywords", [])),
            "cast": json.dumps(data.get("credits", {}).get("cast", [])),
            "crew": json.dumps(data.get("credits", {}).get("crew", [])),
            "vote_average": float(data.get("vote_average", 0.0)),
            "popularity": float(data.get("popularity", 0.0)),
        }

    completed = 0
    total = len(ids_to_fetch)
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(_worker, mid): mid for mid in ids_to_fetch}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                rows.append(result)
            completed += 1
            if completed % 25 == 0 or completed == total:
                progress.progress(completed / max(total, 1), text=f"Fetched {completed}/{total} movies…")

    progress.empty()

    if not rows:
        return base_df

    api_df = pd.DataFrame(rows)
    combined = pd.concat([base_df, api_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["movie_id"], keep="first")
    return combined


# ---------------------------------------------------------------------------
# Preprocessing & similarity
# ---------------------------------------------------------------------------


def preprocess(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values, parse genres to lists, and build weighted tags."""
    df = raw_df.copy()

    df["overview"] = df["overview"].fillna("")
    df["genres"] = df["genres"].fillna("[]")
    df["keywords"] = df["keywords"].fillna("[]")
    df["cast"] = df["cast"].fillna("[]")
    df["crew"] = df["crew"].fillna("[]")
    df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0.0)
    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0.0)

    genres_list = df["genres"].apply(lambda x: parse_json_names(x))
    keywords_list = df["keywords"].apply(lambda x: parse_json_names(x))
    cast_list = df["cast"].apply(lambda x: parse_json_names(x, limit=3))
    crew_list = df["crew"].apply(parse_directors)

    genres_clean = genres_list.apply(clean_tag_list)
    keywords_clean = keywords_list.apply(clean_tag_list)
    cast_clean = cast_list.apply(clean_tag_list)
    crew_clean = crew_list.apply(clean_tag_list)

    overview_tokens = df["overview"].apply(lambda x: str(x).split())
    tag_tokens = overview_tokens + (genres_clean * 3) + keywords_clean + cast_clean + crew_clean

    def build_tags(tokens: list[str]) -> str:
        return " ".join(stem_word(w.lower()) for w in tokens)

    processed = pd.DataFrame(
        {
            "movie_id": df["movie_id"].astype(int),
            "title": df["title"].fillna("Unknown"),
            "overview": df["overview"],
            "genres": genres_list,
            "keywords": keywords_list,
            "cast": cast_list,
            "crew": crew_list,
            "vote_average": df["vote_average"],
            "popularity": df["popularity"],
            "tags": tag_tokens.apply(build_tags),
            "title_lower": df["title"].fillna("").str.lower(),
        }
    )
    return processed.reset_index(drop=True)


@st.cache_resource(show_spinner="Building similarity matrix…")
def build_similarity(df: pd.DataFrame) -> tuple[TfidfVectorizer, Any]:
    """Compute TF-IDF vectors and cosine similarity matrix."""
    vectorizer = TfidfVectorizer(max_features=10_000, stop_words="english")
    matrix = vectorizer.fit_transform(df["tags"].fillna(""))
    similarity = cosine_similarity(matrix)
    return vectorizer, similarity


# ---------------------------------------------------------------------------
# Recommendation engine
# ---------------------------------------------------------------------------


def _genre_match(movie_genres: list[str], selected: list[str], mode: str) -> bool:
    if not selected:
        return True
    movie_set = {g.lower() for g in movie_genres}
    selected_set = {g.lower() for g in selected}
    if mode.lower() == "all":
        return selected_set.issubset(movie_set)
    return bool(movie_set & selected_set)


def _apply_diversity(candidates: pd.DataFrame, num: int) -> list[dict]:
    """Greedy selection with franchise and genre diversity penalties."""
    selected: list[dict] = []
    selected_titles: list[str] = []
    genre_counts: dict[str, int] = {}
    pool = candidates.sort_values("final_score", ascending=False).copy()

    while len(selected) < num and not pool.empty:
        best_idx = None
        best_score = -np.inf

        for idx, row in pool.iterrows():
            if any(is_franchise_duplicate(row["title"], t) for t in selected_titles):
                continue
            penalty = sum(genre_counts.get(g, 0) * 0.05 for g in row["genres"])
            score = row["final_score"] - penalty
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is None:
            # Relax franchise filter
            row = pool.iloc[0]
            best_idx = pool.index[0]
        else:
            row = pool.loc[best_idx]

        selected.append(
            {
                "movie_id": int(row["movie_id"]),
                "title": row["title"],
                "genres": row["genres"],
                "vote_average": float(row["vote_average"]),
                "popularity": float(row["popularity"]),
                "similarity": float(row["similarity"]),
                "final_score": float(row["final_score"]),
            }
        )
        selected_titles.append(row["title"])
        for g in row["genres"]:
            genre_counts[g] = genre_counts.get(g, 0) + 1
        pool = pool.drop(best_idx)

    return selected


def recommend_by_genre(
    df: pd.DataFrame,
    genres: list[str],
    mode: str = "any",
    min_rating: float = 0.0,
    num: int = NUM_RECOMMENDATIONS,
) -> list[dict]:
    """Return top movies matching genre filter using popularity + rating hybrid score."""
    if not genres:
        return []

    mask = df["genres"].apply(lambda g: _genre_match(g, genres, mode))
    candidates = df[mask & (df["vote_average"] >= min_rating)].copy()
    if candidates.empty:
        return []

    candidates["similarity"] = 0.0
    candidates["final_score"] = candidates.apply(
        lambda r: hybrid_score(0.0, r["popularity"], r["vote_average"]), axis=1
    )
    return _apply_diversity(candidates, num)


def recommend(
    movie_name: str,
    df: pd.DataFrame,
    similarity_matrix: np.ndarray,
    genres: list[str] | None = None,
    genre_mode: str = "any",
    min_rating: float = 0.0,
    num: int = NUM_RECOMMENDATIONS,
) -> list[dict]:
    """Content-based recommendations with optional genre filter and hybrid ranking."""
    matches = df[df["title_lower"] == movie_name.lower()]
    if matches.empty:
        return []

    idx = matches.index[0]
    sim_scores = similarity_matrix[idx]

    candidates = df.copy()
    candidates["similarity"] = sim_scores
    candidates["final_score"] = candidates.apply(
        lambda r: hybrid_score(r["similarity"], r["popularity"], r["vote_average"]), axis=1
    )
    candidates = candidates.drop(index=idx)
    candidates = candidates[candidates["vote_average"] >= min_rating]

    if genres:
        candidates = candidates[candidates["genres"].apply(lambda g: _genre_match(g, genres, genre_mode))]

    if candidates.empty:
        return []

    return _apply_diversity(candidates, num)


def combined_recommend(
    movie_name: str | None,
    df: pd.DataFrame,
    similarity_matrix: np.ndarray,
    genres: list[str] | None = None,
    genre_mode: str = "any",
    min_rating: float = 0.0,
    num: int = NUM_RECOMMENDATIONS,
) -> list[dict]:
    """Combined movie + genre recommendation with deduplication."""
    genres = genres or []

    if movie_name:
        recs = recommend(movie_name, df, similarity_matrix, genres, genre_mode, min_rating, num=num * 2)
    elif genres:
        recs = recommend_by_genre(df, genres, genre_mode, min_rating, num=num * 2)
    else:
        return []

    seen: set[int] = set()
    unique: list[dict] = []
    for rec in recs:
        if rec["movie_id"] not in seen:
            seen.add(rec["movie_id"])
            unique.append(rec)
        if len(unique) >= num:
            break
    return unique


# ---------------------------------------------------------------------------
# TMDB poster
# ---------------------------------------------------------------------------


@st.cache_data(ttl=86_400, show_spinner=False)
def fetch_poster(movie_id: int) -> str:
    """Fetch poster URL from TMDB; return placeholder on failure."""
    api_key = get_api_key()
    if not api_key:
        return PLACEHOLDER_POSTER

    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    try:
        response = requests.get(url, params={"api_key": api_key}, timeout=5)
        if response.status_code == 200:
            poster_path = response.json().get("poster_path")
            if poster_path:
                return f"{TMDB_IMAGE_BASE}{poster_path}"
    except Exception:
        pass
    return PLACEHOLDER_POSTER


# ---------------------------------------------------------------------------
# Data orchestration
# ---------------------------------------------------------------------------


def _cached_to_raw(cached: pd.DataFrame) -> pd.DataFrame:
    """Convert processed cache back to raw column format for TMDB merge."""
    df = cached.copy()
    for col in ("genres", "keywords", "cast", "crew"):
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: json.dumps(x) if isinstance(x, list) else (x if isinstance(x, str) else "[]")
            )
    keep = ["movie_id", "title", "overview", "genres", "keywords", "cast", "crew", "vote_average", "popularity"]
    return df[keep]


@st.cache_data(show_spinner="Loading movie database…")
def load_data() -> pd.DataFrame:
    """
    Load processed dataset from bundled cache (deployment) or rebuild from raw/local data.
    """
    if DATA_CACHE.exists():
        try:
            cached = pd.read_csv(DATA_CACHE)
            if len(cached) > 0:
                df = _deserialize_cached(cached)
                if _cache_has_genres(df):
                    return df
        except Exception:
            pass

    if RAW_DATA_CACHE.exists():
        raw = pd.read_csv(RAW_DATA_CACHE)
    else:
        raw = _load_local_base()

    processed = preprocess(raw)
    _serialize_for_cache(processed).to_csv(DATA_CACHE, index=False)
    return processed


def _parse_list_column(value: Any) -> list:
    """Parse list columns from CSV (JSON or Python repr)."""
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.strip() or value.strip() == "[]":
        return []
    text = value.strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        pass
    try:
        parsed = ast.literal_eval(text)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def _serialize_for_cache(df: pd.DataFrame) -> pd.DataFrame:
    """Serialize list columns as JSON before writing CSV."""
    out = df.copy()
    for col in ("genres", "keywords", "cast", "crew"):
        if col in out.columns:
            out[col] = out[col].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)
    return out


def _deserialize_cached(cached: pd.DataFrame) -> pd.DataFrame:
    df = cached.copy()
    for col in ("genres", "keywords", "cast", "crew"):
        if col in df.columns:
            df[col] = df[col].apply(_parse_list_column)
    if "title_lower" not in df.columns:
        df["title_lower"] = df["title"].str.lower()
    return df


def _cache_has_genres(df: pd.DataFrame) -> bool:
    return bool(df["genres"].apply(lambda g: isinstance(g, list) and len(g) > 0).any())


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def render_movie_card(movie: dict) -> None:
    poster = fetch_poster(movie["movie_id"])
    genres_str = ", ".join(movie["genres"][:2]) if movie["genres"] else "—"
    st.markdown(
        f"""
        <div class="movie-card">
            <img class="movie-poster" src="{poster}" alt="{movie['title']}">
            <div class="movie-info">
                <div class="movie-title">{movie['title']}</div>
                <div class="movie-meta">
                    <span>{genres_str}</span>
                    <span class="movie-rating">★ {movie['vote_average']:.1f}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.markdown(
        """
        <div class="header-container">
            <h1 class="app-title">🎬 Movie Recommender Pro</h1>
            <p class="app-subtitle">Hybrid AI recommendations · Genre filtering · TMDB posters</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        df = load_data()
        _, similarity_matrix = build_similarity(df)
    except FileNotFoundError as exc:
        st.error(f"Dataset error: {exc}")
        st.info(
            "For Streamlit Cloud: commit `expanded_movies.csv` to your GitHub repo and redeploy. "
            "For local dev: place `movies.csv` and `movies2.csv` in the project folder."
        )
        return
    except Exception as exc:
        st.error(f"Failed to initialize: {exc}")
        return

    all_genres = sorted({g for genres in df["genres"] for g in genres if g})

    col_movie, col_genre = st.columns(2)
    with col_movie:
        movie_options = ["— Select a movie —"] + sorted(df["title"].unique().tolist())
        selected_movie = st.selectbox("Search movie", movie_options, index=0)

    with col_genre:
        selected_genres = st.multiselect("Filter genres", all_genres)

    genre_mode = "any"
    if selected_genres:
        genre_mode = st.radio(
            "Genre match",
            options=["any", "all"],
            horizontal=True,
            help="'any' = at least one genre · 'all' = must include every selected genre",
        )

    col_rating, col_btn = st.columns([3, 1])
    with col_rating:
        min_rating = st.slider("Minimum rating", 0.0, 10.0, 0.0, 0.5)
    with col_btn:
        st.write("")
        recommend_clicked = st.button("Recommend", use_container_width=True)

    if not recommend_clicked:
        return

    movie_name = None if selected_movie == "— Select a movie —" else selected_movie

    if not movie_name and not selected_genres:
        st.warning("Select a movie and/or at least one genre.")
        return

    if movie_name and movie_name.lower() not in set(df["title_lower"]):
        st.error(f"Movie '{movie_name}' not found in the database.")
        return

    with st.spinner("Generating recommendations…"):
        recs = combined_recommend(
            movie_name=movie_name,
            df=df,
            similarity_matrix=similarity_matrix,
            genres=selected_genres,
            genre_mode=genre_mode,
            min_rating=min_rating,
            num=NUM_RECOMMENDATIONS,
        )

    if not recs:
        st.error("No movies match your filters. Try lowering the rating threshold or changing genres.")
        return

    suffix = f" similar to **{movie_name}**" if movie_name else ""
    genre_suffix = f" · Genres: {', '.join(selected_genres)}" if selected_genres else ""
    st.markdown(f"<div class='category-header'>Top picks for you{suffix}{genre_suffix}</div>", unsafe_allow_html=True)

    columns = st.columns(NUM_RECOMMENDATIONS)
    for col, movie in zip(columns, recs):
        with col:
            render_movie_card(movie)


if __name__ == "__main__":
    main()
