"""Rebuild expanded_movies.csv with proper genre/metadata serialization."""
from pathlib import Path

import pandas as pd

from app import RAW_DATA_CACHE, DATA_CACHE, _load_local_base, _serialize_for_cache, preprocess

if RAW_DATA_CACHE.exists():
    raw = pd.read_csv(RAW_DATA_CACHE)
    print(f"Loading raw dataset: {len(raw):,} movies")
else:
    raw = _load_local_base()
    print(f"Loading local dataset: {len(raw):,} movies")

processed = preprocess(raw)
_serialize_for_cache(processed).to_csv(DATA_CACHE, index=False)

genre_count = processed["genres"].apply(len).gt(0).sum()
sample = processed.iloc[0]["genres"]
print(f"Saved {len(processed):,} movies to {DATA_CACHE.name}")
print(f"Movies with genres: {genre_count:,}")
print(f"Sample genres (Avatar): {sample}")
