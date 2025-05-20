import pickle

import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer

from config import TMDB_API_KEY


def fetch_tmdb_movies(api_key, pages=5):
    movies = []
    base_url = "https://api.themoviedb.org/3/movie/popular"
    for page in range(1, pages + 1):
        url = f"{base_url}?api_key={api_key}&page={page}"
        response = requests.get(url).json()
        if "results" in response:
            movies.extend(response["results"])

    df = pd.DataFrame(movies)[
        ["id", "title", "genre_ids", "overview", "release_date", "poster_path"]
    ]
    df.to_csv("data/raw/movies.csv", index=False)
    return df


def preprocess_data(df):
    df["content"] = df["overview"].fillna("") + " " + df["genre_ids"].astype(str)
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["content"])
    with open("data/processed/processed_data.pkl", "wb") as f:
        pickle.dump((tfidf_matrix, tfidf), f)
    return tfidf_matrix, tfidf


if __name__ == "__main__":
    movies_df = fetch_tmdb_movies(TMDB_API_KEY, pages=5)
    preprocess_data(movies_df)
