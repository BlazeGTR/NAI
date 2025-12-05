"""
Movie Recommendation System using User Clustering (KMeans)
Autor: Błażej Majchrzak

Opis:
    Program implementuje prosty system rekomendacji filmów oparty na klastrowaniu
    użytkowników według ich podobieństwa ocen.
    wykorzystuje algorytm KMeans do grupowania
    użytkowników o zbliżonych preferencjach. Na tej podstawie generuje:

        - rekomendacje filmów, których dany użytkownik jeszcze nie widział,
        anty-rekomendacje (filmy najmniej oceniane przez podobnych użytkowników),
        - opcjonalne pobieranie dodatkowych informacji o filmach z API OMDb.

Potrzebne biblioteki:
     pip install requests

Instrukcja użycia:
    1. Umieść plik „filmy_utf8.csv” w tym samym katalogu co skrypt.
    2. Uruchom program (np. w Google Colab lub Pythonie lokalnie).
    3. Program wyświetli listę dostępnych użytkowników — podaj ich nazwę lub numer.
    4. Wybierz, czy pobierać dodatkowe dane z OMDb.
    5. Wynjk:
        – listę rekomendowanych filmów,
        – listę filmów do unikania,

"""

import pandas as pd
import os
from sklearn.cluster import KMeans
import numpy as np
import requests
from urllib.parse import quote
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def load_movie_data(filename):
    """
    Load raw movie rating data from a CSV file.

    Args:
        filename (str): CSV filename.

    Returns:
        pandas.DataFrame: Raw table of rows like:
            [user, film1, rating1, film2, rating2, ...]
    """
    script_dir = os.path.dirname(os.path.abspath(filename))
    filepath = os.path.join(script_dir, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Nie znaleziono pliku: {filepath}")

    return pd.read_csv(filepath, header=None, sep=",")


def parse_ratings(raw_data):
    """
    Parse raw data into structured dict:
        {user: {movie: rating}}.

    Args:
        raw_data (DataFrame): Raw input rows.

    Returns:
        dict: Nested dict of user → film → rating.
    """
    data_dict = {}

    for i in range(raw_data.shape[0]):
        user = str(raw_data.iloc[i, 0]).strip()
        user_data = raw_data.iloc[i, 1:]
        films = user_data[::2]
        scores = user_data[1::2]

        film_ratings = {}
        for film, score in zip(films, scores):
            if pd.notna(film) and pd.notna(score):
                try:
                    film_ratings[str(film).strip()] = float(score)
                except ValueError:
                    continue

        data_dict[user] = film_ratings

    return data_dict


def create_ratings_matrix(data_dict):
    """
    Convert {user → {film → rating}} into a DataFrame matrix.

    Args:
        data_dict (dict)

    Returns:
        DataFrame: rows = users, columns = films, values = ratings.
    """
    all_movies = sorted({film for user in data_dict.values() for film in user})
    ratings = pd.DataFrame(index=data_dict.keys(), columns=all_movies, dtype=float)

    for user, movies in data_dict.items():
        for film, score in movies.items():
            ratings.loc[user, film] = score

    return ratings


def clean_movie_title(title):
    """
    Clean title for API lookup (strip years, brackets, whitespace).

    Args:
        title (str)

    Returns:
        str: cleaned title
    """
    import re

    title = title.strip()
    title = re.sub(r'\s*\(\d{4}\)\s*', '', title)
    title = re.sub(r'\s*\[.*?\]\s*', '', title)

    return title


def get_omdb_info(movie_title, api_key="7e188fd4", cache=None):
    """
    Fetch movie metadata from OMDb API via HTTPS.

    Args:
        movie_title (str): Title to search.
        api_key (str): OMDb API key.
        cache (dict): Optional lookup cache.

    Returns:
        dict | None: Parsed movie info, or None if not found.
    """
    if cache is not None and movie_title in cache:
        return cache[movie_title]

    cleaned_title = clean_movie_title(movie_title)
    encoded_title = quote(cleaned_title)

    url = f"https://www.omdbapi.com/?apikey={api_key}&t={encoded_title}"

    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)

    try:
        response = session.get(url, timeout=6)

        if response.status_code != 200:
            if cache is not None:
                cache[movie_title] = None
            return None

        data = response.json()
        if data.get("Response") == "False":
            if cache is not None:
                cache[movie_title] = None
            return None

        info = {
            "title": data.get("Title", "Nieznany"),
            "year": data.get("Year", "Nieznany"),
            "rating": data.get("imdbRating", "Brak oceny"),
            "genres": data.get("Genre", "Nieznane"),
            "directors": data.get("Director", "Nieznani"),
            "plot": data.get("Plot", "Brak opisu"),
            "runtime": data.get("Runtime", "Nieznany"),
            "actors": data.get("Actors", "Nieznani")
        }

        if cache is not None:
            cache[movie_title] = info

        return info

    except Exception:
        if cache is not None:
            cache[movie_title] = None
        return None


def recommend_movies(ratings, target_user, n_clusters=4, top_n=5, min_ratings=2):
    """
    Generate recommendations using KMeans clustering.

    Args:
        ratings (DataFrame)
        target_user (str)
        n_clusters (int)
        top_n (int)
        min_ratings (int)

    Returns:
        pandas.Series: Film → predicted rating
    """
    if target_user not in ratings.index:
        raise ValueError(f"Użytkownik '{target_user}' nie znajduje się w danych")

    ratings_filled = ratings.fillna(ratings.mean())
    n_clusters = min(n_clusters, len(ratings) - 1)

    kmeans = KMeans(n_clusters=n_clusters, random_state=17, n_init=15)
    clusters = kmeans.fit_predict(ratings_filled)

    user_cluster = clusters[ratings.index.get_loc(target_user)]
    cluster_mask = pd.Series(clusters, index=ratings.index) == user_cluster
    similar_users = ratings.loc[cluster_mask].drop(target_user, errors="ignore")

    if similar_users.empty:
        popular = ratings.mean().sort_values(ascending=False)
        unrated = ratings.loc[target_user].isna()

        valid = []
        for movie in popular[unrated].index:
            if ratings[movie].notna().sum() >= min_ratings:
                valid.append(movie)
            if len(valid) >= top_n:
                break

        return popular[valid]

    unrated = ratings.loc[target_user].isna()
    recs = {}

    for movie in ratings.columns:
        if unrated[movie]:
            movie_ratings = similar_users[movie].dropna()
            if len(movie_ratings) >= min_ratings:
                recs[movie] = movie_ratings.mean()

    if not recs:
        for movie in ratings.columns:
            if unrated[movie]:
                movie_ratings = similar_users[movie].dropna()
                if len(movie_ratings) >= 1:
                    recs[movie] = movie_ratings.mean()

    return pd.Series(recs).sort_values(ascending=False).head(top_n)


def get_anti_recommendations(ratings, target_user, n_clusters=4, bottom_n=5, min_ratings=2):
    """
    Generate anti-recommendations (lowest predicted ratings).

    Args:
        ratings (DataFrame)
        target_user (str)
        n_clusters (int)
        bottom_n (int)
        min_ratings (int)

    Returns:
        pandas.Series: Film → predicted low rating
    """
    if target_user not in ratings.index:
        raise ValueError(f"Użytkownik '{target_user}' nie znajduje się w danych")

    ratings_filled = ratings.fillna(ratings.mean())
    n_clusters = min(n_clusters, len(ratings) - 1)

    kmeans = KMeans(n_clusters=n_clusters, random_state=17, n_init=15)
    clusters = kmeans.fit_predict(ratings_filled)

    user_cluster = clusters[ratings.index.get_loc(target_user)]
    mask = pd.Series(clusters, index=ratings.index) == user_cluster
    similar_users = ratings.loc[mask].drop(target_user, errors="ignore")

    if similar_users.empty:
        return pd.Series(dtype=float)

    unrated = ratings.loc[target_user].isna()
    anti = {}

    for movie in ratings.columns:
        if unrated[movie]:
            movie_ratings = similar_users[movie].dropna()
            if len(movie_ratings) >= min_ratings:
                anti[movie] = movie_ratings.mean()

    if not anti:
        for movie in ratings.columns:
            if unrated[movie]:
                movie_ratings = similar_users[movie].dropna()
                if len(movie_ratings) >= 1:
                    anti[movie] = movie_ratings.mean()

    return pd.Series(anti).sort_values(ascending=True).head(bottom_n)


def display_movie_with_info(idx, film, score, cluster_ratings, api_key, show_info=True, cache=None):
    """
    Display formatted recommendation output with optional OMDb metadata.
    """
    print(f"\n{idx}. {film}")
    print(f"   Przewidywana ocena: {score:.2f}")
    print(f"   Bazując na {len(cluster_ratings)} ocenach: {cluster_ratings.tolist()}")

    if show_info and api_key:
        info = get_omdb_info(film, api_key, cache)
        if info:
            print(f"   OMDb: {info['title']} ({info['year']})")
            print(f"   Ocena IMDb: {info['rating']}/10")
            print(f"   Gatunki: {info['genres']}")
            print(f"   Reżyseria: {info['directors']}")
            print(f"   Czas trwania: {info['runtime']}")
            if info['plot'] and info['plot'] != "Brak opisu":
                short = info["plot"][:150]
                print(f"   Opis: {short}{'...' if len(info['plot']) > 150 else ''}")
        else:
            print("   OMDb: Nie znaleziono")


def main():
    """
    Main program: load data, build rating matrix, ask user,
    compute recommendations and display results.
    """
    raw = load_movie_data("filmy_utf8.csv")

    data_dict = parse_ratings(raw)
    ratings = create_ratings_matrix(data_dict)

    print(f"\nLiczba użytkowników: {len(ratings)}")
    print(f"Liczba filmów: {len(ratings.columns)}")

    print("\nDostępni użytkownicy:")
    for idx, user in enumerate(ratings.index, 1):
        print(f"{idx}. {user}")

    target_user = input("\nWpisz nazwę użytkownika lub numer: ").strip()

    if target_user.isdigit():
        idx = int(target_user) - 1
        if 0 <= idx < len(ratings):
            target_user = ratings.index[idx]
        else:
            print("Nieprawidłowy numer użytkownika.")
            return

    if target_user not in ratings.index:
        print(f"\nBłąd: Użytkownik '{target_user}' nie został znaleziony.")
        return

    use_omdb = input("\nCzy pobrać dodatkowe informacje z OMDb? (tak/nie): ").strip().lower()
    show_info = use_omdb in ["tak", "t", "yes", "y"]

    api_key = "7e188fd4"
    movie_cache = {}

    if show_info:
        test = get_omdb_info("The Matrix", api_key)
        if not test:
            print("Nie udało się połączyć z OMDb. Dane API zostaną pominięte.")
            show_info = False
        else:
            print("Połączenie z OMDb działa!")

    print("\n============================================================")
    print(f"Rekomendacje dla użytkownika: {target_user}")
    print("============================================================")

    n_clusters = 4
    recommendations = recommend_movies(ratings, target_user, n_clusters=n_clusters, top_n=5, min_ratings=2)

    ratings_filled = ratings.fillna(ratings.mean())
    n_clusters_actual = min(n_clusters, len(ratings) - 1)
    kmeans = KMeans(n_clusters=n_clusters_actual, random_state=17, n_init=15)
    clusters = kmeans.fit_predict(ratings_filled)
    user_cluster = clusters[ratings.index.get_loc(target_user)]
    cluster_users = ratings.index[clusters == user_cluster].tolist()

    print("\n============================================================")
    print("REKOMENDOWANE FILMY:")
    print("============================================================")

    if recommendations.empty:
        print("Brak wystarczających rekomendacji.")
    else:
        for idx, (film, score) in enumerate(recommendations.items(), 1):
            cluster_r = ratings.loc[cluster_users, film].dropna()
            display_movie_with_info(idx, film, score, cluster_r, api_key, show_info, movie_cache)

    anti = get_anti_recommendations(ratings, target_user, n_clusters=n_clusters, bottom_n=5, min_ratings=2)

    if not anti.empty:
        print("\n============================================================")
        print("FILMY DO UNIKANIA:")
        print("============================================================")

        for idx, (film, score) in enumerate(anti.items(), 1):
            cluster_r = ratings.loc[cluster_users, film].dropna()
            display_movie_with_info(idx, film, score, cluster_r, api_key, show_info, movie_cache)


if __name__ == "__main__":
    main()
