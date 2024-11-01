import requests
import pandas as pd
import numpy as np
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY = #Add your API key here
BASE_URL = 'https://api.themoviedb.org/3'

# Function to get movie details (update to include animated and maturity flags)
def get_movie_details(movie_id):
    url = f"{BASE_URL}/movie/{movie_id}?api_key={API_KEY}"
    response = requests.get(url)
    movie_details = response.json()
    is_animated = any(genre['name'] == 'Animation' for genre in movie_details.get('genres', []))
    maturity_rating = movie_details.get('adult', False) 
    movie_details['is_animated'] = is_animated
    movie_details['maturity_rating'] = maturity_rating
    return movie_details

# Function to get movie credits (for cast and crew)
def get_movie_credits(movie_id):
    url = f"{BASE_URL}/movie/{movie_id}/credits?api_key={API_KEY}"
    response = requests.get(url)
    return response.json()

# Function to get all movies concurrently
def get_all_movies(api_key, max_pages=50):
    all_movies = []
    futures = []

    with ThreadPoolExecutor(max_workers=40) as executor:
        for page in range(1, max_pages + 1):
            url = f"{BASE_URL}/movie/popular?api_key={api_key}&page={page}"
            response = requests.get(url)

            if response.status_code != 200:
                print(f"Error fetching page {page}: {response.status_code}")
                continue

            data = response.json()

            if 'results' not in data:
                print(f"No 'results' key found in the response data for page {page}")
                continue

            for movie in data['results']:
                movie_id = movie.get('id')
                if movie_id:
                    futures.append(executor.submit(get_movie_details, movie_id))
                    futures.append(executor.submit(get_movie_credits, movie_id))

            print(f"Scheduled fetching for page {page}")
            time.sleep(1)  # Avoid rate-limiting for page requests

        for future in as_completed(futures):
            try:
                result = future.result()
                if result is not None:
                    all_movies.append(result)
            except Exception as e:
                print(f"Error occurred: {e}")

    movies_data = []
    for movie in all_movies:
        movie_id = movie.get('id')
        title = movie.get('title', 'Unknown Title')
        genres = [genre['name'] for genre in movie.get('genres', [])]
        credits = movie.get('credits', {})
        cast = [cast_member['name'] for cast_member in credits.get('cast', [])[:5]]
        rating = movie.get('vote_average', np.random.uniform(1, 10))
        runtime = movie.get('runtime', 0)
        popularity = movie.get('popularity', 0)
        crew = [crew_member['name'] for crew_member in credits.get('crew', []) if crew_member['job'] == 'Director']
        release_date = movie.get('release_date', '2000')[:4]
        is_animated = movie.get('is_animated', False)

        movies_data.append({
            'movie_id': movie_id,
            'title': title,
            'genres': genres,
            'cast': cast,
            'rating': rating,
            'runtime': runtime,
            'popularity': popularity,
            'crew': crew,
            'release_date': release_date,
            'is_animated': is_animated
        })

    return pd.DataFrame(movies_data)

# Function to get target movie index based on user input
def get_target_movie_index(movies_df, title, year=None):
    if year:
        target_movie = movies_df[(movies_df['title'].str.contains(title, case=False, na=False)) & (movies_df['release_date'] == year)]
    else:
        target_movie = movies_df[movies_df['title'].str.contains(title, case=False, na=False)]

    if not target_movie.empty:
        print(f"'{title}' found in the dataset.")
        return target_movie.index[0]
    else:
        print(f"'{title} ({year if year else 'any year'})' not found in the dataset.")
        return None

# Main code block
# Ask the user for movie title and year (optional)
movie_title = input("Enter the movie title: ")
movie_year = input("Enter the release year (optional): ").strip() or None

# Fetch all movies initially with a small set
print("Fetching initial movie dataset...")
movies_df = get_all_movies(API_KEY, max_pages=5)

# Get the index of the target movie
target_movie_index = get_target_movie_index(movies_df, movie_title, movie_year)

# If target movie is not found in the initial dataset, expand the search
if target_movie_index is None:
    print("Expanding the dataset to search for the target movie...")
    movies_df = get_all_movies(API_KEY, max_pages=50)
    target_movie_index = get_target_movie_index(movies_df, movie_title, movie_year)

if target_movie_index is not None:
    # Normalize and prepare data if the target movie is found
    print("Preparing and normalizing movie data for recommendations...")

    # Ensure genres and cast are lists of strings
    movies_df['genres'] = movies_df['genres'].apply(lambda x: x if isinstance(x, list) else [])
    movies_df['cast'] = movies_df['cast'].apply(lambda x: x if isinstance(x, list) else [])

    unique_genres = sorted(set(genre for sublist in movies_df['genres'] for genre in sublist))
    unique_cast = sorted(set(cast_member for sublist in movies_df['cast'] for cast_member in sublist))

    # One-hot encoding function for lists
    def one_hot_encode(data, unique_labels):
        encoded = np.zeros((len(data), len(unique_labels)))
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        for i, row in enumerate(data):
            for item in row:
                if item in label_to_index:
                    encoded[i][label_to_index[item]] = 1
        return encoded

    # Normalization function
    def normalize(column):
        column = pd.to_numeric(column, errors='coerce')
        min_val = column.min()
        max_val = column.max()
        if min_val == max_val:
            return np.zeros(len(column))
        return (column - min_val) / (max_val - min_val)

    genre_encoded = one_hot_encode(movies_df['genres'], unique_genres)
    cast_encoded = one_hot_encode(movies_df['cast'], unique_cast)

    movies_df['runtime'] = normalize(movies_df['runtime'])
    movies_df['popularity'] = normalize(movies_df['popularity'])
    movies_df['rating'] = normalize(movies_df['rating'])
    movies_df['release_date'] = normalize(pd.to_numeric(movies_df['release_date'], errors='coerce'))

    features = np.hstack([
        genre_encoded * 2.5,
        cast_encoded * 3.0,
        np.nan_to_num(np.array(movies_df[['runtime']])) * 0.5,
        np.nan_to_num(np.array(movies_df[['popularity']])) * 0.5,
        np.nan_to_num(np.array(movies_df[['rating']])) * 2.5,
        np.nan_to_num(np.array(movies_df[['release_date']])) * 1.0
    ])

    def compute_cosine_similarity(vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2)

    # Add to the 'find_similar_movies' function
def find_similar_movies(movie_index, feature_data, n_neighbors=5):
    target_vector = feature_data[movie_index]
    target_title = movies_df.iloc[movie_index]['title']
    similarities = []

    for idx, movie_vector in enumerate(feature_data):
        if idx == movie_index:
            continue

        similarity = compute_cosine_similarity(target_vector, movie_vector)
        # Exclude movies that appear to be sequels of the target movie
        movie_title = movies_df.iloc[idx]['title']
        if re.search(r'\b(2|3|4|II|III|IV|Part|Chapter)\b', movie_title, re.IGNORECASE) and target_title.lower() in movie_title.lower():
            continue

        similarities.append((idx, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    unique_indices = [idx for idx, _ in similarities[:n_neighbors]]

    return movies_df.iloc[unique_indices[:n_neighbors]]

    similar_movies = find_similar_movies(target_movie_index, features, n_neighbors=5)

    print(f"Movies similar to '{movies_df.iloc[target_movie_index]['title']}':")
    for idx in similar_movies.index:
        print(f"Title: {movies_df.iloc[idx]['title']}, Rating: {movies_df.iloc[idx]['rating']:.2f}, Release Date: {movies_df.iloc[idx]['release_date']}")

else:
    print("Movie not found in the extended dataset.")