import requests
import pandas as pd
import numpy as np
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY = ' ' # Insert API key here
BASE_URL = 'https://api.themoviedb.org/3'

# Function to get movie details (update to include animated and maturity flags)
def get_movie_details(movie_id):
    url = f"{BASE_URL}/movie/{movie_id}?api_key={API_KEY}"
    response = requests.get(url)
    movie_details = response.json()
    # Add animated flag (check if it's animated) and maturity rating
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

def get_movie():
    movie_title = input("Please enter the movie title: ")
    search_url = f"{BASE_URL}/search/movie?api_key={API_KEY}&query={movie_title}"
    response = requests.get(search_url)
    search_results = response.json().get('results', [])
    
    if search_results:
        # List the titles and years for the user to choose from
        print("\nFound the following movies with that title:")
        for idx, movie in enumerate(search_results):
            title = movie.get('title', 'Unknown Title')
            release_year = movie.get('release_date', 'Unknown')[:4]  # Extract the year
            print(f"{idx + 1}. {title} ({release_year})")

        # Ask the user to select a movie by number
        try:
            selected_idx = int(input("\nPlease select the number corresponding to the correct movie: ")) - 1
            if 0 <= selected_idx < len(search_results):
                selected_movie = search_results[selected_idx]
                movie_id = selected_movie.get('id')  # Get the ID of the selected movie
                movie_details = get_movie_details(movie_id)  # Fetch the movie details using the ID
                movie_details['title'] = selected_movie.get('title')
                movie_details['release_year'] = selected_movie.get('release_date', 'Unknown')[:4]
                return movie_details
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    else:
        print("No movies found with that title.")
    return None

# Function to get movies within the specified year range
def get_movies_by_year_range(api_key, target_year, year_range=5, input_movie_id=None):
    all_movies = []
    futures = []
    
    # Fetch movies within the 5 years before and after the inputted year
    for year in range(target_year - year_range, target_year + year_range + 1):
        url = f"{BASE_URL}/discover/movie?api_key={api_key}&primary_release_year={year}&sort_by=popularity.desc"
        response = requests.get(url)
        
        if response.status_code != 200:
            print(f"Error fetching movies for year {year}: {response.status_code}")
            continue
        
        data = response.json()
        
        if 'results' not in data:
            print(f"No 'results' key found in the response data for year {year}")
            continue
        
        # Schedule fetching movie details and credits for each movie in the current year
        with ThreadPoolExecutor(max_workers=10) as executor:
            for movie in data['results']:
                movie_id = movie.get('id')
                if movie_id and (movie_id != input_movie_id):  # Ensure the movie is not the input movie
                    futures.append(executor.submit(get_movie_details, movie_id))
                    futures.append(executor.submit(get_movie_credits, movie_id))
        
        print(f"Scheduled fetching for movies in year: {year}")
        time.sleep(1)  # Avoid rate-limiting for page requests

    # Collect results as they complete
    for future in as_completed(futures):
        try:
            result = future.result()
            if result is not None:
                all_movies.append(result)
        except Exception as e:
            print(f"Error occurred: {e}")
    
    return all_movies

# Function to process movie data into a DataFrame
def process_movie_data(all_movies):
    movies_data = []
    for movie in all_movies:
        movie_id = movie.get('id')
        title = movie.get('title', 'Unknown Title')  # Provide a default if title is missing
        genres = [genre['name'] for genre in movie.get('genres', [])]
        credits = movie.get('credits', {})
        cast = [cast_member['name'] for cast_member in credits.get('cast', [])[:5]]  # Top 5 cast
        rating = movie.get('vote_average', np.random.uniform(1, 10))
        runtime = movie.get('runtime', 0)
        popularity = movie.get('popularity', 0)
        crew = [crew_member['name'] for crew_member in credits.get('crew', []) if crew_member['job'] == 'Director']
        release_date = movie.get('release_date', '2000')[:4]  # Take year only
        is_animated = movie.get('is_animated', False)  # Default to False if not present

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

# Search for the target movie based on user input
def find_target_movie(movies_df, target_movie_title, target_movie_year):
    target_movie = movies_df[(movies_df['title'].str.contains(target_movie_title, case=False, na=False)) & (movies_df['release_date'] == target_movie_year)]
    if not target_movie.empty:
        return target_movie.index[0]
    else:
        print(f"'{target_movie_title} ({target_movie_year})' not found in the dataset. Check the movie title.")
        return None

# Function to exclude sequels
def is_sequel(original_title, title):
    original_base = re.sub(r'\s*[Ii]+[iI]*$', '', original_title)  # Remove 'I', 'II', etc.
    return original_base.lower() in title.lower() and original_title.lower() != title.lower()

# One-hot encoding function for lists
def one_hot_encode(data, unique_labels):
    encoded = np.zeros((len(data), len(unique_labels)))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    for i, row in enumerate(data):
        for item in row:
            if item in label_to_index:
                encoded[i][label_to_index[item]] = 1
    return encoded

# Normalize runtime, popularity, and rating
def normalize(column):
    column = pd.to_numeric(column, errors='coerce')  # Convert to numeric, forcing errors to NaN
    min_val = column.min()
    max_val = column.max()
    if min_val == max_val:  # Avoid division by zero
        return np.zeros(len(column))  # Return zeros if all values are the same
    return (column - min_val) / (max_val - min_val)

# Function to compute cosine similarity
def compute_cosine_similarity(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0  # No similarity if either vector is zero
    return np.dot(vec1, vec2) / (norm1 * norm2)

# Function to find similar movies to the target movie
def find_similar_movies(movie_index, feature_data, n_neighbors=5):
    target_vector = feature_data[movie_index]
    is_target_animated = movies_df.iloc[movie_index]['is_animated']  # Get animation flag for the target movie

    similarities = []

    for idx, movie_vector in enumerate(feature_data):
        if idx == movie_index or is_sequel(movies_df.iloc[movie_index]['title'], movies_df.iloc[idx]['title']):
            continue
        
        if movies_df.iloc[idx]['is_animated'] != is_target_animated:
            continue

        similarity = compute_cosine_similarity(target_vector, movie_vector)
        similarities.append((idx, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    
    unique_indices = []
    for idx, _ in similarities:
        if idx not in unique_indices:
            unique_indices.append(idx)

        if len(unique_indices) >= n_neighbors:
            break
    
    if len(unique_indices) < n_neighbors:
        remaining_indices = [idx for idx in range(len(movies_df)) if idx != movie_index and idx not in unique_indices]
        unique_indices.extend(remaining_indices[:n_neighbors - len(unique_indices)])

    # Prepare the result with similarity score
    similar_movies_with_scores = []
    for idx in unique_indices[:n_neighbors]:
        similarity_score = next(score for i, score in similarities if i == idx)
        similar_movies_with_scores.append((movies_df.iloc[idx]['title'], similarity_score))
    
    return similar_movies_with_scores

# Main program
if __name__ == '__main__':
    print("Welcome to the Movie Recommendation System!")
    
    # Get movie details for the selected movie after listing the options
    input_movie = get_movie()
    if input_movie:
        target_movie_title = input_movie.get('title')
        target_movie_year = input_movie.get('release_year')
        all_movies = get_movies_by_year_range(API_KEY, int(input_movie.get('release_date', '2000')[:4]), input_movie_id=input_movie['id'])

        # Ensure the input movie is included in the all_movies list
        all_movies.append(input_movie)
        movies_df = process_movie_data(all_movies)

        target_movie_index = find_target_movie(movies_df, target_movie_title, target_movie_year)

        if target_movie_index is not None:
        # (Continue with recommendation processing...)

            # Extract features for recommendation algorithm
            movies_df['genres'] = movies_df['genres'].apply(lambda x: x if isinstance(x, list) else [])
            movies_df['cast'] = movies_df['cast'].apply(lambda x: x if isinstance(x, list) else [])

            unique_genres = sorted(set(genre for sublist in movies_df['genres'] for genre in sublist))
            unique_cast = sorted(set(cast_member for sublist in movies_df['cast'] for cast_member in sublist))

            genre_encoded = one_hot_encode(movies_df['genres'], unique_genres)
            cast_encoded = one_hot_encode(movies_df['cast'], unique_cast)

            movies_df['runtime'] = normalize(movies_df['runtime'])
            movies_df['popularity'] = normalize(movies_df['popularity'])
            movies_df['rating'] = normalize(movies_df['rating'])
            movies_df['release_date'] = normalize(pd.to_numeric(movies_df['release_date'], errors='coerce'))

            features = np.hstack([ 
                genre_encoded, 
                cast_encoded, 
                np.array(movies_df['runtime']).reshape(-1, 1), 
                np.array(movies_df['popularity']).reshape(-1, 1), 
                np.array(movies_df['rating']).reshape(-1, 1), 
                np.array(movies_df['release_date']).reshape(-1, 1)
            ])

            # Get similar movies with similarity scores
            similar_movies = find_similar_movies(target_movie_index, features)
            print("\nHere are some similar movies with similarity scores:")
            for title, score in similar_movies:
                print(f"- {title}: Similarity Score: {score:.4f}")
        else:
            print("Movie not found.")
    else:
        print("Movie not found.")
