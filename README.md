# Movie Recommendation System

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Setup](#setup)
5. [Usage](#usage)
6. [Code Structure](#code-structure)
7. [Example](#example)
8. [License](#license)
9. [Contributing](#contributing)
10. [Contact](#contact)

## Overview

This Movie Recommendation System uses the TMDb (The Movie Database) API to fetch movie details and generate recommendations based on user input. The system implements a cosine similarity algorithm to find movies similar to a specified title, considering various features such as genres, cast, runtime, popularity, rating, and release date.

## Features

- Fetches movie data from TMDb API, including details like title, genres, cast, and ratings.
- Allows users to input a movie title and optional release year to find similar movies.
- Implements a cosine similarity algorithm to provide recommendations.
- Filters out sequels from the recommendation results to enhance relevance.

## Requirements

- Python 3.6 or higher
- Required packages:
  - `requests`
  - `pandas`
  - `numpy`
  - `concurrent.futures`

You can install the necessary packages using pip:

```bash
pip install requests pandas numpy
```

## Setup

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/movie-recommendation-system.git
   cd movie-recommendation-system
   ```

2. **API Key**:
   - Obtain an API key from [TMDb](https://www.themoviedb.org/documentation/api).
   - Replace the placeholder in the code with your API key (e.g., `API_KEY = 'insert your API key'`).

## Usage

Open your terminal and navigate to the project directory.

Run the script:

```bash
python movie_recommendation.py
```

Follow the prompts to enter the movie title and (optionally) the release year.

The system will display similar movies based on your input.

## Code Structure

- `movie_recommendation.py`: The main script that handles user input, fetches movie data from the TMDb API, and processes the recommendations.
  
Helper functions include:
- `get_movie_details(movie_id)`: Fetches details for a specific movie.
- `get_movie_credits(movie_id)`: Fetches cast and crew details for a specific movie.
- `get_all_movies(api_key, max_pages)`: Retrieves popular movies from TMDb.
- `get_target_movie_index(movies_df, title, year)`: Finds the index of the target movie based on user input.
- `find_similar_movies(movie_index, feature_data, n_neighbors)`: Finds and returns similar movies based on cosine similarity.

## Example

```bash
Enter the movie title: Deadpool
1. Deadpool (2016)
2. Deadpool 2 (2018)
```

The system will output similar movies, including their titles, ratings, and release dates.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have suggestions for improvements or new features.

## Contact

For questions or feedback, please reach out to [alexanderharshman@gmail.com].
