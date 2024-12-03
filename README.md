# Movie Recommendation System

This project implements a movie recommendation system using different machine learning techniques and data processing.

## Datasets Used

- tmdb_5000_movies.csv: Contains information about 5000 movies including budget, genres, keywords, revenue, etc.
- tmdb_5000_credits.csv: Contains information about movie cast and crew
- ratings.csv: Contains user ratings for different movies

## Main Features

### 1. Data Preprocessing

#### convert_json_to_list()
- Converts JSON strings to Python lists
- Handles error cases by returning empty list

#### adjust_df()
- Main preprocessing function for movies dataset
- Extracts and organizes information about:
  - Director (ID, name, gender)
  - Main cast (first 3 actors)
  - Main production country
  - Main production company
  - Organized genres
  - Normalized keywords
- Converts dates and IDs to appropriate formats

### 2. Keyword Cleaning and Normalization

#### replace_keyword_with_most_frequent()
- Normalizes keywords using stemming
- Replaces variations with most frequent form

#### clean_keywords()
- Removes rare keywords (less than 5 occurrences)
- Replaces synonyms with most common form
- Keeps only relevant keywords

#### validate_and_plot_keyword_occurrences()
- Validates cleaning process
- Generates visualizations comparing frequencies before/after

### 3. Exploratory Analysis

#### process_association()
- Analyzes associations between categorical columns and numerical values
- Calculates grouped averages

#### process_col_count()
- Counts frequencies of values in categorical columns

#### generate_wordcloud()
- Generates word clouds to visualize frequencies

### 4. Recommendation System

The project implements three different approaches:

#### 4.1 Content-Based (Description)
- Uses TF-IDF on movie descriptions
- Calculates cosine similarity
- Function: get_recommendations()

#### 4.2 Content-Based (Metadata)
- Combines genres, keywords, cast and director
- Uses CountVectorizer and cosine similarity
- Function: get_recommendations() with cosine_sim2

#### 4.3 Collaborative Filtering
- Uses SVD (Singular Value Decomposition)
- Predicts user ratings for unwatched movies
- Function: get_recommendations_in_batches()

## Results and Insights

1. Keyword cleaning significantly reduced noise in data, keeping only relevant and frequent terms

2. Recommendations based only on description tend to capture general thematic similarities

3. Recommendations based on combined metadata (genres, cast, etc) produce more specific and accurate results

4. Collaborative filtering allows personalizing recommendations for each user based on their rating history

## Visualizations

The notebook includes several visualizations:
- Statistical distributions of numerical variables
- Word clouds of genres and keywords
- Comparative graphs of keyword cleaning process

## Possible Improvements

1. Implement cross-validation to evaluate recommendation quality
2. Add different weights for each type of metadata
3. Create a hybrid system combining different approaches
4. Optimize SVD hyperparameters
5. Add more evaluation metrics

## Requirements

- pandas
- numpy
- scikit-learn
- surprise
- nltk
- wordcloud
- matplotlib
- seaborn

## How to Use

1. Load required datasets
2. Run preprocessing with adjust_df()
3. Clean keywords with clean_keywords()
4. Generate recommendations using one of three approaches:
   - get_recommendations() for content-based recommendations
   - get_recommendations_in_batches() for personalized user recommendations

## Code Examples

### Content-Based Recommendation

```python
# Get movie recommendations based on description
recommendations = get_recommendations("Avatar")

# Get recommendations based on metadata
recommendations = get_recommendations("Avatar", cosine_sim2)
```

### Collaborative Filtering

```python
# Get personalized recommendations for user
user_id = 1
recommendations = get_recommendations_in_batches(user_id, movies)
```

## Data Processing Flow

1. Initial Data Loading
   - Load movie data
   - Load credits data
   - Load ratings data

2. Data Cleaning
   - Convert JSON fields
   - Extract relevant information
   - Normalize text data
   - Handle missing values

3. Feature Engineering
   - Create combined metadata
   - Generate TF-IDF matrices
   - Calculate similarity matrices

4. Model Building
   - Build content-based models
   - Train SVD model for collaborative filtering

5. Recommendation Generation
   - Generate recommendations based on content similarity
   - Generate personalized user recommendations

## Performance Considerations

- Batch processing is used for large-scale predictions
- Vectorized operations are preferred over loops
- Preprocessing results are cached when possible
- Memory usage is optimized for large datasets

## Error Handling

The system includes robust error handling for:
- Invalid JSON data
- Missing values
- Unknown movies/users
- Data type mismatches

## Future Development

Planned future enhancements include:
1. API integration for real-time recommendations
2. Enhanced performance optimization
3. Additional recommendation algorithms
4. Improved evaluation metrics
5. User interface development

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Bug fixes
- New features
- Documentation improvements
- Performance optimizations

## Dataset source

https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset
