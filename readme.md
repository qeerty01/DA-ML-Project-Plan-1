# DA&ML Project Plan

## Course Name
Data Analytics & Machine Learning

## Project Title
Movie Ratings & Recommendation System

## Team Name
777



# 1. Project Overview
The Movie Ratings & Recommendation System project aims to analyze user ratings of movies to uncover trends, preferences, and patterns in viewing behavior. The goal is to design a system that can provide personalized movie recommendations based on past ratings and user similarities. The project focuses on data analysis and machine learning techniques to derive actionable insights from a movie ratings dataset.



# 2. Team Members & Roles

| Name & Surname | Student ID | Role | Responsibilities |
|---------------|-----------|------|------------------|
| Mariya Gorbacheva | 202490123 | Project Manager | Oversees project progress, coordinates team activities, ensures deadlines are met |
| Karina Gumerova | 202490124| Data Analyst | Cleans, processes, and analyzes the dataset; identifies patterns and trends |
| Maksim Em | 202490106 | ML Engineer | Implements recommendation algorithms and predictive models |
| Diyorbek Voidjonov | 202490357| Documentation Specialist | Prepares and maintains project documentation, visualizations, and reports |



# 3. Problem Statement
With the increasing number of movies and streaming platforms, users often struggle to discover content that suits their preferences. Manual selection can be time-consuming and inefficient. This project addresses the challenge by analyzing historical movie ratings and providing personalized recommendations, enhancing user experience and engagement.


# 4. Dataset Information
**Source:** [MovieLens 1M](https://grouplens.org/datasets/movielens/)  

**Description:**
- ~1,000,000 movie ratings from real users  
- Contains information about users, movies, and ratings  

**Dataset structure:**
| Column | Description |
|--------|-------------|
| user_id | Unique user identifier |
| movie_id | Unique movie identifier |
| rating | Rating value (1–5) |
| timestamp | When rating was given |
| movie_title | Movie title |
| genres | Movie genres |

---

# 5. Project Objectives
1. Load and prepare the dataset  
2. Perform exploratory data analysis (EDA)  
3. Build a recommendation model (SVD)  
4. Evaluate model performance using RMSE  
5. Generate movie recommendations for users  

---

# 6. Data Preparation

## Importing Libraries
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
```
---

## Loading the Dataset

 ### remove duplicate rows
 ```python
data = data.drop_duplicates()
```

### check missing values
```python
print(data.isnull().sum())
```

### remove missing values
 ```python
data = data.dropna()
```
## Feature Extraction

```python
# convert genres into numeric features
genres_encoded = data["genres"].str.get_dummies(sep="|")

# add encoded genres to dataset
data = pd.concat([data, genres_encoded], axis=1)

data.head()
```

## Aggregation
```python
# average rating per movie
avg_movie_rating = data.groupby("title")["rating"].mean()

# number of ratings per movie
rating_count = data.groupby("title")["rating"].count()

movie_stats = pd.DataFrame({
    "average_rating": avg_movie_rating,
    "rating_count": rating_count
})

movie_stats.head()
```
## Splitting Data
```python
X = data[["user_id","movie_id"]]
y = data["rating"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Train size:", len(X_train))
print("Test size:", len(X_test))
```
---

# 7. Data Analysis Tasks

## Descriptive Statistics
```python
# rating distribution
rating_distribution = data["rating"].value_counts().sort_index()

# top rated movies
top_rated_movies = movie_stats.sort_values(
    by="average_rating",
    ascending=False
).head(10)

print(rating_distribution)
print(top_rated_movies)
```

## User Behavior Analysis
```python
# average rating per user
user_avg_rating = data.groupby("user_id")["rating"].mean()

# most active users
active_users = data.groupby("user_id")["rating"].count().sort_values(
    ascending=False
)

print(user_avg_rating.head())
print(active_users.head(10))
```
## Genre Analysis
```python
genre_columns = genres_encoded.columns

genre_popularity = data[genre_columns].sum().sort_values(
    ascending=False
)

print(genre_popularity.head(10))
```
## Correlation Analysis
```python
# user-movie matrix
user_movie_matrix = data.pivot_table(
    index="user_id",
    columns="title",
    values="rating"
)

# movie correlations
movie_correlations = user_movie_matrix.corr()

movie_correlations.head()
```
---

# 8. Key Findings and Insights

Based on the exploratory data analysis of the MovieLens 1M dataset, several important insights were identified.

### Popular Genres
The most frequently watched genres include **Drama, Comedy, and Action**. These genres appear in a large proportion of the movies and receive a high number of user ratings, indicating strong audience interest.

### Rating Distribution
Most user ratings fall between **3 and 4 stars**, suggesting that users tend to rate movies positively but still distinguish between average and highly enjoyable films.

### Top-Rated Movies
Movies with a high average rating typically have **both strong ratings and a sufficient number of user reviews**. This indicates that highly rated movies are not only liked by a few users but appreciated by a broader audience.

### User Activity Patterns
User activity varies significantly:
- Some users rate only a few movies.
- A smaller group of highly active users contributes a large portion of the ratings.

This pattern is common in recommender system datasets and highlights the importance of identifying active users when building recommendation models.

### Movie Popularity
A small subset of movies receives the majority of ratings. These movies are considered **popular titles** and often belong to well-known genres or franchises.

### User–Movie Interaction Patterns
The analysis shows that user preferences tend to cluster around certain genres or movie styles. This pattern makes it possible to apply **collaborative filtering techniques** to recommend movies based on similarities between users and their past ratings.

---

# 9. Project Timeline

| Week | Activity |
|-----|---------|
| Week 1 | Project planning and dataset selection |
| Week 2 | Data cleaning, preprocessing, and exploratory analysis |
| Week 3 | Building recommendation algorithms and ML models |
| Week 4 | Model evaluation, insights extraction, visualization |
| Week 5 | Documentation, final report preparation, submission |

---

# 10. Expected Outcomes
- An organized analysis of movie ratings  
- A working recommendation algorithm (proof-of-concept)  
- Insights into user preferences and movie popularity  
- Comprehensive project documentation



# 11. Conclusion
This project demonstrates the application of data analytics and machine learning to solve a real-world problem of personalized movie recommendations. By following a structured approach, the project will provide actionable insights and a functional recommendation model.



# 12. References
- MovieLens Dataset – https://grouplens.org/datasets/movielens/  
- Python Libraries: pandas, numpy, scikit-learn  
- Academic materials and lecture notes on Data Analytics & Machine Learning



