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
```python
ratings = pd.read_csv("ratings.dat", sep="::", engine="python",
                      names=["user_id","movie_id","rating","timestamp"])

movies = pd.read_csv("movies.dat", sep="::", engine="python",
                     names=["movie_id","title","genres"])

data = pd.merge(ratings, movies, on="movie_id")

data.head()
```

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
# Key Findings and Insights

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

# 8. Machine Learning

## Import ML Libraries
```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
```
## Model Training
```python
lr = LinearRegression()
lr.fit(X_train, y_train)

pred_lr = lr.predict(X_test)
rmse_lr = np.sqrt(mean_squared_error(y_test, pred_lr))

print("Linear Regression RMSE:", rmse_lr)
```
## Random Forest Model
```python
rf = RandomForestRegressor(n_estimators=50, random_state=42)
rf.fit(X_train, y_train)

pred_rf = rf.predict(X_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, pred_rf))

print("Random Forest RMSE:", rmse_rf)
```
## Prepare Data for SVD
```python
reader = Reader(rating_scale=(1, 5))

data_surprise = Dataset.load_from_df(
    data[['user_id', 'movie_id', 'rating']],
    reader
)

trainset, testset = train_test_split(data_surprise, test_size=0.2)
```
## Train SVD Model
```python
svd = SVD()
svd.fit(trainset)
```
## Evaluate SVD
```python
predictions = svd.test(testset)

rmse_svd = accuracy.rmse(predictions)
print("SVD RMSE:", rmse_svd)
```

## Movie Recommendation Function (SVD-based)
```python
def recommend_movies(user_id, top_n=10):
    all_movies = data['movie_id'].unique()
    predictions = []

    for movie_id in all_movies:
        pred = svd.predict(user_id, movie_id)
        predictions.append((movie_id, pred.est))

    recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]
    
    return recommendations
```

---


# 9. Comparison: EDA vs Machine Learning

## EDA Insights
- Reveals distribution of movie ratings in the MovieLens dataset  
- Identifies most popular genres such as Drama, Comedy, and Action  
- Helps understand user behavior patterns and movie popularity  
- Provides initial insights for designing recommendation strategies  

---

## ML Insights
- Predicts user ratings using collaborative filtering and regression models  
- Captures hidden relationships between users and movies  
- Enables personalized movie recommendation systems (e.g., SVD-based model)  
- Improves decision-making beyond simple descriptive analysis  

---

## Key Difference
Exploratory Data Analysis is **descriptive and interpretative**, focusing on understanding patterns in data, while Machine Learning is **predictive**, aiming to forecast user preferences and generate recommendations.

---

## Conclusion
EDA is essential for understanding the structure and behavior of the dataset, while Machine Learning transforms these insights into a functional recommendation system. In this project, both approaches complement each other: EDA helps to understand user and movie patterns, and ML enables personalized predictions and recommendations.


---


# 10. Conclusion

This project compared Exploratory Data Analysis (EDA) and Machine Learning (ML) approaches using the MovieLens dataset.

EDA provided a clear understanding of the data by identifying key patterns such as rating distribution, popular genres, and user activity. It helped explain user behavior and movie popularity in a simple and interpretable way.

Machine Learning, on the other hand, enabled prediction of user ratings and demonstrated how models can be used to build recommendation systems. In particular, collaborative filtering techniques such as SVD showed strong performance in capturing user–item relationships and generating personalized recommendations.

The comparison highlights that EDA and ML serve different but complementary purposes. EDA is essential for understanding and interpreting data, while ML is necessary for prediction and automation.

In real-world applications, both approaches should be used together: EDA to explore and understand the data, and ML to build scalable and personalized recommendation systems.

---


# 12. References
- MovieLens Dataset – https://grouplens.org/datasets/movielens/  
- Python Libraries: pandas, numpy, scikit-learn  
- Academic materials and lecture notes on Data Analytics & Machine Learning



