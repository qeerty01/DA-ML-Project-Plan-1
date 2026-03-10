# DA&ML Project Plan

## Course Name
Data Analytics & Machine Learning

## Subject
Data Analytics & Machine Learning

## Project Title
Movie Ratings & Recommendation System

## Team Name
777



## 1. Project Overview
The Movie Ratings & Recommendation System project aims to analyze user ratings of movies to uncover trends, preferences, and patterns in viewing behavior. The goal is to design a system that can provide personalized movie recommendations based on past ratings and user similarities. The project focuses on data analysis and machine learning techniques to derive actionable insights from a movie ratings dataset.



## 2. Team Members & Roles

| Name & Surname | Student ID | Role | Responsibilities |
|---------------|-----------|------|------------------|
| Mariya Gorbacheva | 202490123 | Project Manager | Oversees project progress, coordinates team activities, ensures deadlines are met |
| Karina Gumerova | 202490124| Data Analyst | Cleans, processes, and analyzes the dataset; identifies patterns and trends |
| Maksim Em | 202490106 | ML Engineer | Implements recommendation algorithms and predictive models |
| Diyorbek Voidjonov | 202490357| Documentation Specialist | Prepares and maintains project documentation, visualizations, and reports |



## 3. Problem Statement
With the increasing number of movies and streaming platforms, users often struggle to discover content that suits their preferences. Manual selection can be time-consuming and inefficient. This project addresses the challenge by analyzing historical movie ratings and providing personalized recommendations, enhancing user experience and engagement.


## 4. Dataset Information
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

## 5. Project Objectives
1. Load and prepare the dataset  
2. Perform exploratory data analysis (EDA)  
3. Build a recommendation model (SVD)  
4. Evaluate model performance using RMSE  
5. Generate movie recommendations for users  

---

## 6. Full Python Code
```python
# ================================
# Movie Ratings & Recommendation System
# Full Python Script
# ================================

import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse

# ================================
# Step 1: Load datasets
# ================================
ratings = pd.read_csv('ratings.dat', sep='::', engine='python',
                      names=['user_id','movie_id','rating','timestamp'])
movies = pd.read_csv('movies.dat', sep='::', engine='python',
                     names=['movie_id','movie_title','genres'])
data = pd.merge(ratings, movies, on='movie_id')
print("Merged Data Preview:")
print(data.head())

# ================================
# Step 2: Exploratory Data Analysis
# ================================
print("\nRating Statistics:")
print(data['rating'].describe())

print("\nTop Genres:")
print(data['genres'].value_counts().head())

ratings_per_movie = data.groupby('movie_title')['rating'].count().sort_values(ascending=False)
print("\nTop 5 movies by number of ratings:")
print(ratings_per_movie.head())

# ================================
# Step 3: Prepare data for Surprise
# ================================
reader = Reader(rating_scale=(1,5))
dataset = Dataset.load_from_df(data[['user_id','movie_id','rating']], reader)
trainset, testset = train_test_split(dataset, test_size=0.2, random_state=42)

# ================================
# Step 4: Train SVD model
# ================================
algo = SVD()
algo.fit(trainset)
predictions = algo.test(testset)
print("\nRMSE on test set:")
rmse(predictions)

# ================================
# Step 5: Generate recommendations for a user
# ================================
user_id = 1  # Example user
user_movies = data[data['user_id'] == user_id]['movie_id'].tolist()
all_movies = data['movie_id'].unique()
predictions_list = []

for movie in all_movies:
    if movie not in user_movies:
        pred = algo.predict(user_id, movie)
        predictions_list.append((movie, pred.est))

top_5 = sorted(predictions_list, key=lambda x: x[1], reverse=True)[:5]
recommended_movies = [movies[movies['movie_id']==m[0]]['movie_title'].values[0] for m in top_5]
print("\nTop 5 movie recommendations for user 1:")
print(recommended_movies)


## 9. Project Timeline

| Week | Activity |
|-----|---------|
| Week 1 | Project planning and dataset selection |
| Week 2 | Data cleaning, preprocessing, and exploratory analysis |
| Week 3 | Building recommendation algorithms and ML models |
| Week 4 | Model evaluation, insights extraction, visualization |
| Week 5 | Documentation, final report preparation, submission |



## 10. Expected Outcomes
- An organized analysis of movie ratings  
- A working recommendation algorithm (proof-of-concept)  
- Visualizations and insights into user preferences  
- Comprehensive project documentation



## 11. Conclusion
This project demonstrates the application of data analytics and machine learning to solve a real-world problem of personalized movie recommendations. By following a structured approach, the project will provide actionable insights and a functional recommendation model.



## 12. References

- Python Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn  
- Academic materials and lecture notes on Data Analytics & ML




