import pandas as pd

# Load CSV files
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('rating.csv')

# Merge both files on movieId
data = pd.merge(ratings, movies, on='movieId')
# Check for missing values
print(data.isnull().sum())

# Create a pivot table for collaborative filtering
user_movie_matrix = data.pivot_table(index='userId', columns='title', values='rating')
user_movie_matrix.fillna(0, inplace=True)

# Removed the leading whitespace from this line
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

movies['genres'] = movies['genres'].fillna('')
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def get_recommendations(title):
    idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]

print(get_recommendations('Toy Story (1995)'))

# Removed the leading whitespace from this line as well
from sklearn.neighbors import NearestNeighbors

model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(user_movie_matrix.values)

# Example: Recommendations for user 1
# Changed n_neighbors from 6 to 4 to be less than or equal to n_samples_fit (4)
distances, indices = model_knn.kneighbors([user_movie_matrix.iloc[0].values], n_neighbors=4)
print(indices)

import matplotlib.pyplot as plt
import seaborn as sns

top_movies = data['title'].value_counts().head(10)

plt.figure(figsize=(10,5))
sns.barplot(x=top_movies.values, y=top_movies.index)
plt.title("Top 10 Most Rated Movies")
plt.xlabel("Ratings Count")
plt.ylabel("Movie Title")
plt.show()
