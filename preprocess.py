import pandas as pd
import numpy as np
import pickle
import gzip  # Added this
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

def create_X(df):
    M = df['userId'].nunique()
    N = df['movieId'].nunique()
    user_mapper = dict(zip(np.unique(df["userId"]), list(range(M))))
    movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(N))))
    user_inv_mapper = dict(zip(list(range(M)), np.unique(df["userId"])))
    movie_inv_mapper = dict(zip(list(range(N)), np.unique(df["movieId"])))
    
    user_index = [user_mapper[i] for i in df['userId']]
    item_index = [movie_mapper[i] for i in df['movieId']]
    X = csr_matrix((df["rating"], (user_index,item_index)), shape=(M,N))
    
    return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper

print("Loading data...")
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')

print("Filtering data...")
movie_counts = ratings['movieId'].value_counts()
user_counts = ratings['userId'].value_counts()

active_movies = movie_counts[movie_counts > 100].index
ratings = ratings[ratings['movieId'].isin(active_movies)]

active_users = user_counts[user_counts > 50].index
ratings = ratings[ratings['userId'].isin(active_users)]

print("Creating matrix...")
X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_X(ratings)

print("Running SVD...")
svd = TruncatedSVD(n_components=20, n_iter=10)
Q = svd.fit_transform(X.T)

print("Saving compressed preprocessed data...")
# Save with gzip compression
with gzip.open('X_matrix.pkl.gz', 'wb') as f:
    pickle.dump(X, f)

with gzip.open('Q_matrix.pkl.gz', 'wb') as f:
    pickle.dump(Q, f)

with gzip.open('movie_mapper.pkl.gz', 'wb') as f:
    pickle.dump(movie_mapper, f)

with gzip.open('movie_inv_mapper.pkl.gz', 'wb') as f:
    pickle.dump(movie_inv_mapper, f)

with gzip.open('movies_filtered.pkl.gz', 'wb') as f:
    pickle.dump(movies, f)

print("Done! All compressed files saved.")
print(f"Total ratings: {len(ratings):,}")
print(f"Total movies: {len(movies):,}")