import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gzip  # Added this
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

st.title("🎬 Movie Recommendation System")

# Functions
def fix_title(title):
    """Move 'The', 'A', 'An' from end to beginning"""
    if ', The (' in title:
        title = 'The ' + title.replace(', The (', ' (')
    elif ', A (' in title:
        title = 'A ' + title.replace(', A (', ' (')
    elif ', An (' in title:
        title = 'An ' + title.replace(', An (', ' (')
    return title

def find_similar_movies(movie_id, X, movie_mapper, movie_inv_mapper, k, metric='cosine'):
    X_T = X.T
    neighbour_ids = []
    
    movie_ind = movie_mapper[movie_id]
    movie_vec = X_T[movie_ind]
    if isinstance(movie_vec, np.ndarray):
        movie_vec = movie_vec.reshape(1,-1)
    
    kNN = NearestNeighbors(n_neighbors=k+1, algorithm="brute", metric=metric)
    kNN.fit(X_T)
    neighbour = kNN.kneighbors(movie_vec, return_distance=False)
    
    for i in range(1, k+1):
        n = neighbour.item(i)
        neighbour_ids.append(movie_inv_mapper[n])
    
    return neighbour_ids

# Load compressed preprocessed data
@st.cache_resource
def load_preprocessed_data():
    with gzip.open('X_matrix.pkl.gz', 'rb') as f:
        X = pickle.load(f)
    
    with gzip.open('Q_matrix.pkl.gz', 'rb') as f:
        Q = pickle.load(f)
    
    with gzip.open('movie_mapper.pkl.gz', 'rb') as f:
        movie_mapper = pickle.load(f)
    
    with gzip.open('movie_inv_mapper.pkl.gz', 'rb') as f:
        movie_inv_mapper = pickle.load(f)
    
    with gzip.open('movies_filtered.pkl.gz', 'rb') as f:
        movies = pickle.load(f)
    
    return X, Q, movie_mapper, movie_inv_mapper, movies

# Load
X, Q, movie_mapper, movie_inv_mapper, movies = load_preprocessed_data()
movie_titles = dict(zip(movies['movieId'], movies['title']))

# Sidebar for settings
st.sidebar.header("Settings")
model = st.sidebar.radio("Select Model:", ["Original Matrix", "SVD Model"])
num_recs = st.sidebar.slider("Number of recommendations:", 5, 20, 10)

# Main input
movie_name = st.text_input("Enter a movie name:")

# Get recommendations
if st.button("Get Recommendations"):
    if movie_name:
        movie_matches = movies[movies['title'].str.contains(movie_name, case=False, na=False)]
        
        if movie_matches.empty:
            st.error(f"Movie '{movie_name}' not found.")
        else:
            movie_id = movie_matches.iloc[0]['movieId']
            movie_title = movie_matches.iloc[0]['title']
            
            if movie_id not in movie_mapper:
                st.error(f"Movie has no ratings data.")
            else:
                if model == "SVD Model":
                    similar = find_similar_movies(movie_id, Q.T, movie_mapper, movie_inv_mapper, k=num_recs)
                else:
                    similar = find_similar_movies(movie_id, X, movie_mapper, movie_inv_mapper, k=num_recs)
                
                st.write(f"\n**Because you watched: {fix_title(movie_title)}**\n")
                for i, mid in enumerate(similar, 1):
                    st.text(f"{i}. {fix_title(movie_titles[mid])}")
    else:
        st.warning("Please enter a movie name.")