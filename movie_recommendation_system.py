#importing the dependencies
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#loading data from csv file to pandas dataframe
df = pd.read_csv("C:\\Users\\HP\\Downloads\\movies.csv", encoding ='utf-8')
#looking for missing data
selected_features = ['genres', 'keywords', 'tagline','cast','director']
#replacing the null value with null string
for feature in selected_features:
    df[feature] = df[feature].fillna('')
#combining 5 selected features
combined_features = df['genres']+' '+df['keywords']+' '+df['tagline']+' '+df['cast']+' '+df['director']
#converting the text data into feature vectors
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
#cosine similarity
#getting the similarity score using cosine similarity
similarity = cosine_similarity(feature_vectors)
movie_name = input("Enter favorite movie name:")
#creating a list with all the movies given in the dataset
list_of_all_titles = df['title'].tolist()
#finding a close match for the movie names given by the user
find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
close_match = find_close_match[0]
#find the index of the movie
movie_index = df[df.title == close_match]['index'].values[0]
#getting a list of similar movies
similarity_score = list(enumerate(similarity[movie_index]))
#sorting the movies based on their similarity score
sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True)
#print the name of similar movies based on index
print("Recommended Movies: \n")
i = 1
for movie in sorted_similar_movies:
  index = movie[0]
  title_from_index = df[df.index == index]['title'].values[0]
  if i<30:
    print(i, '.', title_from_index)
    i += 1






















