import numpy as np
import pandas as pd
import sys # can use sys to take command line arguments

class Recommender:
    """
    Class to provide movie recommendations based on FunkSVD, content-based and knowledge based recommendation.
    FunkSVD works with Matrix Factorization and uses latent features.
    Content-based recommendation recommends the movies, that are similar to those the user rated high before.
    Knowledge-based recommendation recommends the highest rated movies.
    """
    def __init__(self, ):
        """
        Initizalize Recommender
        """


    def fit(self, movies_path, reviews_path, latent_features = 15, learning_rate = 0.001, iters = 50):
        """
        fit the recommender to your dataset and also have this save the results
        to pull from when you need to make predictions
        """

        # Read in the data
        movies = pd.read_csv(movies_path)
        reviews = pd.read_csv(reviews_path)

        # Create user-by-item matrix - nothing to do here
        train_user_item = reviews[['user_id', 'movie_id', 'rating', 'timestamp']]
        train_data_df = train_user_item.groupby(['user_id', 'movie_id'])['rating'].max().unstack()
        train_data_np = np.array(train_data_df)




    def predict_rating(self, ):
        """
        makes predictions of a rating for a user on a movie-user combo
        """

    def make_recs(self,):
        """
        given a user id or a movie that an individual likes
        make recommendations
        """


if __name__ == '__main__':
    # test different parts to make sure it works
