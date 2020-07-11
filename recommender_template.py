import numpy as np
import pandas as pd
import recommender_functions as rf
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
        self.movies = pd.read_csv(movies_path)
        self.reviews = pd.read_csv(reviews_path)

        # Create user-by-item matrix - nothing to do here
        train_user_item = self.reviews[['user_id', 'movie_id', 'rating', 'timestamp']]
        self.train_data_df = train_user_item.groupby(['user_id', 'movie_id'])['rating'].max().unstack()
        self.train_data_np = np.array(self.train_data_df)

        # Set up useful values to be used through the rest of the function
        self.n_users = self.train_data_np.shape[0]
        self.n_movies = self.train_data_np.shape[1]
        self.num_ratings = np.count_nonzero(~np.isnan(self.train_data_np))

        # initialize the user and movie matrices with random values
        user_mat = np.random.rand(self.users, self.latent_features)
        movie_mat = np.random.rand(self.latent_features, self.n_movies)

        # initialize sse at 0 for first iteration
        sse_accum = 0

        # keep track of iteration and MSE
        print("Optimization Statistics")
        print("Iterations | Mean Squared Error ")

        # for each iteration
        for iteration in range(iters):

            # update our sse
            sse_accum = 0

            # For each user-movie pair
            for i in range(self.n_users):
                for j in range(self.n_movies):

                    # if the rating exists
                    if self.train_data_np[i, j] > 0:

                        # compute the error as the actual minus the dot product of the user and movie latent features
                        diff = self.train_data_np[i, j] - np.dot(user_mat[i, :], movie_mat[:, j])

                        # Keep track of the sum of squared errors for the matrix
                        sse_accum += diff ** 2

                        # update the values in each matrix in the direction of the gradient
                        for k in range(latent_features):
                            user_mat[i, k] += learning_rate * (2 * diff * movie_mat[k, j])
                            movie_mat[k, j] += learning_rate * (2 * diff * user_mat[i, k])

            # print results
            print("%d \t\t %f" % (iteration + 1, sse_accum / self.num_ratings))

        # SVD approach:
        self.user_mat = user_mat
        self.movie_mat = movie_mat

        # Content-based approach:
        self.ranked_movies = rf.create_ranked_df(self.movies, self.reviews)


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
