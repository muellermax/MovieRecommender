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
        Initizalizes Recommender, no necessary args.
        """


    def fit(self, movies_path, reviews_path, latent_features = 15, learning_rate = 0.001, iters = 50):
        """
        Fits recommender to dataset, using the FunkSVD and knowledge-based approach.

        Args:
            movies_path: Path of CSV file with movies data with necessary columns 'movie', 'rating', 'date'
            reviews_path: Path of CSV file with reviews (ratings) data with necessary columns 'user_id', 'movie_id',
            'rating', 'timestamp'
            latent_features: Number of latent features (for FunkSVD) to be considered
            learning_rate: Learning rate for FunkSVD
            iters: Iterations of FunkSVD to find best user_mat and movies_mat

        Returns:
            None - stores the following attributes
            n_users - the number of users (int)
            n_movies - the number of movies (int)
            num_ratings - the number of ratings made (int)
            reviews - DataFrame with four columns: 'user_id', 'movie_id', 'rating', 'timestamp'
            movies - DataFrame of movies
            user_item_mat - (np array) a user by item numpy array with ratings and nans for values
            user_mat - Matrix with number of users (rows) and latent features (columns)
            movies_mat - Matrix with number of movies (columns) and latent features (rows)
            ranked_movies - DataFrame with with movies that are sorted by highest avg rating, more reviews,
            then time, and must have more than 4 ratings
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

        # Knowledge-based approach:
        self.ranked_movies = rf.create_ranked_df(self.movies, self.reviews)


    def predict_rating(self, user_id, movie_id):
        """
        makes predictions of a rating for a user on a movie-user combo
        """

        if user_id in self.train_data_df.index:

            # Use the training data to create a series of users and movies that matches the ordering in training data
            user_ids_series = np.array(self.train_data_df.index)
            movie_ids_series = np.array(self.train_data_df.columns)
            movie_name = self.movies[self.movies.movie_id == movie_id]['movies'][0]

            # User row and Movie Column
            user_row = np.where(user_ids_series == user_id)[0][0]
            movie_col = np.where(movie_ids_series == movie_id)[0][0]

            # Take dot product of that row and column in U and V to make prediction
            pred = np.dot(self.user_mat[user_row, :], self.movie_mat[:, movie_col])

            print('The predicted rating for user {} and the movie {} is {}'.format(user_id, movie_name, round(pred, 2)))

        else:
            print('Looks like for this user-movie pair no prediction could be made, because the user and/or the '
                  'movie are in the user-movie-matrix.')


    def make_recs(self,):
        """
        given a user id or a movie that an individual likes
        make recommendations
        """

        


if __name__ == '__main__':
    # test different parts to make sure it works
