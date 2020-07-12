# Class: Recommender

A class to provide movie recommendations based on FunkSVD, content-based and knowledge based recommendation.
* FunkSVD works with Matrix Factorization and uses latent features.
* Knowledge-based recommendation recommends the highest rated movies.

### File description
This repository consists of the following files: 
* movies_clean.csv: A CSV with movies data
* train_data.csv: A CSV with user_ids, movie_ids and corresponding ratings. 
* recommender.py: The "solution" file for this task that was provided by Udacity
* recommender_functions.py: A Python file that includes necessary functions for the recommender, e.g. create a DataFrame with movies ranked by rating, find similar movies and convert movie_ids to movie names. 
* recommender_template.py: My own approach to build a movie recommender. 


### Installation
After importing the class, you can access and test it as follows: 

```
    rec = Recommender()

    # fit recommender
    rec.fit(reviews_path='train_data.csv', movies_path='movies_clean.csv', learning_rate=.01, iters=1)

    # predict
    rec.predict_rating(user_id=8, movie_id=2844)

    # make recommendations
    print(rec.make_recs(8, 'user'))  # user in the dataset
    print(rec.make_recs(1, 'user'))  # user not in dataset
    print(rec.make_recs(1853728))  # movie in the dataset
    print(rec.make_recs(1))  # movie not in dataset
    print(rec.n_users)
    print(rec.n_movies)
    print(rec.num_ratings)
```

### Acknowledgments
Thanks to Udacity and MovieTweetings for providing this dataset! 


### Author

Maximilian Müller, Business Development Manager in the Renewable Energy sector. Now diving into the field of data analysis. 


### License
Copyright 2020 Maximilian Müller

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the 
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit 
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the 
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

From opensource.org