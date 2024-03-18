import pandas as pd
import numpy as np
from dataset import Dataset
from user_recommendation import UserRecommendation

class Evaluation:

    def split_dataset(df: pd.DataFrame, percentage: float, shuffle: bool = False):
        percentage = max(0, min(1, percentage))
        
        if shuffle:
            df = df.sample(frac=1)

        df_size = len(df)
        training_size = int(df_size * percentage)
        test_size = df_size - training_size

        return (df.iloc[:training_size], df.iloc[test_size:])
    

    def evaluate(training_df: pd.DataFrame, test_df: pd.DataFrame, similarity_function, neighbor_size):
        #for each (userId,movieId,rating) in test_df
        #    sim with userId and user2 in training_df
        #    prediction user in test_df
        #    for each prediction measure Error (MSE, MRSE, ACCURACY, ROC, AUC)
        # order sim with Error

        

        training_ds = Dataset(training_df)
        test_ds = Dataset(test_df)

        training_user_rec = UserRecommendation(training_ds)

        prediction_errors = []

        for test_user, movie_to_ratings in test_ds._user_to_movie_ratings.items():
            if not training_ds.has_user(test_user):
                continue

            neighbors = training_user_rec.top_n_similar_users(test_user, training_user_rec.sim_wpcc_jaccard, neighbor_size)
            
            for movie, real_rating in movie_to_ratings.items():
                pred_rating = training_user_rec.prediction_from_neighbors(test_user, movie, neighbors)
                prediction_errors.append(abs(pred_rating - real_rating))

        return np.mean(prediction_errors)


                
