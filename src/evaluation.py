import pandas as pd
import numpy as np
import math
from dataset import Dataset
from user_recommendation import UserRecommendation
from collections import defaultdict 
from concurrent.futures import ThreadPoolExecutor

class Evaluation:

    def split_dataset(df: pd.DataFrame, percentage: float, shuffle: bool = True):
        percentage = max(0, min(1, percentage))
        
        if shuffle:
            df = df.sample(frac=1)

        df_size = len(df)
        training_size = int(df_size * percentage)

        return (df.iloc[:training_size], df.iloc[training_size:])
    

    def evaluate_similarities(similarities, training_set, test_set, k_range) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
        mae_points = defaultdict(list)
        rmse_points = defaultdict(list)

        with ThreadPoolExecutor() as executor:
            futures = []

            for sim in similarities:
                for k in k_range:
                    futures.append(executor.submit(Evaluation.evaluate, training_set, test_set, sim, k))

            for future in futures:
                sim, mae, rmse = future.result()
                mae_points[sim].append(mae)
                rmse_points[sim].append(rmse)

        return mae_points, rmse_points


    def evaluate(training_df: pd.DataFrame, test_df: pd.DataFrame, similarity_function, neighbor_size) -> tuple[str, int, float, float]:
        training_ds = Dataset(training_df)
        test_ds = Dataset(test_df)

        training_user_rec = UserRecommendation(training_ds)

        num_errors = 0
        mae_prediction_errors = 0
        rmse_prediction_error = 0

        for test_user, movie_to_ratings in test_ds._user_to_movie_ratings.items():
            if not training_ds.has_user(test_user):
                continue
            
            if similarity_function == 'jaccard':
                sim = training_user_rec.sim_jaccard
            elif similarity_function == 'cosine':
                sim = training_user_rec.sim_cosine
            elif similarity_function == 'acosine':
                sim = training_user_rec.sim_acosine
            elif similarity_function == 'manhattan':
                sim = training_user_rec.sim_manhattan
            elif similarity_function == 'chebyshev':
                sim = training_user_rec.sim_chebyshev
            elif similarity_function == 'euclidean':
                sim = training_user_rec.sim_euclidean
            elif similarity_function == 'pcc_jaccard':
                sim = training_user_rec.sim_wpcc_jaccard
            elif similarity_function == 'acosine_jaccard':
                sim = training_user_rec.sim_acosine_jaccard
            else:
                sim = training_user_rec.sim_pcc

            neighbors = training_user_rec.top_n_similar_users(test_user, sim, neighbor_size)
            
            for movie, real_rating in movie_to_ratings.items():
                pred_rating = training_user_rec.prediction_from_neighbors(test_user, movie, neighbors)
                
                # Clip the prediction to the max possible rating value to avoid inflated errors
                pred_rating = min(pred_rating, 5.0)

                mae_prediction_errors += abs(pred_rating - real_rating)
                rmse_prediction_error += (pred_rating - real_rating) ** 2
                num_errors += 1

        if num_errors == 0: 
            return similarity_function, neighbor_size, 0, 0

        mae = mae_prediction_errors / num_errors
        rmse = math.sqrt(rmse_prediction_error / num_errors)

        return similarity_function, mae, rmse

                