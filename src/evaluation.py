import pandas as pd
import numpy as np
import math
from dataset import Dataset
from user_recommendation import UserRecommendation
from collections import defaultdict 
from threading import Thread


class Evaluation:

    def split_dataset(df: pd.DataFrame, n_splits: int, shuffle: bool = True) -> list[pd.DataFrame]:
        
        if shuffle:
            df = df.sample(frac=1)
            
        split_size = len(df) // n_splits
        splits = [df[i:i+split_size] for i in range(0, len(df), split_size)]
        
        # If there are elements left over, add them to the last split
        if len(splits) > n_splits:
            splits[-1] = pd.concat([splits[-1], splits[-2]])
            splits.pop(-2)
        
        return splits


    def evaluate_sim(sim: str, mae_points: dict[str, list[float]], rmse_points: dict[str, list[float]], splits: list[pd.DataFrame]):
        for k in range(5, 55, 5):
            test_index = int((k/5 - 1) % len(splits))
            print(f"Testing similarity {sim} with k={k} on test split {test_index}")
            current_test = splits[test_index]
            current_training = pd.concat([splits[i] for i in range(len(splits)) if i != test_index])
            
            mae, rmse = Evaluation.evaluate(current_training, current_test, sim, k)
            mae_points[sim].append(mae)
            rmse_points[sim].append(rmse)


    def evaluate_similarities(similarities, splits: list[pd.DataFrame]) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
        mae_points = defaultdict(list)
        rmse_points = defaultdict(list)

        threads = []

        for sim in similarities:
            thread = Thread(target=Evaluation.evaluate_sim, args=(sim, mae_points, rmse_points, splits))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        return mae_points, rmse_points


    def evaluate(training_df: pd.DataFrame, test_df: pd.DataFrame, similarity_function, neighbor_size) -> tuple[float, float]:
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
            elif similarity_function == 'pcc_jaccard':
                sim = training_user_rec.sim_wpcc_jaccard
            else:
                sim = training_user_rec.sim_pcc

            neighbors = training_user_rec.top_n_similar_users(test_user, sim, neighbor_size)
            
            for movie, real_rating in movie_to_ratings.items():
                pred_rating = training_user_rec.prediction_from_neighbors(test_user, movie, neighbors)
                
                pred_rating = min(5, pred_rating)
                
                mae_prediction_errors += abs(pred_rating - real_rating)
                rmse_prediction_error += (pred_rating - real_rating) ** 2
                num_errors += 1

        if num_errors == 0: return 0, 0

        mae = mae_prediction_errors / num_errors
        rmse = math.sqrt(rmse_prediction_error / num_errors)

        return mae, rmse


                
