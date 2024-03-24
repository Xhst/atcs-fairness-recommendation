import operator
from typing import Callable
import numpy as np
from user_recommendation import UserRecommendation
from collections import defaultdict 

class GroupRecommendation:
    
    def __init__(self, user_recommendation: UserRecommendation) -> None:
        self.user_recommendation = user_recommendation


    def users_top_recommendations(self, users: set[int], n: int = 10, neighbor_size: int = 50, exclude_movies: set[int] = set()):
        # userId -> list[(movieId, rating)]
        users_recommendations: dict[int, list[tuple[int, float]]] = defaultdict(list[tuple[int, float]])

        for user in users:
            users_recommendations[user] = self.user_recommendation.top_n_recommendations(user, n=n, neighbor_size=neighbor_size, exclude_movies=exclude_movies)

        return users_recommendations
    

    def aggregate_users_recommendations(self, users_top_rec: dict[int, list[tuple[int, float]]], n: int = 10, 
                                        neighbor_size: int = 50) -> dict[int, list[float]]:
        movies: set[int] = set()

        # movie -> list[user ratings]
        aggregate_recommendations: dict[int, list[float]] = defaultdict(list[float])

        # find top n recommended movies for each user
        for top_rec in users_top_rec.values():
            for (movie, _) in top_rec:
                movies.add(movie)

        # aggregate predictions for each movie
        for user in users_top_rec.keys():
            neighbors = self.user_recommendation.top_n_similar_users(user, n=neighbor_size)
            for movie in movies:
                # Take the rating from the user if the user has rated the movie
                if self.user_recommendation.dataset.has_user_rated_movie(user, movie):
                    aggregate_recommendations[movie].append(self.user_recommendation.dataset.get_rating(user, movie))
                    continue
                # Otherwise, predict the rating
                aggregate_recommendations[movie].append(self.user_recommendation.prediction_from_neighbors(user, movie, neighbors))

        return aggregate_recommendations

    
    def average_aggregation(self, users: set[int], n: int = 10) -> list[tuple[int, float]]:
        """
        Rank recommendations by averaging predicted ratings.
        It produces better overall satisfacrion than the least misery aggregation.

        Args:
            users (set[int]): Set of user IDs.
            n (int): Number of recommendations to return. Defaults to 10.

        Returns:
            list[tuple[int, float]]: List of tuples containing movie ID and average predicted rating.
        """
        aggreg_rec = self.aggregate_users_recommendations(users)

        return self.average_aggregation_from_users_recommendations(aggreg_rec, n)
    
    
    def average_aggregation_from_users_recommendations(self, aggreg_rec: dict[int, list[float]], n: int = 10) -> list[tuple[int, float]]:
        avg_rec: list[tuple[int, float]] = []

        for movie, predicted_ratings in aggreg_rec.items():
            average = sum(predicted_ratings) / len(predicted_ratings)
            avg_rec.append((movie, average))
        
        avg_rec.sort(key=lambda x: x[1], reverse=True)
        
        return avg_rec[:n]

    
    def least_misery_aggregation(self, users: set[int], n: int = 10) -> list[tuple[int, float]]:
        """
        Rank recommendations by selecting the minimum predicted rating.
        In theory, less disagreement among users than the average aggregation.

        Args:
            users (set[int]): Set of user IDs.
            n (int): Number of recommendations to return. Defaults to 10.

        Returns:
            list[tuple[int, float]]: List of tuples containing movie ID and minimum predicted rating.
        """
        aggreg_rec = self.aggregate_users_recommendations(users)

        return self.least_misery_aggregation_from_users_recommendations(aggreg_rec, n)
    
    
    def least_misery_aggregation_from_users_recommendations(self, aggreg_rec: dict[int, list[float]], n: int = 10) -> list[tuple[int, float]]:
        least_misery_rec: list[tuple[int, float]] = []

        for movie, predicted_ratings in aggreg_rec.items():
            least_misery_rec.append((movie, min(predicted_ratings)))
        
        least_misery_rec.sort(key=lambda x: x[1], reverse=True)
        
        return least_misery_rec[:n]
    

    def get_disagreement(self, ratings: list[float]) -> float:
        """
        Calculate the disagreement among ratings.

        This method computes the disagreement among ratings, which is the standard deviation
        of the ratings.

        Args:
            ratings (list[float]): List of ratings.

        Returns:
            float: Disagreement, calculated as the standard deviation of the ratings.
        """
        return np.std(ratings)
    

    def get_disagreement_weight(self, ratings: list[float]) -> float:
        """
        Calculate the disagreement weight based on the standard deviation of ratings.

        This method computes the disagreement weight, which represents the inverse of
        the standard deviation of a list of ratings.

        Args:
            ratings (list[float]): List of ratings.

        Returns:
            float: Disagreement weight, calculated as 1 divided by the standard deviation
                of the ratings, with a small constant added to avoid division by zero.
        """
        return 1 / (self.get_disagreement(ratings) + 0.0001)
    
    
    def weighted_average_aggregation(self, users: set[int], n: int = 10) -> list[tuple[int, float]]:
        """
        Rank recommendations by weighted average of predicted ratings.
        It is a compromise between average and least misery aggregations.
        We take into account the disagreement among users. We calculate the disagreement weight
        as 1 over the standard deviation of the predicted ratings.

        Args:
            users (set[int]): Set of user IDs.
            n (int): Number of recommendations to return. Defaults to 10.

        Returns:
            list[tuple[int, float]]: List of tuples containing movie ID and weighted average predicted rating.
        """
        aggreg_rec = self.aggregate_users_recommendations(users)

        return self.weighted_average_aggregation_from_users_recommendations(aggreg_rec, n)


    def weighted_average_aggregation_from_users_recommendations(self, aggreg_rec: dict[int, list[float]], n: int = 10) -> list[tuple[int, float]]:
        w_avg_rec: list[tuple[int, float]] = []

        for movie, predicted_ratings in aggreg_rec.items():
            disagreement_weight = self.get_disagreement_weight(predicted_ratings)
 
            average = sum(predicted_ratings) / len(predicted_ratings)
            w_average = average * disagreement_weight
            w_avg_rec.append((movie, w_average))
        
        w_avg_rec.sort(key=lambda x: x[1], reverse=True)
        
        return w_avg_rec[:n]
    
    
    def get_recommendations_satisfactions_and_disagreements_for_group(self, user_group: set[int], aggreg_method: Callable = None) -> dict[int, list[tuple[int, float]]]:
        satisfactions: list[tuple[int, float]] = []

        if aggreg_method == None:
            aggreg_method = self.weighted_average_aggregation_from_users_recommendations

        users_rec = self.users_top_recommendations(user_group)
        aggreg_rec = self.aggregate_users_recommendations(users_rec)
        
        group_rec = aggreg_method(aggreg_rec)

        # (user, satisfaction) tuples
        satisfactions = self.calculate_satisfactions(user_group, users_rec, group_rec, aggreg_rec)

        # (movie, disagreement) tuples
        disagreements = [(movie, self.get_disagreement(aggreg_rec[movie])) for movie, _ in group_rec]
        
        return group_rec, satisfactions, disagreements
    

    def calculate_satisfactions(self, group: set[int], users_rec: dict[int, list[tuple[int, float]]], group_rec: list[tuple[int, float]], 
                                aggreg_rec: dict[int, list[float]]):
        satisfactions: list[tuple[int, float]] = []
        
        group_movies = set(map(operator.itemgetter(0), group_rec))
        
        for index, user in enumerate(group):
            group_val = sum(aggreg_rec[movie][index] for movie in group_movies)
            user_val = sum(rating for _, rating in users_rec[user])
            user_sat = group_val / user_val

            satisfactions.append((user, user_sat))

        return satisfactions
