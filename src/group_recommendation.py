import numpy as np
from user_recommendation import UserRecommendation
from collections import defaultdict 

class GroupRecommendation:
    
    def __init__(self, user_recommendation: UserRecommendation) -> None:
        self.user_recommendation = user_recommendation

    
    def aggregate_users_recommendations(self, users: set[int], n: int = 10, neighbor_size: int = 50) -> dict[int, list[float]]:
        """
        Aggregate recommendations from a set of users.

        Args:
            users (set[int]): Set of user IDs.
            n (int): Number of recommendations per user. Defaults to 10.
            neighbor_size (int): Size of the neighborhood for recommendation. Defaults to 50.

        Returns:
            dict[int, list[float]]: Dictionary where keys are movie IDs and values are lists of predicted ratings.
        """
        movies: set[int] = set()
        aggregate_recommendations: dict[int, list[float]] = defaultdict(list[float])

        # find top n recommended movies for each user
        for user in users:
            user_recommendations = self.user_recommendation.top_n_recommendations(user, n=n, neighbor_size=neighbor_size)

            for (movie, _) in user_recommendations:
                movies.add(movie)

        # aggregate predictions for each movie
        for user in users:
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
        users_rec = self.aggregate_users_recommendations(users)

        avg_rec: list[tuple[int, float]] = []

        for movie, predicted_ratings in users_rec.items():
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
        users_rec = self.aggregate_users_recommendations(users)

        least_misery_rec: list[tuple[int, float]] = []

        for movie, predicted_ratings in users_rec.items():
            least_misery_rec.append((movie, min(predicted_ratings)))

        least_misery_rec.sort(key=lambda x: x[1], reverse=True)

        return least_misery_rec[:n]
    

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
        return 1 / (np.std(ratings) + 0.0001)
    
    
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
        users_rec = self.aggregate_users_recommendations(users)

        w_avg_rec: list[tuple[int, float]] = []

        for movie, predicted_ratings in users_rec.items():
            disagreement_weight = self.get_disagreement_weight(predicted_ratings)
 
            average = sum(predicted_ratings) / len(predicted_ratings)
            w_average = average * disagreement_weight
            w_avg_rec.append((movie, w_average))
        
        w_avg_rec.sort(key=lambda x: x[1], reverse=True)
        
        return w_avg_rec[:n]

