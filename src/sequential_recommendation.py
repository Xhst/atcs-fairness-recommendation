from collections import defaultdict
from group_recommendation import GroupRecommendation
import operator

class SequentialRecommendation:
    def __init__(self, group_recommendation: GroupRecommendation):
        self.group_recommendation = group_recommendation


    def get_sequential_recommendations_for_group(self, user_group: set[int], iterations: int = 5) -> dict[int, list[tuple[int, float]]]:
        """
        Get sequential recommendations of certain length for a group of users.

        Args:
            user_group (list[int]): List of user IDs.
            ser_len (int): Amount of group recommendations in a sequential for a group. Defaults to 5.

        Returns:
            list[tuple[int, float]]: list of tuples containing sequential ID and the recommendetions for the group.
        """
        # We don't want to recommend movies from previous iterations
        already_recommended = []

        # sequential_recommendations: iteration -> list[(movieId, rating)]
        sequential_recommendations: dict[int, list[tuple[int, float]]] = defaultdict(list[tuple[int, float]])

        # satisfactions: iteration -> list[(userId, satisfaction)]
        satisfactions: dict[int, list[tuple[int, float]]] = defaultdict(list[tuple[int, float]])

        for i in range(iterations):
            # Aggregate recommendations for the current iteration, excluding movies from previous iterations
            users_rec = self.group_recommendation.users_top_recommendations(user_group, exclude_movies=already_recommended)
            aggreg_rec = self.group_recommendation.aggregate_users_recommendations(users_rec)

            if i == 0:
                group_rec = self.group_recommendation.weighted_average_aggregation_from_users_recommendations(aggreg_rec)
            else:
                group_rec = self.aggregation_from_users_recommendations_and_satisfaction(aggreg_rec, satisfactions[i-1])

            satisfactions[i] = self.calculate_satisfactions(user_group, users_rec, group_rec, aggreg_rec)

            already_recommended.extend(movie for movie, _ in group_rec)
            sequential_recommendations[i] = group_rec
        
        return sequential_recommendations, satisfactions
    

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


    def aggregation_from_users_recommendations_and_satisfaction(self, aggreg_rec: dict[int, list[float]],
                                                                satisfactions: list[tuple[int, float]], n: int = 10) -> list[tuple[int, float]]:
            aggregated_ratings = defaultdict(float)
            satisfaction_weights = [sat for _, sat in satisfactions]

            # Weighted aggregation of recommendations based on user satisfaction
            for movie_id, ratings in aggreg_rec.items():
                for user_id, rating in enumerate(ratings):
                    aggregated_ratings[movie_id] += rating * (1 - satisfaction_weights[user_id])

            # Sort recommendations by aggregated weighted rating
            sorted_recommendations = sorted(aggregated_ratings.items(), key=lambda x: x[1], reverse=True)

            return sorted_recommendations[:n]
    