from user_recommendation import UserRecommendation
import dataset
import math 

class GroupRecommendation:

    def aggregate_users_recommendations(users: set[int], n: int = 10, neighbor_size: int = 50):

        aggregate_recommendations: dict[int, list[float]] = {}

        for user in users:
            user_recommendations = UserRecommendation.top_n_recommendations(user, n, neighbor_size)

            for (movie, predicted_rating) in user_recommendations:
                if aggregate_recommendations.get(movie) == None:
                    aggregate_recommendations[movie] = [predicted_rating]
                else:
                    aggregate_recommendations.get(movie).append(predicted_rating)

        return aggregate_recommendations
