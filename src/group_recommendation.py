from user_recommendation import UserRecommendation
import dataset
import math 

class GroupRecommendation:

    def aggregate_users_recommendations(users: set[int], n: int = 10, neighbor_size: int = 50) -> dict[int, list[float]]:

        aggregate_recommendations: dict[int, list[float]] = {}

        for user in users:
            user_recommendations = UserRecommendation.top_n_recommendations(user, n, neighbor_size)

            for (movie, predicted_rating) in user_recommendations:
                if aggregate_recommendations.get(movie) == None:
                    aggregate_recommendations[movie] = [predicted_rating]
                else:
                    aggregate_recommendations.get(movie).append(predicted_rating)

        return aggregate_recommendations
    

    def avarage_aggregation(users: set[int], n: int = 10) -> list[(int, float)]:
        users_rec = GroupRecommendation.aggregate_users_recommendations(users)

        avg_rec: list[(int, float)] = []

        for movie, predicted_ratings in users_rec.items():
            avarage = sum(predicted_ratings) / len(predicted_ratings)
            avg_rec.append((movie, avarage))

        return avg_rec[:n]
    

    def least_misery_aggregation(users: set[int], n: int = 10) -> list[(int, float)]:
        users_rec = GroupRecommendation.aggregate_users_recommendations(users)

        least_misery_rec: list[(int, float)] = []

        for movie, predicted_ratings in users_rec.items():
            least_misery_rec.append((movie, min(predicted_ratings)))

        return least_misery_rec[:n]
