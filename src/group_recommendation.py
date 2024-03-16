from user_recommendation import UserRecommendation

class GroupRecommendation:

    @staticmethod
    def aggregate_users_recommendations(users: set[int], n: int = 10, neighbor_size: int = 50) -> dict[int, list[float]]:
        """
        Aggregate recommendations from a set of users.

        Args:
            users (set[int]): Set of user IDs.
            n (int): Number of recommendations per user. Defaults to 10.
            neighbor_size (int): Size of the neighborhood for recommendation. Defaults to 50.

        Returns:
            dict[int, list[float]]: Dictionary where keys are movie IDs and values are lists of predicted ratings.
        """
        aggregate_recommendations: dict[int, list[float]] = {}

        for user in users:
            user_recommendations = UserRecommendation.top_n_recommendations(user, n, neighbor_size)

            for (movie, predicted_rating) in user_recommendations:
                if aggregate_recommendations.get(movie) == None:
                    aggregate_recommendations[movie] = [predicted_rating]
                else:
                    aggregate_recommendations.get(movie).append(predicted_rating)

        return aggregate_recommendations


    @staticmethod
    def average_aggregation(users: set[int], n: int = 10) -> list[tuple[int, float]]:
        """
        Aggregate recommendations by averaging predicted ratings.

        Args:
            users (set[int]): Set of user IDs.
            n (int): Number of recommendations to return. Defaults to 10.

        Returns:
            list[tuple[int, float]]: List of tuples containing movie ID and average predicted rating.
        """
        users_rec = GroupRecommendation.aggregate_users_recommendations(users)

        avg_rec: list[tuple[int, float]] = []

        for movie, predicted_ratings in users_rec.items():
            average = sum(predicted_ratings) / len(predicted_ratings)
            avg_rec.append((movie, average))
        
        avg_rec.sort(key=lambda x: x[1], reverse=True)

        return avg_rec[:n]


    @staticmethod
    def least_misery_aggregation(users: set[int], n: int = 10) -> list[tuple[int, float]]:
        """
        Aggregate recommendations by selecting the minimum predicted rating.

        Args:
            users (set[int]): Set of user IDs.
            n (int): Number of recommendations to return. Defaults to 10.

        Returns:
            list[tuple[int, float]]: List of tuples containing movie ID and minimum predicted rating.
        """
        users_rec = GroupRecommendation.aggregate_users_recommendations(users)

        least_misery_rec: list[tuple[int, float]] = []

        for movie, predicted_ratings in users_rec.items():
            least_misery_rec.append((movie, min(predicted_ratings)))

        least_misery_rec.sort(key=lambda x: x[1], reverse=True)

        return least_misery_rec[:n]

