import dataset
import math 

# UserBasedCollaborativeFiltering
class UserRecommendation:    
    
    dataset = dataset.Dataset() 

    @staticmethod
    def sim_pcc(user1: int, user2: int) -> float:
        """
        Computes the Pearson Correlation Coefficient between two users based on their ratings.

        Args:
            user1 (int): ID of the first user.
            user2 (int): ID of the second user.

        Returns:
            float: Pearson Correlation Coefficient between the two users.
        """
        # Find common movies rated by both users
        common_movies = UserRecommendation.dataset.get_common_movies(user1, user2)

        # Calculate Pearson correlation coefficient
        numerator = 0
        denominator1 = 0
        denominator2 = 0

        for movie in common_movies:
            user1_mean_centered_movie_rating = UserRecommendation.dataset.get_rating_mean_centered(user1, movie)
            user2_mean_centered_movie_rating = UserRecommendation.dataset.get_rating_mean_centered(user2, movie)

            numerator += user1_mean_centered_movie_rating * user2_mean_centered_movie_rating
            
            denominator1 += user1_mean_centered_movie_rating ** 2
            denominator2 += user2_mean_centered_movie_rating ** 2

        denominator = math.sqrt(denominator1) * math.sqrt(denominator2)

        if denominator == 0: return 0

        return numerator / denominator
    
    
    @staticmethod
    def sim_wpcc_common_movies(user1: int, user2: int):
        """
        Computes the Pearson Correlation Coefficient between two users based on their ratings,
        using the number of common movies divided by the total number of movies rated by the second user as a weight.
        
        This helps because penalizes the similarity score when the two users have very few rated movies in common.
        Furthermore, it helps to avoid the problem of having a similarity score of 1 when the two users have only one movie in common.
        Additionally, it penalizes the similarity score when the second user has rated a very large number of movies compared to
        the movies rated in common with the first user.
        
        Args:
            user1 (int): ID of the first user.
            user2 (int): ID of the second user.
            
        Returns:
            float: Pearson Correlation Coefficient between the two users, multiplied by the weight.
        """
        number_of_common_movies = len(UserRecommendation.dataset.get_common_movies(user1, user2))
        number_of_movies_rated_by_user2 = len(UserRecommendation.dataset.get_movies_rated_by_user(user2))
        
        weight = number_of_common_movies / number_of_movies_rated_by_user2
        
        return UserRecommendation.sim_pcc(user1, user2) * weight
    
    
    @staticmethod
    def prediction_from_neighbors(user: int, movie: int, neighbors: list[tuple[int, float]]) -> float:
        """
        Predicts the rating for a movie by a user based on the ratings of similar users.

        Args:
            user (int): ID of the user.
            movie (int): ID of the movie.
            neighbors (list[tuple[int, float]]): List of tuples containing IDs of similar users and their similarity scores.

        Returns:
            float: Predicted rating for the movie by the user.
        """
        numerator = 0
        denominator = 0

        for other_user, similarity in neighbors:
            if not UserRecommendation.dataset.has_user_rated_movie(other_user, movie): continue

            numerator += similarity * UserRecommendation.dataset.get_rating_mean_centered(other_user, movie)
            denominator += abs(similarity)

        if denominator == 0: return 0
        
        return UserRecommendation.dataset.get_user_mean_rating(user) + (numerator / denominator)
    

    @staticmethod
    def top_n_similar_users(user: int, similarity_function = None, n: int = 10) -> list[tuple[int, float]]:
        """
        Finds the top N similar users to a given user based on a similarity function.

        Args:
            user (int): ID of the user.
            similarity_function (function, optional): Function to compute similarity between users. 
                Defaults to sim_pcc.
            n (int, optional): Number of similar users to find. Defaults to 10.

        Returns:
            List: List of tuples containing similar user IDs and their corresponding similarity scores.
        """
        if similarity_function is None:
            similarity_function = UserRecommendation.sim_pcc
        
        ls: list[tuple[int, float]] = []
        
        for other_user in UserRecommendation.dataset.get_users():
            if user == other_user: continue

            ls.append((other_user, similarity_function(user, other_user)))

        # Sort the users by similarity in descending order
        ls.sort(key=lambda x: x[1], reverse=True)    

        return ls[:n]


    @staticmethod
    def top_n_recommendations(user: int, similarity_function = None, n: int = 10, neighbor_size: int = 50) -> list[tuple[int, float]]:
        """
        Generates top N movie recommendations for a given user, excluding movies already rated by the user.

        Args:
            user (int): ID of the user.
            similarity_function (function, optional): A function to compute similarity between users. If None, defaults to Pearson correlation.
            n (int, optional): Number of recommendations to generate. Defaults to 10.
            neighbor_size (int, optional): Number of neighbors to consider for recommendation. Defaults to 50.

        Returns:
            List[tuple[int, float]]: List of tuples containing movie IDs and their predicted ratings.
        """
        unrated_movies = UserRecommendation.dataset.get_movies_unrated_by_user(user)

        # Predicted ratings for movies
        predicted_ratings: list[tuple[int, float]] = []

        neighbors = UserRecommendation.top_n_similar_users(user, similarity_function=similarity_function, n=neighbor_size)

        for movie_id in unrated_movies:
            # Predict the rating for the movie
            predicted_rating = UserRecommendation.prediction_from_neighbors(user, movie_id, neighbors)
            # Store the predicted rating
            predicted_ratings.append((movie_id, predicted_rating))

        # Predicted ratings in descending order
        predicted_ratings.sort(key=lambda x: x[1], reverse=True)

        # Return the top N movies along with their predicted ratings
        return predicted_ratings[:n]