import dataset
import math 
from typing import Callable
import numpy as np

# UserBasedCollaborativeFiltering
class UserRecommendation:    
    
    def __init__(self, dataset: dataset.Dataset) -> None:
        self.dataset = dataset


    def sim_pcc(self, user1: int, user2: int) -> float:
        """
        Computes the Pearson Correlation Coefficient between two users based on their ratings.

        Args:
            user1 (int): ID of the first user.
            user2 (int): ID of the second user.

        Returns:
            float: Pearson Correlation Coefficient between the two users.
        """
        # Find common movies rated by both users
        common_movies = self.dataset.get_common_movies(user1, user2)

        # Check if there are no common movies or if one user has rated all movies the same
        if len(common_movies) == 0:
            return 0  # Return 0 correlation when there are no common movies

        # Calculate mean ratings for both users
        mean_rating_user1 = sum(self.dataset.get_rating(user1, movie) for movie in common_movies) / len(common_movies)
        mean_rating_user2 = sum(self.dataset.get_rating(user2, movie) for movie in common_movies) / len(common_movies)

        # Calculate numerator and denominators for Pearson correlation coefficient formula
        numerator = np.sum((self.dataset.get_rating(user1, movie) - mean_rating_user1) *
                        (self.dataset.get_rating(user2, movie) - mean_rating_user2) for movie in common_movies)
        denominator_user1 = np.sum((self.dataset.get_rating(user1, movie) - mean_rating_user1) ** 2 for movie in common_movies)
        denominator_user2 = np.sum((self.dataset.get_rating(user2, movie) - mean_rating_user2) ** 2 for movie in common_movies)

        # Handle division by zero (when one of the users has rated all movies the same)
        if denominator_user1 == 0 or denominator_user2 == 0:
            return 0  # Return 0 correlation when division by zero occurs

        # Compute Pearson correlation coefficient
        correlation_coefficient = numerator / math.sqrt(denominator_user1) * math.sqrt(denominator_user2)

        return correlation_coefficient
    
    
    def sim_wpcc(self, user1: int, user2: int, weight: Callable[[int, int], float]) -> float:
        number_of_common_movies = len(self.dataset.get_common_movies(user1, user2))
        number_of_movies_rated_by_user2 = len(self.dataset.get_movies_rated_by_user(user2))
        
        weight = number_of_common_movies / number_of_movies_rated_by_user2 if number_of_movies_rated_by_user2 != 0 else 0
        
        return self.sim_pcc(user1, user2) * weight
    
    
    
    def sim_jaccard(self, user1: int, user2: int):
        """
        Computes the Jaccard similarity coefficient between two users based on the movies they have rated.

        Args:
            user1 (int): ID of the first user.
            user2 (int): ID of the second user.

        Returns:
            float: Jaccard similarity coefficient between the two users.
        """
        common_movies = self.dataset.get_common_movies(user1, user2)
        movies_rated_by_user1 = self.dataset.get_movies_rated_by_user(user1)
        movies_rated_by_user2 = self.dataset.get_movies_rated_by_user(user2)
        
        return len(common_movies) / len(movies_rated_by_user1 | movies_rated_by_user2)


    
    def sim_wpcc_jaccard(self, user1: int, user2: int):
        """
        Computes Pearson correlation coefficient (PCC) weighted with the Jaccard similarity coefficient between two users.

        Args:
            user1 (int): ID of the first user.
            user2 (int): ID of the second user.

        Returns:
            float: Weighted product of PCC and Jaccard similarity between the two users.
        """
        return self.sim_pcc(user1, user2) * self.sim_jaccard(user1, user2)
    

    
    def prediction_from_neighbors(self, user: int, movie: int, neighbors: list[tuple[int, float]]) -> float:
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
            if not self.dataset.has_user_rated_movie(other_user, movie): continue

            numerator += similarity * self.dataset.get_rating_mean_centered(other_user, movie)
            denominator += abs(similarity)

        if denominator == 0: return self.dataset.get_user_mean_rating(user)
        
        return self.dataset.get_user_mean_rating(user) + (numerator / denominator)
    

    
    def similarity_for_all_users(self, user: int, similarity_function: Callable = None) -> list[tuple[int, float]]:
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
            similarity_function = self.sim_pcc
        
        ls: list[tuple[int, float]] = []
        
        for other_user in self.dataset.get_users():
            if user == other_user: continue

            ls.append((other_user, similarity_function(user, other_user)))

        # Sort the users by similarity in descending order
        ls.sort(key=lambda x: x[1], reverse=True)    

        return ls
    
    
    
    def top_n_similar_users(self, user: int, similarity_function = None, n: int = 10) -> list[tuple[int, float]]:
        """
        Finds the top N similar users to a given user based on the Pearson correlation coefficient.

        Args:
            user (int): ID of the user.

        Returns:
            List: List of tuples containing similar user IDs and their corresponding similarity scores.
        """
        all_similar_users = self.similarity_for_all_users(user, similarity_function)
        return all_similar_users[:n]
    
    
    
    def get_all_recommendations_for_user(self, user: int, similarity_function = None, 
                                         neighbor_size: int = 50, exclude_movies: set[int] = set()) -> list[tuple[int, float]]:
        """
        Get all movie recommendations for a user.

        Args:
            user (int): ID of the user.
            similarity_function (function, optional): A function to compute similarity between users. If None, defaults to Pearson correlation.
            neighbor_size (int, optional): Number of neighbors to consider for recommendation. Defaults to 50.

        Returns:
            List[tuple[int, float]]: List of tuples containing movie IDs and their predicted ratings.
        """
        unrated_movies = self.dataset.get_movies_unrated_by_user(user)

        # Predicted ratings for movies
        predicted_ratings: list[tuple[int, float]] = []

        neighbors = self.top_n_similar_users(user, similarity_function=similarity_function, n=neighbor_size)

        for movie_id in unrated_movies:
            if movie_id in exclude_movies:
                continue
            # Predict the rating for the movie
            predicted_rating = self.prediction_from_neighbors(user, movie_id, neighbors)
            predicted_ratings.append((movie_id, predicted_rating))

        # Predicted ratings in descending order
        predicted_ratings.sort(key=lambda x: x[1], reverse=True)

        return predicted_ratings
    
    
    
    def top_n_recommendations(self, user: int, similarity_function = None, n: int = 10, 
                              neighbor_size: int = 50, exclude_movies: set[int] = set()) -> list[tuple[int, float]]:
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
        movies_predicted_ratings = self.get_all_recommendations_for_user(user, similarity_function, neighbor_size, exclude_movies)
        
        # uncomment the following lines if you want to normalize the predicted ratings between 0 and 5
        #
        #predicted_ratings = [rating for _, rating in movies_predicted_ratings]
        #
        #normalized_rating = lambda rating: (rating - min(predicted_ratings)) / (max(predicted_ratings) - min(predicted_ratings))
        #
        # We transform the predicted ratings to the range 0-5
        # We normalize them and multiply the result by 5
        #transformed_movies_predicted_ratings = [(movie, normalized_rating(rating) * 5) for (movie, rating) in movies_predicted_ratings]
        
        return movies_predicted_ratings[:n]
    
    