import dataset
import math 

# UserBasedCollaborativeFiltering
class UBCF:    
    
    dataset = dataset.Dataset() 

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
        common_movies = UBCF.dataset.get_common_movies(user1, user2)

        # Calculate Pearson correlation coefficient
        numerator = 0
        denominator1 = 0
        denominator2 = 0

        for movie in common_movies:
            user1_mean_centered_movie_rating = UBCF.dataset.get_rating_mean_centered(user1, movie)
            user2_mean_centered_movie_rating = UBCF.dataset.get_rating_mean_centered(user2, movie)

            numerator += user1_mean_centered_movie_rating * user2_mean_centered_movie_rating
            
            denominator1 += user1_mean_centered_movie_rating ** 2
            denominator2 += user2_mean_centered_movie_rating ** 2

        denominator = math.sqrt(denominator1) * math.sqrt(denominator2)

        if denominator == 0: return 0

        return numerator / denominator
    
    
    def prediction_from_neighbors(user: int, movie: int, neighbors: list) -> float:
        numerator = 0
        denominator = 0

        for other_user, similarity in neighbors:
            if not UBCF.dataset.has_user_rated_movie(other_user, movie): continue

            numerator += similarity * UBCF.dataset.get_rating_mean_centered(other_user, movie)
            denominator += similarity

        if denominator == 0: return 0
        
        return UBCF.dataset.get_user_mean_rating(user) + (numerator / denominator)
    

    def top_n_similar_users(user: int, similarity_function = None, n: int = 10) -> list:
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
            similarity_function = UBCF.sim_pcc
        
        ls = []
        
        for other_user in UBCF.dataset.get_users():
            if user == other_user: continue

            ls.append((other_user, similarity_function(user, other_user)))

        # Sort the users by similarity in descending order
        ls.sort(key=lambda x: x[1], reverse=True)    

        return ls[:n]


    def top_n_recommendations(user: int, n: int = 10, neighbor_size: int = 50):
        """
        Generates top N movie recommendations for a given user, excluding movies already rated by the user.

        Args:
            user (int): ID of the user.
            n (int, optional): Number of recommendations to generate.

        Returns:
            List: List of tuples containing movie IDs and their predicted ratings.
        """
        unrated_movies = UBCF.dataset.get_movies_unrated_by_user(user)

        # Predicted ratings for movies
        predicted_ratings = []

        neighbors = UBCF.top_n_similar_users(user, n=neighbor_size)

        for movie_id in unrated_movies:
            # Predict the rating for the movie
            predicted_rating = UBCF.prediction_from_neighbors(user, movie_id, neighbors)
            # Store the predicted rating
            predicted_ratings.append((movie_id, predicted_rating))

        # Predicted ratings in descending order
        predicted_ratings.sort(key=lambda x: x[1], reverse=True)

        # Return the top N movies along with their predicted ratings
        return predicted_ratings[:n]