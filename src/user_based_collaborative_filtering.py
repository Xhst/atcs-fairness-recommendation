import dataset
import math

class UserBasedCollaborativeFiltering:
    def __init__(self):
        self.dataset = dataset.Dataset()
        

    def compute_similarities(self) -> None:
        self.similarities = {}

        for user1 in self.dataset.get_users():
            self.similarities[user1] = {}
            for user2 in self.dataset.get_users():
                self.similarities[user1][user2] = self.sim_pcc(user1, user2)
    

    def sim_pcc(self, user1: int, user2: int) -> float:
        """
        Computes the Pearson Correlation Coefficient between two users based on their ratings.

        Args:
            user1 (int): ID of the first user.
            user2 (int): ID of the second user.

        Returns:
            float: Pearson Correlation Coefficient between the two users.
        """
        '''
        Pearson Correlation Coefficient

        Measures linear correlation between two sets of data.
        It's the ratio between the covariance of two variables and the product of their standard deviations.
        It's a normalized measurement of the covariance, such that the result always has a value between −1 and 1.
        '''     

        # Find common movies rated by both users
        common_movies = self.dataset.get_common_movies(user1, user2)

        # Calculate Pearson correlation coefficient
        numerator = 0
        denominator1 = 0
        denominator2 = 0

        for movie in common_movies:
            user1_mean_centered_movie_rating = self.dataset.get_rating_mean_centered(user1, movie)
            user2_mean_centered_movie_rating = self.dataset.get_rating_mean_centered(user2, movie)

            numerator += user1_mean_centered_movie_rating * user2_mean_centered_movie_rating
            
            denominator1 += user1_mean_centered_movie_rating ** 2
            denominator2 += user2_mean_centered_movie_rating ** 2

        denominator = math.sqrt(denominator1) * math.sqrt(denominator2)

        if denominator == 0: return 0

        return numerator / denominator
    
    
    def prediction_from_neighbors(self, user: int, movie: int, neighbors: list) -> float:
        numerator = 0
        denominator = 0

        for other_user, similarity in neighbors:
            if not self.dataset.has_user_rated_movie(other_user, movie): continue

            numerator += similarity * self.dataset.get_rating_mean_centered(other_user, movie)
            denominator += similarity

        if denominator == 0: return 0
        
        return self.dataset.get_user_mean_rating(user) + (numerator / denominator)
    

    def top_n_similar_users(self, user: int, similarity_function = None, n: int = 10) -> list:
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
        
        ls = []
        
        for other_user in self.dataset.get_users():
            if user == other_user: continue

            ls.append((other_user, self.similarities[user][other_user]))

        # Sort the users by similarity in descending order
        ls.sort(key=lambda x: x[1], reverse=True)    

        return ls[:n]
