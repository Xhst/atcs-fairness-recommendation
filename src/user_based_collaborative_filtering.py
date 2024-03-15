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