# Assignment 1: User-based Collaborative Filtering Recommendations
import os
import pandas as pd
import math

class UserBasedCFR:
    def __init__(self):
        project_folder = os.path.dirname(__file__) + '/../'
        dataset_path = os.path.join(project_folder, 'dataset', 'movielens-edu')

        self.dataframe_names = ['links', 'movies', 'ratings', 'tags']
        
        self.dataframe: dict[int, pd.DataFrame] = dict()

        self.load_dataframes(dataset_path)
        self.init_user_ratings()


    def load_dataframes(self, dataset_path: str) -> dict[str, pd.DataFrame]:
        for name in self.dataframe_names:
            path = os.path.join(dataset_path, name + '.csv')
            self.dataframe[name] = pd.read_csv(path)
    

    def init_user_ratings(self) -> None:
        self.user_ratings: dict[int, dict[int, float]] = dict()

        # ratings dataframe grouped by user
        df_grouped_by_user = self.dataframe['ratings'].groupby('userId')

        self.mean_rating = df_grouped_by_user.rating.mean()

        for user_id, rating_df in df_grouped_by_user:
            self.user_ratings[user_id] = dict(zip(rating_df['movieId'], rating_df['rating']))



    def display_dataset_first_rows(self, nrows: int = 5) -> None:
        print('Max '+ str(nrows) +' csv rows displayed per file.\n')

        for name in self.dataframe_names:
            print('Display '+ name +'.csv')
            print('Number of elements: ', len(self.dataframe[name]))
            print('First elements: ', self.dataframe[name].head(nrows))
            print('\n')

    
    def pcc(self, user1_id, user2_id) -> float:
        '''
        Pearson Correlation Coefficient

        Measures linear correlation between two sets of data.
        It's the ratio between the covariance of two variables and the product of their standard deviations.
        It's a normalized measurement of the covariance, such that the result always has a value between −1 and 1.
        '''     
        user1_ratings = self.user_ratings.get(user1_id)
        user2_ratings = self.user_ratings.get(user2_id)

        mean_rating1 = self.mean_rating.get(user1_id)
        mean_rating2 = self.mean_rating.get(user2_id)

        # intersection of user1 ratings with user2 ratings
        common_movies = set(user1_ratings.keys()) & set(user2_ratings.keys())

        numerator = 0
        denominator1 = 0
        denominator2 = 0

        for movie_id in common_movies:
            numerator += (user1_ratings[movie_id] - mean_rating1) * (user2_ratings[movie_id] - mean_rating2)
            denominator1 += (user1_ratings[movie_id] - mean_rating1) ** 2
            denominator2 += (user2_ratings[movie_id] - mean_rating2) ** 2

        denominator = math.sqrt(denominator1) * math.sqrt(denominator2)

        return numerator / denominator
        

if __name__ == '__main__':
    m = UserBasedCFR()

    m.display_dataset_first_rows()

    print(m.pcc(89,45))