# Assignment 1: User-based Collaborative Filtering Recommendations
import os
import pandas as pd

class MovieLensDataSet:
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
        

if __name__ == '__main__':
    m = MovieLensDataSet()

    m.display_dataset_first_rows()