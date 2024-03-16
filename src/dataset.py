import pandas as pd
import os

class Dataset:

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # List of CSV files to load as dataframes
        self._dataframe_names = ['movies', 'ratings']

        self._prepare()


    def _prepare(self):
        self._dataframe: dict[int, pd.DataFrame] = {}

        self._load_dataframes()
        self._init_movies()
        self._init_ratings()
    

    def _load_dataframes(self) -> None:
        """
        Loads CSV files into pandas dataframes.
        """
        # Define project and dataset paths
        project_folder = os.path.dirname(__file__) + '/../'
        dataset_path = os.path.join(project_folder, 'dataset', 'movielens-edu')

        self.movies_df = pd.read_csv(dataset_path + "/movies.csv", converters={"genres": lambda x: x.strip("[]").replace("'","").split("|")})
        self.ratings_df = pd.read_csv(dataset_path + "/ratings.csv")


    def _init_movies(self):
        unique_genre = self.movies_df['genres'].explode().unique()

        # Make a dict assigning an index to a genre
        self.genres = list(unique_genre)

        self.df_grouped_by_movieId = self.movies_df.groupby('movieId')

        ratings_grouped_by_movie_df = self.ratings_df.groupby('movieId')

        self.movies_df['avg_rating'] =  ratings_grouped_by_movie_df.rating.mean(numeric_only=True)


    
    def _init_ratings(self):
        self.ratings_df['datetime'] = pd.to_datetime(self.ratings_df['timestamp'], unit='s').dt.strftime('%d-%m-%Y')

        # Dictionary to store user ratings
        self._user_to_movie_ratings: dict[int, dict[int, float]] = {}

        # Group ratings dataframe by user
        ratings_grouped_by_user_df = self.ratings_df.groupby('userId')

        # Calculate mean rating for each user
        self._user_ratings_mean = ratings_grouped_by_user_df.rating.mean()

        # Initialize user ratings dictionary
        for user_id, rating_df in ratings_grouped_by_user_df:
            self._user_to_movie_ratings[user_id] = dict(zip(rating_df['movieId'], rating_df['rating']))

        self.rating_count_df = pd.DataFrame(self.ratings_df.groupby(['rating']).size(), columns=['count'])


    def has_user_rated_movie(self, user_id: int, movie_id: int) -> bool:
        """
        Checks if a user has rated a specific movie.

        Args:
            user_id (int): ID of the user.
            movie_id (int): ID of the movie.

        Returns:
            bool: True if the user has rated the movie, False otherwise.
        """
        return self._user_to_movie_ratings[user_id].get(movie_id) != None


    def get_user_mean_rating(self, user_id: int) -> float:
        """
        Retrieves the mean rating of a user.

        Args:
            user_id (int): ID of the user.

        Returns:
            float: Mean rating of the user.
        """
        return self._user_ratings_mean[user_id]
    

    def get_rating(self, user_id: int, movie_id: int) -> float:
        """
        Retrieves the rating given by a user for a movie.

        Args:
            user_id (int): ID of the user.
            movie_id (int): ID of the movie.

        Returns:
            float: Rating given by the user for the movie.
        """
        return self._user_to_movie_ratings[user_id][movie_id]
    

    def get_rating_mean_centered(self, user_id: int, movie_id: int) -> float:
        """
        Retrieves the mean-centered rating of a user for a movie.

        Args:
            user_id (int): ID of the user.
            movie_id (int): ID of the movie.

        Returns:
            float: Mean-centered rating of the user for the movie.
        """
        return self.get_rating(user_id, movie_id) - self.get_user_mean_rating(user_id)
    

    def get_movies_rated_by_user(self, user_id: int) -> set[int]:
        """
        Retrieves the movies rated by a user.

        Args:
            user_id (int): ID of the user.

        Returns:
            set: Set of movie IDs rated by the user.
        """
        return set(self._user_to_movie_ratings[user_id].keys())
    

    def get_movies_unrated_by_user(self, user_id: int) -> set[int]:
        """
        Retrieves the movies not rated by a user.

        Args:
            user_id (int): ID of the user.

        Returns:
            set: Set of movie IDs not rated by the user.
        """
        # Calculate the difference between all movies and rated movies
        return self.get_movies() - self.get_movies_rated_by_user(user_id)
    

    def get_common_movies(self, user1_id: int, user2_id: int) -> set[int]:
        """
        Retrieves the movies rated by both users.

        Args:
            user1_id (int): ID of the first user.
            user2_id (int): ID of the second user.

        Returns:
            set: Set of movie IDs rated by both users.
        """
        return self.get_movies_rated_by_user(user1_id) & self.get_movies_rated_by_user(user2_id)
    

    def get_users(self) -> set[int]:
        """
        Retrieves the set of user IDs.

        Returns:
            set: Set of user IDs.
        """
        return set(self.ratings_df['userId'])
    

    def get_movies(self) -> set[int]:
        """
        Retrieves the set of movie IDs.

        Returns:
            set: Set of movie IDs.
        """
        return set(self.movies_df['movieId'])
    

    def get_movie_name(self, movie_id: int) -> str:
        """
        Retrieves the name of a movie by its ID.

        Args:
            movie_id (int): ID of the movie.

        Returns:
            str: Name of the movie.
        """
        return self.df_grouped_by_movieId.get_group(movie_id).title[0]
    

    def get_movie_genres(self, movie_id: int):
        return self.df_grouped_by_movieId.get_group(movie_id).genres.values[0]