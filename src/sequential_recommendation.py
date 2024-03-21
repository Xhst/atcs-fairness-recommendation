from collections import defaultdict
from group_recommendation import GroupRecommendation
import random


class SequentialRecommendation:
    def __init__(self, group_recommendation: GroupRecommendation):
        self.group_recommendation = group_recommendation


    # maybe get every group recommendation with the "fair" method (see last slides)
    # and then we  calculate the disagreemnt for each group recommendation and we change 
    # the couples to consider when running the "fair" method for the next group recommendation
    # i.e if we considered user x and y, and we calculate user z and w are dissagreing the most
    #     we then consider the couple z and w for the next group recommendation
    def get_sequential_recommendations_for_group(self, userGroup: list[int], seq_len: int = 5) -> dict[int, list[tuple[int, float]]]:
        """
        Get sequential recommendations of certain length for a group of users.

        Args:
            userGroup (list[int]): List of user IDs.
            ser_len (int): Amount of group recommendations in a sequential for a group. Defaults to 5.

        Returns:
            list[tuple[int, float]]: list of tuples containing sequential ID and the recommendetions for the group.
        """
        
        #giusto?
        sequential_recommendations: dict[int, list[tuple[int, float]]] = defaultdict(list[tuple[int, float]])
        
        for i in range(seq_len):
            group_recommendation = self.group_recommendation.weighted_average_aggregation(userGroup)
            
            fairer_group_recommendation = group_recommendation
            
            if i != 0:
                # get the most disagreeing users
                # and change the couples to consider for the next group recommendation
                # (or do something, consider disagreement nonetheless)
                fairer_group_recommendation = self.transform_group_recomm_to_fair(group_recommendation)
            
            sequential_recommendations[i] = fairer_group_recommendation
        
        return sequential_recommendations
    
    
    def transform_group_recomm_to_fair(self, group_recommendations: list[tuple[int, float]]) -> list[tuple[int, float]]:
        """
        Tries to transform group recommendations to be more fair for certain users who were disagreeing the most with
        the recommendations given in the past.

        Args:
            group_recommendations (list[tuple[int, float]]): List of tuples containing movie ID and average predicted rating.
            userGroup (list[int]): List of user IDs.

        Returns:
            list[tuple[int, float]]: list of tuples containing movie ID and average predicted rating.
        """
        # get the most disagreeing users
        # and change the couples to consider for the next group recommendation
        # (or do something, consider disagreement nonetheless)
        
        fairer_group_recommendation = [(movie, rating * (random.random())) 
                                       for movie, rating in group_recommendations]
        
        fairer_group_recommendation.sort(key=lambda x: x[1], reverse=True)
        
        return fairer_group_recommendation
    