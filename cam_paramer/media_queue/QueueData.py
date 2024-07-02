from cam_paramer.media_queue.GroupData import GroupData
from cam_paramer.utils.input_utils import search_input
from typing import List

class QueueData:
    """
    Class to manage media groups based on input data.
    """

    def __init__(self, input: str):
        """
        Initialize QueueData with input string.

        Args:
        - input (str): Input data to search and organize into media groups.
        """
        self.media_groups: List[GroupData] = []  
        groups = search_input(input)
        
        for group_dict in groups:      
            group = GroupData(group_dict)
            self.media_groups.append(group)