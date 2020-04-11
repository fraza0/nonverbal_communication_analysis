import pandas as pd
import json
from pathlib import Path

from nonverbal_communication_analysis.environment import DATASET_DIR, DATASET_SYNC, GROUPS_INFO_FILE


def get_group_from_file_path(group_directory_path: {str, Path}):
    group_directory_path = str(group_directory_path)
    sample_indicator = 'SAMPLE'
    df = pd.read_csv(GROUPS_INFO_FILE)
    is_sample = sample_indicator in group_directory_path

    id_match = None
    for group_id in list(df['Group ID']):
        match = group_id in group_directory_path
        if is_sample:
            match = (match and sample_indicator in group_id)

        if match:
            id_match = group_id
            break

    if match and id_match is not None:
        return id_match

    return False


class Experiment(object):
    """Experiment Class

    Each experiment is recorded by 3 cameras and comprises 
    a group of 4 elements performing 2 different tasks

    """

    _n_cameras = 3
    _n_subjects = 4
    _n_tasks = 2

    def __init__(self, _id: str):
        self._id = _id
        self.type = self.match_id_type(_id)
        self.people = dict()

    def match_id_type(self, _id: str):
        """Get Group Conflict Type from GroupInfo data

        Args:
            _id (str): Group identification

        Returns:
            str: Group Conflict Type 
        """
        df = pd.read_csv(DATASET_DIR / 'groups_info.csv')
        return df[df['Group ID'] == _id]['Conflict Type'].tolist()[0]

    def to_json(self):
        """Transform Experiment object to JSON format

        Returns:
            str: JSON formatted Experiment object
        """

        people_data = dict()
        for camera, people in self.people.items():
            people_data[camera] = list()
            for frame in people:
                people_data[camera].append(frame.to_json())

        obj = {
            "experiment": {
                "id": self._id,
                "type": self.type,
                # "people": people_data,
            }
        }

        return obj

    def from_json(self):
        """Create Experiment object from JSON string

        Returns:
            Experiment: Experiment object
        """
        return None

    def __str__(self):
        return "Experiment { id: %s, type: %s, people: %s }" % (self._id, self.type,  str(self.people))
