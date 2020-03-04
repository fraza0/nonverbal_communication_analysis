import pandas as pd
import json 

from nonverbal_communication_analysis.environment import DATASET_SYNC


class Experiment(object):
    _n_subjects = 4
    _n_tasks = 2
    _n_cameras = 3

    def __init__(self, _id: str):
        self._id = _id
        self.type = self.match_id_type(_id)
        self.people = dict()  # {camera_id:x ,people:[]}

    def match_id_type(self, _id: str):
        df = pd.read_csv(DATASET_SYNC + 'groups_info.csv')
        return df[df['Group ID'] == _id]['Conflict Type'].tolist()[0]

    def to_json(self):

        people_data = dict()
        for camera, people in self.people.items():
            people_data[camera] = list()
            for frame in people:
                people_data[camera].append(frame.to_json())

        obj = {
            "experiment": {
                "id": self._id,
                "type": self.type,
                "people": people_data,
            }
        }

        return json.dumps(obj)

    def from_json(self):
        return None

    def __str__(self):
        return "Experiment { id: %s, type: %s, people: %s }" % (self._id, self.type,  str(self.people))
