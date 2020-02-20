from datetime import datetime


class Experiment(object):
    n_subjects = 4
    n_tasks = 2
    n_cameras = 3

    def __init__(self, _id: str, _type: str, timestamp: datetime):
        self._id = _id
        self.type = _type
        self.timestamp = timestamp
