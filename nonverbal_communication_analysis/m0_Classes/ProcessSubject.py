class ProcessSubject(object):
    def __init__(self, _id):
        self.id = _id
        self.previous_pose = dict()
        self.current_pose = dict()
        self.expansiveness = dict()

    def __str__(self):
        return "ProcessSubject: {id: %s}" % self.id

    def _update_pose(self, camera, current_pose, verbose: bool = False):
        if verbose:
            print("Subject", self.id,
                  "\nPrev.: ", self.previous_pose.keys())
        self.previous_pose = self.current_pose
        self.current_pose[camera] = current_pose
        if verbose:
            print("Current:", self.current_pose.keys())

    def metric_expansiveness(self, verbose: bool = False):
        if verbose:
            print("Expansiveness on ", self)
        # TODO: Change if necessary to use kypoint confidence to get
        # minimum keypoint value
        horizontal = {'min': None, 'max': None}
        vertical = {'min': None, 'max': None}

        print("=====", self, "=====")

        for camera, keypoints in self.current_pose.items():
            for _, keypoint in keypoints.items():
                if not horizontal['min']:
                    horizontal['min'] = keypoint[0]
                elif keypoint[0] < horizontal['min']:
                    horizontal['min'] = keypoint[0]

                if not horizontal['max']:
                    horizontal['max'] = keypoint[0]
                elif keypoint[0] > horizontal['max']:
                    horizontal['max'] = keypoint[0]

                if not vertical['min']:
                    vertical['min'] = keypoint[1]
                elif keypoint[1] < vertical['min']:
                    vertical['min'] = keypoint[1]

                if not vertical['max']:
                    vertical['max'] = keypoint[1]
                elif keypoint[1] > vertical['max']:
                    vertical['max'] = keypoint[1]

            if camera not in self.expansiveness:
                self.expansiveness[camera] = dict()

            self.expansiveness[camera]['x'] = [horizontal['min'],
                                               horizontal['max']]
            self.expansiveness[camera]['y'] = [vertical['min'],
                                               vertical['max']]

        print(self.expansiveness)

    def from_json(self):
        return None