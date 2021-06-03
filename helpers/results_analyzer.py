import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

class ResultAnalyzer:
    def __init__(self, data_dir, experiment_name):
        self.data_dir = data_dir
        self.experiment_name = experiment_name
        self.total_num_objects = 0
        self.num_baskets = 0
        self.num_passes_exp_1 = 0
        self.num_passes_exp_0 = 0
        self.data_columns = ["date", "num_basket_objects", "object_iterator", "object_name", "Y1", "X1", "Y2", "X2",
                             "gaze_x", "gaze_y", "t_start", "t_select", "t_close", "t_completion", "NOSC", "NGSC",
                             "NAC", "TTA", "TTG", "OST", "TTT", "success", "prosthetic_err", "obj_select_err",
                             "grasp_config_err", "aperture_err"]
        self.flattened_data = np.zeros([1, len(self.data_columns)])
        self.data_loaded = False
        self.data_frame = None

    def parse_log_dir(self):
        # Only parse the log directory if the data has not been loaded into self.flattened_data (Checking for
        # self.data_loaded flag)
        if not self.data_loaded:
            trial_folders = os.listdir(self.data_dir)
            for folder in trial_folders:
                if "Miracast" in folder or ".DS_Store" in folder:
                    continue
                if self.experiment_name in folder:
                    folder_path = os.path.join(log_dir, folder)
                    json_file_name = folder + '.json'
                    json_path = os.path.join(folder_path, json_file_name)
                    data = self.load_from_json(json_path)
                    self.num_baskets += 1
                    basket_objects = data['objects']
                    num_basket_objects = data['num_objects']
                    experiment_date = data['date']
                    for object_iterator, object in enumerate(basket_objects):
                        failures = data[object]['fail_reason']
                        fail_reasons = np.array([failures['prosthetic_err'],
                                                 failures['obj_select_err'],
                                                 failures['grasp_config_err'],
                                                 failures['aperture_err']])
                        data_row = [experiment_date, num_basket_objects, object_iterator, object]
                        for key in data[object].keys():
                            if key == "fail_reason":
                                data[object][key] = fail_reasons.tolist()
                            if type(data[object][key]) is list:
                                for val in data[object][key]:
                                    data_row.append(val)
                                continue
                            data_row.append(data[object][key])
                        data_row = np.array(data_row).reshape(1, -1)
                        self.flattened_data = np.concatenate([self.flattened_data, np.array(data_row)], axis=0)
                self.flattened_data = self.flattened_data[1:]
                self.data_loaded = True

    def create_pandas_df(self):
        self.data_frame = pd.DataFrame(data=self.flattened_data, columns=self.data_columns)

    def load_from_json(self, file_path):
        with open(file_path, "r") as read_file:
            return json.load(read_file)

log_dir = "Experiment Logs"
experiment_name = "Experiment_1"
results = ResultAnalyzer(log_dir, experiment_name)
results.parse_log_dir()
results.create_pandas_df()
data_frame = results.data_frame
print(data_frame.head(10))
import code; code.interact(local=dict(globals(), **locals()))