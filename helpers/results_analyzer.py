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
        self.flattened_data = []
        self.data_loaded = False
        self.data_frame = None
        self.exp_0_TAR = 0
        self.exp_1_TAR = 0
        self.exp_0_errors = None
        self.exp_1_errors = None
        self.exp_0_timing = None
        self.exp_1_timing = None
        self.exp_1_minus_0_timing = None

    def parse_log_dir(self):
        # Only parse the log directory if the data has not been loaded into self.flattened_data (Checking for
        # self.data_loaded flag)
        image_center_crop_size = 384
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
                        data[object]['gaze_y'] += image_center_crop_size
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
                        # data_row = np.array(data_row).reshape(1, -1)
                        self.flattened_data.append(data_row)
            self.flattened_data = self.flattened_data[1:]
            self.data_loaded = True

    def create_pandas_df(self):
        self.data_frame = pd.DataFrame(data=self.flattened_data, columns=self.data_columns)

    def load_from_json(self, file_path):
        with open(file_path, "r") as read_file:
            return json.load(read_file)

    def compute_TARs(self):
        self.num_passes_exp_0 = len(self.data_frame[self.data_frame['exp_0_verdict']])
        self.exp_0_TAR = self.num_passes_exp_0 / self.total_num_objects
        self.num_passes_exp_1 = len(self.data_frame[self.data_frame['exp_1_verdict']])
        self.exp_1_TAR = self.num_passes_exp_1 / self.total_num_objects

    def generate_verdicts(self):
        self.total_num_objects = len(self.data_frame)
        df = results.data_frame
        # exp_0_verdicts = ~df['prosthetic_err'] & ~df['obj_select_err'] & ~df['grasp_config_err'] & ~df['aperture_err'] \
        #                  & df['success'] & (df['NOSC'] == 0) & (df['NAC'] == 0)
        exp_0_verdicts = df['success'] & (df['NOSC'] == 0) & (df['NAC'] == 0)
        self.data_frame = df.assign(exp_0_verdict=exp_0_verdicts)
        exp_1_verdicts = self.data_frame['success']
        self.data_frame = self.data_frame.assign(exp_1_verdict=exp_1_verdicts)

    def analyze_error_sources(self):
        simplified_df = self.data_frame[['NOSC', 'NAC', 'prosthetic_err', 'obj_select_err', 'grasp_config_err',
                                         'aperture_err', 'exp_0_verdict', 'exp_1_verdict']]
        self.exp_0_errors = simplified_df.groupby('exp_0_verdict').sum().drop(['exp_1_verdict'], axis=1)
        self.exp_1_errors = simplified_df.groupby('exp_1_verdict').sum().drop(['exp_0_verdict'], axis=1)

    def analyze_time_info(self):

        # Need to flag the cases where the hand was never commanded to close (t_close = 0)
        df = self.data_frame.drop(['NOSC', 'NAC', 'object_name', 'success', 'prosthetic_err',
                                   'obj_select_err', 'grasp_config_err', 'aperture_err'], axis=1)
        df = df[(df['TTA'] > 0) & (df['TTG'] > 0) & (df['OST'] > 0) & (df['TTT'] > 0)]
        exp_0_df = df[df['exp_0_verdict']].drop(['t_start', 't_select', 't_close', 't_completion'], axis=1)
        # Only consider the data points that were considered a fail in Experiment 0
        exp_1_df = df[df['exp_1_verdict']].drop(['t_start', 't_select', 't_close', 't_completion'], axis=1)
        exp_1_minus_0_df = df[df['exp_1_verdict'] & ~df['exp_0_verdict']].drop(['t_start', 't_select', 't_close',
                                                                                't_completion'], axis=1)
        self.exp_0_timing = exp_0_df
        self.exp_1_timing = exp_1_df
        self.exp_1_minus_0_timing = exp_1_minus_0_df

log_dir = "Experiment Logs"
experiment_name = "Experiment_1"
results = ResultAnalyzer(log_dir, experiment_name)
results.parse_log_dir()
results.create_pandas_df()
df = results.data_frame

# Task Accomplishment Rates
# Need to omit entries that do not have a start time
df_filtered = df[df['t_start'] > 0]

# Selecting relevant rows:
df_relevant_fields = df_filtered[['object_name', 't_start', 't_select', 't_close',
       't_completion', 'NOSC', 'NAC', 'TTA', 'TTG', 'OST', 'TTT',
       'success', 'prosthetic_err', 'obj_select_err', 'grasp_config_err',
       'aperture_err']]

results.data_frame = df_relevant_fields
results.generate_verdicts()
results.compute_TARs()
results.analyze_error_sources()
results.analyze_time_info()

print("\n-----------------------RESULTS SUMMARY----------------------\n")
print("Total Number of Objects: " + str(results.total_num_objects))
print("Total Number of Baskers: " + str(results.num_baskets))

print("\n---------------Experiment 0 - No User Control---------------")
print("Total Number of Passes for Experiment 0 (No user input): " + str(results.num_passes_exp_0))
print("Task Accomplishment Rate for Experiment 0 (No user input): " +  str(round(results.exp_0_TAR * 100, 2)) + "%")
# print("Timing for Experiment 0: " + str(TTT_mean_exp_0) + "+-" + str(TTT_std_exp_0) + " s")
print("Failure Distribution: \n" + str(results.exp_0_errors))
timing_info_exp_0 = results.exp_0_timing.drop(['exp_0_verdict', 'exp_1_verdict'], axis=1)
print("Mean Timing for Experiment 0: \n" + str(timing_info_exp_0.mean()/1000.0))
print("Standard Deviation of Timing for Experiment 0: \n" + str(timing_info_exp_0.std()/1000.0))
print("------------------------------------------------------------\n")

print("\n---------------Experiment 1 - Proposed System---------------")
print("Total Number of Passes for Experiment 1: " + str(results.num_passes_exp_1))
print("Task Accomplishment Rate for Experiment 1: " + str(round(results.exp_1_TAR * 100, 2)) + "%")
print("Failure Distribution: \n" + str(results.exp_1_errors))
timing_info_exp_1 = results.exp_1_timing.drop(['exp_0_verdict', 'exp_1_verdict'], axis=1)
print("Mean Timing for Experiment 1: \n" + str(timing_info_exp_1.mean()/1000.0))
print("Standard Deviation of Timing for Experiment 1: \n" + str(timing_info_exp_1.std()/1000.0))

print("\n------Timing on objects that failed in experiment 0---------")

timing_info_exp_1_minus_0 = results.exp_1_minus_0_timing.drop(['exp_0_verdict', 'exp_1_verdict'], axis=1)
print("Number of objects that failed in experiment 0, but passed in experiment 1: " + str(len(timing_info_exp_1_minus_0)))
print("Mean Timing for Experiment 1 minus 0: \n" + str(timing_info_exp_1_minus_0.mean()/1000.0))
print("Standard Deviation of Timing for Experiment 1 minus 0: \n" + str(timing_info_exp_1_minus_0.std()/1000.0))

print("------------------------------------------------------------\n")
import code; code.interact(local=dict(globals(), **locals()))

#
# # Grouping based on success and sources of error to determine Task Accomplishment rates
# grouping = df_filtered.groupby(['success', 'prosthetic_err', 'obj_select_err', 'grasp_config_err', 'aperture_err']).mean()
# error_distribution = df_filtered[['prosthetic_err', 'obj_select_err', 'grasp_config_err', 'aperture_err', 'success']].groupby(['success']).mean().reset_index()
# df_filtered[['prosthetic_err', 'obj_select_err', 'grasp_config_err', 'aperture_err', 'success']].sum().reset_index()
# df_filtered[['prosthetic_err', 'obj_select_err', 'grasp_config_err', 'aperture_err', 'success']].groupby(['success']).sum().reset_index()
#
# # Need data frames for experiments 1 and 0, with a redefined success column
# # Experiment 0: A successful entry is one where there are no failures specified, and the grasping task actually happened
# # IE: ['prosthetic_err', 'obj_select_err', 'grasp_config_err', 'aperture_err'] are all false, and success is True
#
#
# df_experiment_0 = df_filtered
# df_experiment_0.loc[~df_filtered['prosthetic_err']
#             & ~df_filtered['obj_select_err']
#             & ~df_filtered['grasp_config_err']
#             & ~df_filtered['aperture_err']]['success'] = False
#
#
# # Plotting number of times objects were grasped
# df['object_name'] = df['object_name'].str.replace("- UR", "").str.replace("- RDM", "")
# df['object_name'].value_counts().plot.bar(x='Object Category', y='Number of Occurences', rot=90)
#
# # Plotting the task accomplishment times for each object
# df['TTT'] /= 1000
# df[['object_name', 'TTT']].groupby(df['object_name']).mean()
#

# # Grouping
# pd.value_counts(df['success']).plot.bar()
# df['success'].mean()
# df_simplified = df.drop(df.columns[:10].tolist(), axis=1)
# df_simplified.groupby(['success', 'prosthetic_err', 'obj_select_err', 'grasp_config_err', 'aperture_err']).mean()
#
# import code; code.interact(local=dict(globals(), **locals()))