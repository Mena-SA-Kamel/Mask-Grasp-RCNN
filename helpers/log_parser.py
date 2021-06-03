import json
import os
import numpy as np
import pandas as pd

def load_from_json(json_file):
    with open(json_file, "r") as read_file:
        return json.load(read_file)

log_dir = "Experiment Logs"
trial_folders = os.listdir(log_dir)
experiment_name = "Experiment_1"
total_num_objects = 0
num_baskets = 0
num_passes_exp_1 = 0
num_passes_exp_0 = 0
failures_log = np.zeros(4)
ttt_log_exp_1 = []
ttt_log_exp_0 = []
data_row = []
data_columns=["date", "num_basket_objects", "object_iterator", "object_name", "Y1", "X1", "Y2", "X2", "gaze_x",
              "gaze_y", "t_start", "t_select", "t_close", "t_completion", "NOSC", "NGSC", "NAC", "TTA", "TTG", "OST",
              "TTT", "success", "prosthetic_err", "obj_select_err", "grasp_config_err", "aperture_err"]
flattened_data = np.zeros([1, len(data_columns)])

for folder in trial_folders:
    if "Miracast" in folder or ".DS_Store" in folder:
        continue
    if experiment_name in folder:
        folder_path = os.path.join(log_dir, folder)
        json_file_name = folder + '.json'
        json_path = os.path.join(folder_path, json_file_name)
        data = load_from_json(json_path)
        num_baskets += 1
        basket_objects = data['objects']
        num_basket_objects = data['num_objects']
        experiment_date = data['date']
        for object_iterator, object in enumerate(basket_objects):
            t_start = data[object]['t_start']
            t_select = data[object]['t_select']
            t_close = data[object]['t_close']
            t_completion = data[object]['t_completion']
            TTT = data[object]['TTT']
            TTA = data[object]['TTA']
            TTG = data[object]['TTG']
            OST = data[object]['OST']
            NOSC = data[object]['NOSC']
            NGSC = data[object]['NGSC']
            NAC = data[object]['NAC']
            failures = data[object]['fail_reason']
            fail_numbers = np.array([failures['prosthetic_err'],
                                    failures['obj_select_err'],
                                    failures['grasp_config_err'],
                                    failures['aperture_err']])
            data_row = [experiment_date, num_basket_objects, object_iterator, object]
            for key in data[object].keys():
                if key == "fail_reason":
                    data[object][key] = fail_numbers.tolist()
                if type(data[object][key]) is list:
                    for val in data[object][key]:
                        data_row.append(val)
                    continue
                data_row.append(data[object][key])
            data_row = np.array(data_row).reshape(1, -1)

            flattened_data = np.concatenate([flattened_data, np.array(data_row)], axis=0)

            # if 0 in [t_start, t_select, t_completion]:
            #     print([t_start, t_select, t_close, t_completion])
            #     continue
            total_num_objects += 1
            if (data[object]['success']):
                num_passes_exp_1 += 1
                ttt_log_exp_1.append(TTT)
            if not(fail_numbers.any()):
                num_passes_exp_0 += 1
                ttt_log_exp_0.append(TTT)
            failures_log += fail_numbers.astype('int8')
            # print (t_start, t_select, t_close, t_completion)
import code; code.interact(local=dict(globals(), **locals()))

TAR_exp_1 = round(float(num_passes_exp_1)/total_num_objects * 100, 2)
TAR_exp_0 = round(float(num_passes_exp_0)/total_num_objects * 100, 2)
TTT_mean_exp_1 = round(np.mean(np.array(ttt_log_exp_1))/1000, 2)
TTT_std_exp_1 = round(np.std(np.array(ttt_log_exp_1))/1000, 2)

TTT_mean_exp_0 = round(np.mean(np.array(ttt_log_exp_0))/1000, 2)
TTT_std_exp_0 = round(np.std(np.array(ttt_log_exp_0))/1000, 2)

print("\n---------------RESULTS SUMMARY---------------\n")
print("Total Number of Objects: " + str(total_num_objects))
print("Total Number of Baskers: " + str(num_baskets))
print("\n---------------Experiment 0 - No User Control---------------")
print("Total Number of Passes for Experiment 0 (No user input): " + str(num_passes_exp_0))
print("Task Accomplishment Rate for Experiment 0 (No user input): " +  str(TAR_exp_0) + "%")
print("Total Task Time for Experiment 0: " + str(TTT_mean_exp_0) + "+-" + str(TTT_std_exp_0) + " s")
print("Failure Distribution (prosthetic_err, obj_select_err, grasp_config_err, aperture_err): " + str(failures_log))

print("\n---------------Experiment 1 - Proposed System---------------")
print("Total Number of Passes for Experiment 1: " + str(num_passes_exp_1))
print("Task Accomplishment Rate for Experiment 1: " + str(TAR_exp_1) + "%")
print("Total Task Time for Experiment 1: " + str(TTT_mean_exp_1) + "+-" + str(TTT_std_exp_1) + " s")
print("\n---------------------------------------------\n")
import code; code.interact(local=dict(globals(), **locals()))
