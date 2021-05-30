import json
from datetime import datetime
import os
import tkinter as tk
from PIL import Image

def log_new_trial(objects, trial_name):
    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    data = {"objects": objects,
            "date": date_time,
            "num_objects": len(objects),
            "experiment_ID": trial_name}
    for object in objects:
        data[object] = { "roi_bounds": 0,
                         "gaze_x": 0,
                         "gaze_y": 0,
                         "t_start": 0,
                         "t_select": 0,
                         "t_close": 0,
                         "t_completion": 0,
                         "NOSC": 0,
                         "NGSC": 0,
                         "NAC": 0,
                         "TTA": 0,
                         "TTG": 0,
                         "OST": 0,
                         "TTT": 0,
                         "success": False}
        data[object]["fail_reason"] = {"prosthetic_err": False,
                                       "obj_select_err": False,
                                       "grasp_config_err": False,
                                       "aperture_err": False}
    return data

def load_from_json(json_file):
    with open(json_file, "r") as read_file:
        return json.load(read_file)

def write_to_json(data):
    log_dir = "helpers/Experiment Logs"
    trial_name = data["experiment_ID"] + "_" + data["date"]
    trial_log_dir = os.path.join(log_dir, trial_name)
    if not os.path.exists(trial_log_dir):
        os.mkdir(trial_log_dir)
    # log_dir = "Experiment Logs"
    file_name = trial_name + '.json'
    with open(os.path.join(trial_log_dir, file_name), "w") as write_file:
        json.dump(data, write_file)

def user_pop_up(trial_data, object_name):
    # Opens the trial json file, and confirms trial outcome
    root = tk.Tk()
    v = tk.IntVar()
    v.set(0)  # initializing the choice, i.e. Python
    result_outcomes = [("Pass", True),
                       ("Fail", False)]

    def log_results(trial_data=trial_data, object_name=object_name):
        trial_passed = bool(v.get())
        prosthetic_err = bool(var1.get())
        obj_select_err = bool(var2.get())
        grasp_config_err = bool(var3.get())
        aperture_err = bool(var4.get())

        print('Trial Passed?: ', trial_passed, '\n',
              'Prosthetic Error: ', prosthetic_err, '\n',
              'Incorrect Object Selection: ', obj_select_err, '\n',
              'Incorrect Grasp Configuration: ', grasp_config_err, '\n',
              'Incorrect Aperture: ', aperture_err, '\n')

        trial_data[object_name]["success"] = trial_passed
        trial_data[object_name]["fail_reason"]["prosthetic_err"] = prosthetic_err
        trial_data[object_name]["fail_reason"]["obj_select_err"] = obj_select_err
        trial_data[object_name]["fail_reason"]["grasp_config_err"] = grasp_config_err
        trial_data[object_name]["fail_reason"]["aperture_err"] = aperture_err
        write_to_json(trial_data)
        root.destroy()

    label_font = 'Helvetica 10 bold'
    tk.Label(root, text="Trial ID: " + trial_data["experiment_ID"], justify=tk.LEFT, font=label_font).pack(anchor=tk.N)
    tk.Label(root, text="Trial Date: "  + trial_data["date"], justify=tk.LEFT, font=label_font).pack(anchor=tk.N)
    tk.Label(root, text="Object Type: " + object_name, justify=tk.LEFT, font=label_font).pack(anchor=tk.N)
    tk.Label(root, text="", justify=tk.LEFT, font=label_font).pack(anchor=tk.N)

    tk.Label(root, text="Trial Outcome:", justify=tk.LEFT, font=label_font).pack(anchor=tk.W)
    for result, val in result_outcomes:
        tk.Radiobutton(root, text=result, padx=20, variable=v, value=val).pack(anchor=tk.W)

    tk.Label(root, text="""Fail Reason (If Applicable):""", justify=tk.LEFT, font=label_font).pack(anchor=tk.W)
    var1 = tk.IntVar()
    var2 = tk.IntVar()
    var3 = tk.IntVar()
    var4 = tk.IntVar()
    var5 = tk.IntVar()
    c1 = tk.Checkbutton(root, text='Prosthetic Error', variable=var1, onvalue=1, offvalue=0, padx=20)
    c2 = tk.Checkbutton(root, text='Incorrect Object Selection', variable=var2, onvalue=1, offvalue=0, padx=20)
    c3 = tk.Checkbutton(root, text='Incorrect Grasp Configuration', variable=var3, onvalue=1, offvalue=0, padx=20)
    c4 = tk.Checkbutton(root, text='Incorrect Aperture', variable=var4, onvalue=1, offvalue=0, padx=20)

    log_results = tk.Button(root, text='Log Results', command=log_results, padx=20)
    c1.pack(anchor=tk.W)
    c2.pack(anchor=tk.W)
    c3.pack(anchor=tk.W)
    c4.pack(anchor=tk.W)
    log_results.pack(anchor=tk.S)
    root.geometry("500x400")
    root.mainloop()

def log_image(data, object_name, image):
    log_dir = "helpers/Experiment Logs"
    trial_name = data["experiment_ID"] + "_" + data["date"]
    trial_log_dir = os.path.join(log_dir, trial_name)
    if not os.path.exists(trial_log_dir):
        os.mkdir(trial_log_dir)
    file_name = data["experiment_ID"] + "_" + data["date"] + '_' + object_name +'.png'
    im = Image.fromarray(image)
    im.save(os.path.join(trial_log_dir, file_name))


