import json
from datetime import datetime

def log_new_trial(objects, experiment_ID=0):
    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    data = {"objects": objects,
            "date": date_time,
            "experiment_ID": experiment_ID}
    for object in objects:
        data[object] = { "t_start": 0,
                         "t_select": 0,
                         "t_close": 0,
                         "t_completion": 0,
                         "NOSC": 0,
                         "NGSC": 0,
                         "NAC": 0,
                         "TTA": 0,
                         "TTG": 0,
                         "OSS": 0,
                         "success": False,
                         "fail_reason": ""}
    return data

def load_from_json(json_file):
    with open(json_file, "r") as read_file:
        return json.load(read_file)

def write_to_json(data):
    file_name = data["experiment_ID"] + "_" + data["date"] + '.json'
    with open(file_name, "w") as write_file:
        json.dump(data, write_file)





