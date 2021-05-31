import json
import os
import logger

log_dir = "Experiment Logs"
trial_folders = os.listdir(log_dir)

for folder in trial_folders:
    folder_path = os.path.join(log_dir, folder)
    json_file_name = folder + '.json'
    json_path = os.path.join(folder_path, json_file_name)
    print(logger.load_from_json(json_path), '\n')


