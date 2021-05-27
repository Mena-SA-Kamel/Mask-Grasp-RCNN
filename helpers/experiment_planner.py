import os
import numpy as np
import logger
import random

def get_num_objects_per_image(mean=6, std=2):
    num_objects = int(np.random.normal(mean, std, 1))
    if num_objects < 0:
        num_objects = 0
    return num_objects

def load_data(file_name):
    all_objects = []
    if os.path.exists(file_name):
        f = open(file_name, "r")
        all_objects = f.read().splitlines()
        all_objects = np.array(all_objects)
        f.close()
    return list(all_objects)

def generate_experiment_basket(objects):
    num_objects = get_num_objects_per_image()
    basket_contents = random.sample(objects, num_objects)
    return basket_contents


objects = load_data("all_objects.txt")
basket_contents = generate_experiment_basket(objects)

import code; code.interact(local=dict(globals(), **locals()))

data = logger.log_new_trial(objects, experiment_ID="Experiment_0")

logger.write_to_json(data)