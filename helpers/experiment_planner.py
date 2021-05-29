import os
import numpy as np
import random
import logger

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
    # Sampling without replacement
    basket_contents = random.sample(objects, num_objects)
    return basket_contents

# experiment_ID = 0
# num_baskets = 1
# trial_ID = 0 # 0, 1, 2
# object_ID = 0 # num objects in basket
# objects = load_data("all_objects.txt")
# data = []
# for i in range(num_baskets):
#     basket_contents = generate_experiment_basket(objects)
#     trial_name = "Experiment_" + str(experiment_ID) + "_Basket_" + str(i)
#     data.append(logger.log_new_trial(basket_contents, trial_name))
#     for object_type in data[i]['objects']:
#         logger.user_pop_up(data[i], object_type)
#     logger.write_to_json(data[i])


