# TODO: parse arguments
...


# set seed
seed = 29  # TODO: set seed to allow for reproducibility of results

import os
os.environ['PYTHONHASHSEED'] = str(seed)

import random
random.seed(seed)

import numpy as np
np.random.seed(seed)

import tensorflow as tf
tf.random.set_seed(seed)


# initialize environment
from environment import Environment

data_dir = './data'  # TODO: specify relative path to data directory (e.g., './data', not './data/variant_0')
variant = 0  # TODO: specify problem variant (0 for base variant, 1 for first extension, 2 for second extension)
env = Environment(variant, data_dir)


# TODO: execute training
...
