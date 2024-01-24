"""
define some constant parameters
author: huangxi
date: 2022-04-21
"""

GUEST = "guest"
HOST = "host"
DEFAULT_IP = "127.0.0.1"
GUEST_DEFAULT_PORT = 1234
HOST_DEFAULT_PORT = 1235

DEFAULT_BUFFER_SIZE = 809600000

END_SIGNAL = "end"
CONTINUE_SIGNAL = "continue"

PLAIN_MODE = "plain"
ENCRYPTED_MODE = "encrypted"

# MODE = PLAIN_MODE
MODE = ENCRYPTED_MODE

NET_DELAY = 0.01

DEBUG = True

GAMMA = 1/1000
K = -200
EPOCHS = 1
BATCH_SIZE = -1
LEARNING_RATE = 0.01
LOSS_TOLERANCE = -100

MINI_TEST = True
# MINI_TEST = False

MINI_GUEST_TRAIN_DATA_PATH = "data/mini_nus_wide_train_guest.csv"
MINI_HOST_TRAIN_DATA_PATH = "data/mini_nus_wide_train_host.csv"

GUEST_TRAIN_DATA_PATH = "data/nus_wide_train_guest.csv"
HOST_TRAIN_DATA_PATH = "data/nus_wide_train_host.csv"
GUEST_VALIDATE_DATA_PATH = "data/nus_wide_validate_guest.csv"
HOST_VALIDATE_DATA_PATH = "data/nus_wide_validate_host.csv"

EXPERIMENT_ID = 2
MIDDLE_FEATURES = 8
