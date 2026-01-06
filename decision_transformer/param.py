from simulator._cpu_.param import QUEUE_CAP

SEQ_LEN = 32
BATCH_SIZE = 64
EPISODES_NUM = 1250
TRAINING_FRACTION = 0.8
TESTING_FRACTION = 0.2
TRAIN_STEPS = 6000
ACCUM_STEPS = 10

NUM_ACTIONS = QUEUE_CAP   # number of discrete actions
STATE_DIM = QUEUE_CAP * 6 + 14      # dimension of state representation
D_MODEL = 128     # transformer model dimension
N_HEADS = 4       # number of attention heads
N_LAYERS = 3      # number of transformer layers
MIN_TIMESTEPS = 1800   # minimum timesteps per episode
MAX_TIMESTEPS = 2400  # maximum timesteps per episode

LEARNING_RATE = 0.0003
GAMMA = 0.99
WEIGHT_DECAY = 0.0001
GRAD_CLIP = 1.0
