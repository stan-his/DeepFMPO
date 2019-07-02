# GLOBAL PARAMETERS

# Fragmenting and building the encoding
MOL_SPLIT_START = 70
MAX_ATOMS = 12
MAX_FREE = 3
MAX_FRAGMENTS = 12


# Similarity parameters
ETA = 0.1

# Generation parameters
MAX_SWAP = 5
FEATURES = 4


# Model parameters
N_DENSE = 128
N_DENSE2 = 32
N_LSTM = 32 # Times 2 neurons, since there are both a forward and a backward pass in the bidirectional LSTM

# RL training
GAMMA = 0.95
BATCH_SIZE = 512
EPOCHS = 4000
TIMES = 8
