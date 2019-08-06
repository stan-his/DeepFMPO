from keras import backend as K
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.layers import LSTMCell, LSTM, TimeDistributed, Dense, Input, Lambda, Dropout
from keras.optimizers import Adam, SGD
from sklearn.metrics import roc_auc_score
import numpy as np
from keras import Model
import rdkit.Chem as Chem
from keras.layers import LeakyReLU, Bidirectional, Multiply
from keras.regularizers import l2
from keras.layers import Concatenate, Flatten, Softmax
from global_parameters import MAX_FRAGMENTS, MAX_SWAP, N_DENSE, \
                              N_DENSE2, N_LSTM




# Objective to optimize
def maximization(y_true, y_pred):
    return K.mean(-K.log(y_pred) * y_true)



n_actions = MAX_FRAGMENTS * MAX_SWAP + 1


# Create models
def build_models(inp_shape):

    # Build the actor
    inp = Input(inp_shape)
    hidden_inp = LeakyReLU(0.1)(TimeDistributed(Dense(N_DENSE, activation="linear"))(inp))
    hidden = LSTM(N_LSTM, return_sequences=True)(hidden_inp)
    hidden = Flatten()(hidden)

    hidden2 = LSTM(N_LSTM, return_sequences=True, go_backwards=True)(hidden_inp)
    hidden2 = Flatten()(hidden2)

    inp2 = Input((1,))
    hidden = Concatenate()([hidden, hidden2, inp2])

    hidden = LeakyReLU(0.1)(Dense(N_DENSE2, activation="linear")(hidden))
    out = Dense(n_actions, activation="softmax", activity_regularizer=l2(0.001))(hidden)

    actor = Model([inp,inp2], out)
    actor.compile(loss=maximization, optimizer=Adam(0.0005))


    # Build the critic
    inp = Input(inp_shape)
    hidden = LeakyReLU(0.1)(TimeDistributed(Dense(N_DENSE, activation="linear"))(inp))
    hidden = Bidirectional(LSTM(2*N_LSTM))(hidden)

    inp2 = Input((1,))
    hidden = Concatenate()([hidden, inp2])
    hidden = LeakyReLU(0.1)(Dense(N_DENSE2, activation="linear")(hidden))
    out = Dense(1, activation="linear")(hidden)

    critic = Model([inp,inp2], out)
    critic.compile(loss="MSE", optimizer=Adam(0.0001))


    return actor, critic
