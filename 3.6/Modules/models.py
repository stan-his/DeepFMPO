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
from global_parameters import MAX_FRAGMENTS



def maximization(y_true, y_pred):
    return K.mean(-K.log(y_pred) * y_true)




max_swap = 5


n_dense = 128
n_dense2 = 128
n_dense3 = 64
n_lstm = 32
n_actions = MAX_FRAGMENTS * max_swap + 1


def build_models(inp_shape):

    inp = Input(inp_shape)
    hidden_inp = LeakyReLU(0.1)(TimeDistributed(Dense(n_dense, activation="linear"))(inp))
    hidden = LSTM(n_lstm, return_sequences=True)(hidden_inp)
    hidden = Flatten()(hidden)

    hidden2 = LSTM(n_lstm, return_sequences=True, go_backwards=True)(hidden_inp)
    hidden2 = Flatten()(hidden2)

    inp2 = Input((1,))
    hidden = Concatenate()([hidden, hidden2, inp2])

    hidden = LeakyReLU(0.1)(Dense(n_dense2, activation="linear")(hidden))
    out = Dense(n_actions, activation="softmax", activity_regularizer=l2(0.001))(hidden)

    actor = Model([inp,inp2], out)
    actor.compile(loss=maximization, optimizer=Adam(0.0005))




    inp = Input(inp_shape)
    hidden = LeakyReLU(0.1)(TimeDistributed(Dense(n_dense, activation="linear"))(inp))
    hidden = Bidirectional(LSTM(2*n_lstm))(hidden)

    inp2 = Input((1,))
    hidden = Concatenate()([hidden, inp2])
    hidden = LeakyReLU(0.1)(Dense(n_dense2, activation="linear")(hidden))
    out = LeakyReLU(0.1)(Dense(1, activation="linear")(hidden))

    critic = Model([inp,inp2], out)
    critic.compile(loss="MSE", optimizer=Adam(0.0001))


    return actor, critic
