# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 23:31:31 2018

@author: Zhiyong
"""




from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Embedding, Input
from keras.layers import LSTM, SimpleRNN, GRU, merge, Masking , Flatten
from keras.models import Model
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.layers.wrappers import Bidirectional

import numpy as np
from numpy.random import RandomState
from random import shuffle
import datetime

np.random.seed(1024)

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

def train_2_Bi_LSTM(X, Y, epochs = 30, validation_split = 0.2, patience=20):
    signal = Input(shape = (6000, 4),dtype=np.float32, name = 'signal')
    init = signal
    x2= Bidirectional(LSTM(64,return_sequences = True))(init)
    x3 = Dropout(0.1)(x2)
    x4 = Bidirectional(LSTM(32,return_sequences = True))(x3)
    x5 = Dropout(0.1)(x4)
    x6 = Flatten()(x5)
    x7 = Dense(128, activation='relu')(x6)
    x8 = Dropout(0.2)(x7)
    main_output= Dense(4, activation='softmax')(x8)
    final_model = Model([signal], [main_output])
    
    final_model.summary()
    
    final_model.compile(loss='mse', optimizer='adam')
    
    history = LossHistory()
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=patience, verbose=0, mode='auto')
    h=final_model.fit([X], Y, epochs = epochs,validation_split = 0.2, callbacks=[history, earlyStopping])
    return final_model, history
