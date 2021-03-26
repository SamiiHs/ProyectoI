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


def train_Bi_LSTM(X, Y, epochs = 30, validation_split = 0.2, patience=20):
    speed_input = Input(shape = (X.shape[1], X.shape[2]), name = 'signal')
    
    main_output = Bidirectional(LSTM(4,input_shape = (X.shape[1], X.shape[2]), return_sequences=False), merge_mode='ave')(speed_input)
    
    final_model = Model([speed_input], [main_output])
    
    final_model.summary()
    
    final_model.compile(loss='mse', optimizer='adam')
    
    history = LossHistory()
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=patience, verbose=0, mode='auto')
    final_model.fit([X], Y,epochs = epochs, validation_split = 0.2, callbacks=[history, earlyStopping])
    
    return final_model, history

def train_2_Bi_LSTM_mask(X, Y, epochs = 30, validation_split = 0.2, patience=20):
    
    model = Sequential()
    model.add(Masking(mask_value=0.,input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(4, return_sequences=True, input_shape = (X.shape[1], X.shape[2])))
    model.add(LSTM(4, return_sequences=False, input_shape = (X.shape[1], X.shape[2])))

    model.add(Dense(4))
    model.compile(loss='mse', optimizer='rmsprop')

    history = LossHistory()
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=patience, verbose=0, mode='auto')
    model.fit(X, Y, validation_split = 0.2, epochs = epochs, callbacks=[history, earlyStopping])

    return model, history

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
    print(h)
    return final_model, history