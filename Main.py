# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 23:34:26 2018

@author: Zhiyong
"""
import os
import scipy.io 
import pywt as pw
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from sklearn.preprocessing import LabelEncoder, StandardScaler
from Models import * 

def init_proces(PATH = './',PATH_RES = './'):
    etiquetas = pd.read_csv(PATH + '/etiquetas.csv').values
    archiv_nom = pd.read_csv(PATH + '/archivo_wav.csv').values
    filelist = os.listdir(PATH) #Lista todos los archivos del path (es decir las señales)
    Signals = [] # Vector para las señales
    Labels = []  # Vector para las etiquetas
    Archivos= [] #nombre de los archivos para guardar los punto npy
    for file in filelist: # Archivo en la lista de archivos
        f = PATH + '/' + file # Path completo de cada archivo
        if file.endswith(".mat"): # Comprueba que termina en .mat para saber que es una señal
            data = scipy.io.loadmat(f)['val'] # Carga la señal
            data=data[0]
            #print(data)
            Signals.append(data)
            length = data.shape
    Labels = np.array(etiquetas)
    Signals =np.array(Signals)
    Archivos = np.array(archiv_nom)
    le = LabelEncoder()
    st = StandardScaler()
    # Reestructura adecuadamente los arreglos
    Labels = le.fit_transform(Labels)
    Labels = Labels[:, np.newaxis]
    Signals = st.fit_transform(Signals)
    #/ = 0 > CONTRACCION
    # L = 1 > BLOQUEO DE RAMA IZQUIERDA
    #N = 2 > NORMAL
    # R = 3 > BLOQUEO DE RAMA DERECHA
    # ~ = 4 > LATIDO ESTIMULADO 
    #NOTA: no se necesita redimensionar las seniales ya que vienen de 650000 q signfica que tienen media hora
    #USO DE WAVLET 
    y_train = []
    x_train = []
    X_train = []
    for i,sig in enumerate(Signals):
        coeffs = pw.swt(sig, wavelet = "dB6", level=4)
        subsen = coeffs[3][0]
        length= subsen.shape[0]/4
        Subsen = []
        Subsen.append(subsen[:int(length)])
        Subsen.append(subsen[int(length):int(length)*2])
        Subsen.append(subsen[(int(length)*2):int(length)*3])
        Subsen.append(subsen[(int(length)*3):int(length)*4])
        nom = Archivos[i]
        nombremat = PATH_RES+'/x_'+str(nom[0])+'.npy'
        np.save(nombremat, Subsen)
        label = (np.repeat(Labels[i], 4))
        y_train.append(label)
        x_train = np.array(Subsen)
        x_trin = np.transpose(x_train)
        X_train.append(x_trin)
    y_train=np.array(y_train)
    X_train=np.array(X_train)
    return y_train
def ingre_npy (y,path='./'):
    filelist = os.listdir(path) #Lista todos los archivos del path (es decir las señales)
    #print(filelist)
    Signals = [] # Vector para las señales
    for file in filelist: # Archivo en la lista de archivos
        f = path + '/' + file # Path completo de cada archivo
        if file.endswith(".mat"): # Comprueba que termina en .mat para saber que es una señal
            data = scipy.io.loadmat(f)['matriz_senalC'] # Carga la señal
            data = data[:6000,]
            #for i in [0,1,2,3]:
            #    seg = data[:,i]
            #    seg=np.transpose(seg)
            #    coeffs = pw.swt(seg, wavelet = "dB6", level=4)
            #    subsen = coeffs[3][0]
            #    data[:,i] = np.transpose(subsen)
            #print(data)
            Signals.append(data)
            length = data.shape
    x_train =np.array(Signals)
    return x_train

def init_main(path='./', path_res_dat = './', model_epoch =30):
    
    path_res ='/content/ProyectoI/data'
    y_train = init_proces(path,path_res)
    x_train = ingre_npy(y_train,path_res_dat)
    print(x_train.shape[2])
    print(x_train.shape[1])
    patience = 20
    print("#######################################################")
    print("model_2_Bi_LSTM")
    model_2_Bi_LSTM, history_2_Bi_LSTM = train_2_Bi_LSTM(x_train, y_train, epochs = model_epoch)
    print(history_2_Bi_LSTM)
    model_2_Bi_LSTM.save('Model_2_Bi_LSTM_' + str(len(history_2_Bi_LSTM.losses))+ '.h5')
    #y_train = model_2_Bi_LSTM.predict(x_train)
    #MeasurePerformance(Y_test_scale, Y_pred_test, X_max, model_name = 'default', epochs = len(history_2_Bi_LSTM.losses), model_time_lag = 10)
