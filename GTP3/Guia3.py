#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 21:22:24 2019

@author: joaco
"""

import os
import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

wd = '/home/joaco/Desktop/joaco/facultad/BCI/Guías/GTP3'
os.chdir(wd)


def get_indexes(stim_labels, stim_code):
	indexes = []
	for i in range(len(stim_labels)):
		if stim_labels[i] == stim_code:
			indexes.append(i)
	return indexes

def mean_channel(data, indexes):
	channels = {}
	for channel in range(len(data[0])):
		canal = []
		for sample in range(len(data[0][channel])):
			samples = []
			for i in indexes:
				samples.append(data[i][channel][sample])
			canal.append(np.mean(samples))
		channels[channel] = canal
	return channels

def butter_bandpass_filter(data, frecuencias, sampling_freq, order, axis):
	frecuencias = [frecuencias[i]/(sampling_freq/2) for i in range(len(frecuencias))]
	b, a = signal.butter(order, frecuencias, btype='band')
	y = signal.filtfilt(b, a, data, axis = axis, padlen = None)
	return y

#%% 3.1
from sklearn.model_selection import train_test_split
	
data_256 = spio.loadmat('Datos/Datos_Sujeto1_256.mat')
datos_256 = np.swapaxes(data_256['Datos'], 1, 2)
data_32 = spio.loadmat('Datos/Datos_Sujeto1_32.mat')
datos_32 = np.swapaxes(data_32['Datos'], 1, 2)
labels = data_256['Etiquetas']

datos_256 = np.reshape(datos_256, [3780, 2560])
datos_32 = np.reshape(datos_32, [3780, 320])

###---- 256 ----###

###---- SEPARACION DE DATOS EN TRAIN Y VAL ----###
datos_256_train, datos_256_val, labels_train, labels_val = train_test_split(datos_256, labels, test_size=0.3, random_state=42)
labels_train = np.ravel(labels_train)
labels_val = np.ravel(labels_val)

indexes_0 = get_indexes(labels_train, 0)
indexes_1 = get_indexes(labels_train, 1)

###---- LDA ----###
lda_256 = LinearDiscriminantAnalysis()
datos_256_transformed = lda_256.fit_transform(datos_256_train, labels_train)
predictions_256 = lda_256.predict(datos_256_val)

acert = 0
for i in range(len(labels_val)):
	if labels_val[i] == predictions_256[i]:
		acert += 1
		
acert /= len(labels_val)
accuaracy_256 = acert*100
print('\nAccuaracy 256 = {:.2f}%\n'.format(accuaracy_256))

plt.figure()
plt.hist(datos_256_transformed, bins = 30)
plt.grid()
plt.title('256')
plt.xlabel('datos proyectados')

plt.figure()
plt.grid()
plt.title('Samples (sf = 256)')
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.plot(datos_256_train[indexes_0, 5], datos_256_train[indexes_0, 10], 'b.')
plt.plot(datos_256_train[indexes_1, 5], datos_256_train[indexes_1, 10], 'r.')

###---- 32 ----###

###---- SEPARACION DE DATOS EN TRAIN Y VAL ----#
datos_32_train, datos_32_val, labels_train, labels_val = train_test_split(datos_32, labels, test_size=0.3, random_state=42)
labels_train = np.ravel(labels_train)
labels_val = np.ravel(labels_val)

###---- LDA ----###
lda_32 = LinearDiscriminantAnalysis()
datos_32_transformed = lda_32.fit_transform(datos_32_train, labels_train)
predictions_32 = lda_32.predict(datos_32_val)

acert = 0
for i in range(len(labels_val)):
	if labels_val[i] == predictions_32[i]:
		acert += 1
		
acert /= len(labels_val)
accuaracy_32 = acert*100
print('\nAccuaracy 32 = {:.2f}%\n'.format(accuaracy_32))

plt.figure()
plt.hist(datos_32_transformed, bins = 30)
plt.grid()
plt.title('32')
plt.xlabel('datos proyectados')

plt.figure()
plt.grid()
plt.title('Samples (sf = 32)')
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.plot(datos_256_train[indexes_0, 5], datos_256_train[indexes_0, 10], 'b.')
plt.plot(datos_256_train[indexes_1, 5], datos_256_train[indexes_1, 10], 'r.')

	


