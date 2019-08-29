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
from scipy import signal
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

indexes_0 = get_indexes(labels_val, 0)
indexes_1 = get_indexes(labels_val, 1)

###---- LDA ----###
lda_256 = LinearDiscriminantAnalysis()
lda_256.fit(datos_256_train, labels_train)

fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax1.set_title('Raw data (sf = 256)')
ax1.grid()
ax1.set_xlabel('feature 1')
ax1.set_ylabel('feature 2')
ax1.plot(datos_256_val[indexes_0, 5], datos_256_val[indexes_0, 10], '.')
ax1.plot(datos_256_val[indexes_1, 5], datos_256_val[indexes_1, 10], '.')

ax2= fig.add_subplot(2,1,2)
ax2.grid()
ax2.set_title('Transformed data (sf = 256)')
ax2.plot(lda_256.transform(datos_256_val)[indexes_0], '.')
ax2.plot(lda_256.transform(datos_256_val)[indexes_1], '.')
fig.tight_layout()

###---- 32 ----###

###---- SEPARACION DE DATOS EN TRAIN Y VAL ----#
datos_32_train, datos_32_val, labels_train, labels_val = train_test_split(datos_32, labels, test_size=0.3, random_state=42)
labels_train = np.ravel(labels_train)
labels_val = np.ravel(labels_val)

indexes_0 = get_indexes(labels_val, 0)
indexes_1 = get_indexes(labels_val, 1)

###---- LDA ----###
lda_32 = LinearDiscriminantAnalysis()
lda_32.fit(datos_32_train, labels_train)

fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax1.set_title('Raw data (sf = 32)')
ax1.grid()
ax1.set_xlabel('feature 1')
ax1.set_ylabel('feature 2')
ax1.plot(datos_32_train[indexes_0, 5], datos_32_train[indexes_0, 10], '.')
ax1.plot(datos_32_train[indexes_1, 5], datos_32_train[indexes_1, 10], '.')

ax2= fig.add_subplot(2,1,2)
ax2.grid()
ax2.set_title('Transformed data (sf = 32)')
ax2.plot(lda_32.transform(datos_32_val)[indexes_0], '.')
ax2.plot(lda_32.transform(datos_32_val)[indexes_1], '.')
fig.tight_layout()

#%% 3.1 2
import random
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate

data_32 = spio.loadmat('Datos/Datos_Sujeto1_32.mat')
datos_32 = np.swapaxes(data_32['Datos'], 1, 2)
labels = data_256['Etiquetas']
datos_32 = np.reshape(datos_32, [3780, 320])

datos_32_train, datos_32_val, labels_train, labels_val = train_test_split(datos_32, labels, test_size=0.1, random_state=42)
labels_train = np.ravel(labels_train)
labels_val = np.ravel(labels_val)

indices_0 = get_indexes(labels_train, 0)
indices_1 = get_indexes(labels_train,1)
indices_0_630 = random.sample(indices_0, 630)
indices_bal = indices_0_630 + indices_1
datos_32_bal = datos_32[indices_bal]
labels_bal = labels_train[indices_bal]

shrink_list = [0, 0.01, 0.05, 0.1, 0.3, 0.5, 0.8, 1]
means_test = []
means_train = []
variances_test = []
variances_train = []
means_val = []

for s in shrink_list:
	lda = LinearDiscriminantAnalysis(solver = 'eigen', shrinkage = s)
	scores = cross_validate(lda, datos_32_bal, labels_bal, cv=5, return_train_score=True)
	means_test.append(np.mean(scores['test_score']))
	means_train.append(np.mean(scores['train_score']))
	variances_test.append(np.var(scores['test_score']))
	variances_train.append(np.var(scores['train_score']))
	lda.fit(datos_32_bal, labels_bal)
	means_val.append(lda.score(datos_32_val, labels_val))
	
plt.figure()
plt.grid()
plt.xlabel('Shrinkage value')
plt.ylabel('Scores')
plt.plot(shrink_list, means_train, '.', label='Train')
plt.errorbar(shrink_list, means_train, yerr = variances_train, fmt = 'none', ecolor = 'k', elinewidth = 1)
plt.plot(shrink_list, means_test, '.', label='Test')
plt.errorbar(shrink_list, means_test, yerr = variances_test, fmt = 'none', ecolor = 'k', elinewidth = 1)
plt.plot(shrink_list, means_val, '.', label='Val')
plt.legend()

#%% 1.3 3
import random
	
data_32 = spio.loadmat('Datos/Datos_Sujeto1_32.mat')
datos_32 = np.swapaxes(data_32['Datos'], 1, 2)
labels = data_256['Etiquetas']
datos_32 = np.reshape(datos_32, [3780, 320])

indices_0 = get_indexes(labels, 0)
indices_1 = get_indexes(labels,1)
indices_0_630 = random.sample(indices_0, 630)
indices_bal = indices_0_630 + indices_1
datos_32_bal = datos_32[indices_bal]
labels_bal = labels[indices_bal]

tamaños = [180, 360, 720]
desempeños_trad = {}
desempeños_reg = {}
for tamaño in tamaños:
	resultados_trad = []
	resultados_reg = []
	for i in range(10):
		indices = random.sample(list(np.arange(len(datos_32_bal))), tamaño)
		datos_train = datos_32_bal[indices]
		labels_train = labels_bal[indices]
		datos_val = np.delete(datos_32_bal, indices, axis = 0)
		labels_val = np.delete(labels_bal, indices, axis = 0)
		
		lda_trad = LinearDiscriminantAnalysis(solver='eigen')
		lda_reg = LinearDiscriminantAnalysis(solver = 'eigen', shrinkage = 'auto')
		lda_trad.fit(datos_train, labels_train)
		lda_reg.fit(datos_train, labels_train)
		resultados_trad.append(lda_trad.score(datos_32_bal, labels_bal))
		resultados_reg.append(lda_reg.score(datos_32_bal, labels_bal))
		
	desempeños_trad[tamaño]=np.mean(resultados_trad)
	desempeños_reg[tamaño]=np.mean(resultados_reg)
		
		
	



