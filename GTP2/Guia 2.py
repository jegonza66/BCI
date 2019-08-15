#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 12:18:06 2019

@author: joaco
"""
import os
import numpy as np
import scipy.io as spio
from scipy import signal
import matplotlib.pyplot as plt
from mne.decoding import UnsupervisedSpatialFilter
from sklearn.decomposition import PCA, FastICA

data = spio.loadmat('Datos/DataEEG_EOG_bis.mat')

datos = data['Datos']
datos = np.swapaxes(datos,0,2)

pca = UnsupervisedSpatialFilter(PCA(), average = False)
data_pca = pca.fit_transform(datos)


fig1 = plt.figure()
fig2 = plt.figure()
for i in range(len(datos[0])):
	ax1 = fig1.add_subplot(5,5,(i+1))
	ax1.grid()
	ax1.plot(datos[0][i], label = 'Raw data')
	ax2 = fig2.add_subplot(5,5,i+1)
	ax2.grid()
	ax2.plot(data_pca[0][i], label = 'Principal components')