#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 10:18:35 2019

@author: joaco
"""
import os
import numpy as np
import scipy.io as spio
from scipy import signal
import matplotlib.pyplot as plt

wd = '/home/joaco/Desktop/joaco/facultad/BCI/Guías/GTP1'
os.chdir(wd)

def find_indexes_MI(nombre, stim_codes, path = 'Datos/MI_S01'):
	
	data = spio.loadmat('{}/{}.mat'.format(path, nombre))
	sample_time = data['sampleTime']
	sampling_freq = data['samplingFreq'][0][0]
	stims = data['stims']
	samples = data['samples']
	
	stim_indexes = []
	stim_labels = []
	for i in range(len(stims[:,0])):
		if (any (stims[i,1] == stim_codes[j] for j in range(len(stim_codes)))):
			stim_indexes.append(int(stims[i,0]*sampling_freq))
			stim_labels.append(stims[i,1])
			
	return stim_indexes, stim_labels, samples, sampling_freq

def extract_trial_MI(nombres, stim_codes, orden_filtro, frecuencias, downsample):
	trials = []
	stim_labels_final = []
	
	for nombre in nombres:
		stim_indexes, stim_labels, samples, sampling_freq = find_indexes_MI(nombre, stim_codes)
		frecuencias = [frecuencias[i]/(sampling_freq*2) for i in range(len(frecuencias))]
		b, a = signal.butter(orden_filtro, frecuencias ,btype = 'bandpass')
		stim_labels_final.extend(stim_labels)
	
		for i in stim_indexes:
			trials.append(signal.filtfilt(b, a, samples[i:i+sampling_freq*2], axis = 0, padlen=None))
	
	if downsample > 1:
		for i in range(len(trials)):
			trials[i] = trials[i][::int(downsample)]
	
	return trials, stim_labels_final
	
	
#nombres = ['S01_FILT_S1R1', 'S01_FILT_S1R2', 'S01_FILT_S1R3', 'S01_FILT_S1R4']
#stim_codes = [770, 772]
#orden_filtro = 2
#frecuencias = [0.1, 40]
#downsample = 1
#trials_MI, stim_labels = extract_trial_MI(nombres, stim_codes, orden_filtro, frecuencias, downsample)

	
def find_indexes_P300(nombre, path='Datos/P300_S01'):
	
	data = spio.loadmat('{}/{}.mat'.format(path, nombre))
	stims_begin = data['stimulusbegin']
	stims_type = data['stimulustype']
	samples = data['senial']
	
	stim_indexes = []
	stim_labels = []
	for i in range(len(stims_begin)-1):
		if stims_begin[i+1]-stims_begin[i] == 1:
			if stims_type[i+1] - stims_type[i] == 1:
				stim_indexes.append(i+1)
				stim_labels.append('target')
			else:
				stim_indexes.append(i+1)
				stim_labels.append('non_target')

	return stim_indexes, stim_labels, samples

def extract_trial_P300(nombres, orden_filtro, frecuencias, downsample):
	trials = {}
	sampling_freq = 256
	frecuencias = [frecuencias[i]/(sampling_freq*2) for i in range(len(frecuencias))]
	b, a = signal.butter(orden_filtro, frecuencias, btype = 'bandpass')
	
	trials =[]
	stim_labels_final = []
	for nombre in nombres: 
		stim_indexes, stim_labels, samples = find_indexes_P300(nombre)
		stim_labels_final.extend(stim_labels)
		
		for i in stim_indexes:
			trials.append(signal.filtfilt(b, a, samples[i:i+sampling_freq], axis = 0, padlen = None))
	
	if downsample > 1:
		for i in range(len(trials)):
			trials[i] = trials[i][::int(downsample)]
		
	return trials, stim_labels_final


nombres = ['ACS11_bis', 'ACS12_bis', 'ACS13_bis']
orden_filtro = 3
frecuencias = [1, 12]
downsample = 1
trials_P300, stim_labels_P300 = extract_trial_P300(nombres, orden_filtro, frecuencias, downsample)


def get_indexes(stim_labels, stim_code):
	indexes = []
	for i in range(len(stim_labels)):
		if stim_labels[i] == stim_code:
			indexes.append(i)
	return indexes


def channel_means(trials, stim_indexes):
	channel = {}
	for col in range(len(trials[0][0])):
		col = 0
		lista_cols = []
		for fila in range(len(trials[0][:,0])):
			lista_filas = []
			for stimulus in stim_indexes:
				lista_filas.append(trials_P300[stimulus][fila,col])
			lista_cols.append(np.mean(lista_filas))
		channel[col] = lista_cols
	return channel
		

target_indexes = get_indexes(stim_labels_P300, 'target')

channel_means_P300 = channel_means(trials_P300, target_indexes)

def dynamic_plot(trials, stim_indexes):
	fig = plt.figure()
	ax1 = fig.add_subplot(1,1,1)
	vector = np.arange(0, len(stim_indexes), 2)
	
	for i in vector:
		channel = channel_means(trials, stim_indexes[:i])
		ax1.clear()
		ax1.grid()
		ax1.plot(channel[0])
		ax1.set_title('{}'.format(i))
		plt.pause(0.05)
		
		
dynamic_plot(trials_P300, target_indexes)






	