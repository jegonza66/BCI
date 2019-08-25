#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 10:18:35 2019

@author: joaco
"""
#%% MODULOS Y PATH

import os
import numpy as np
import scipy.io as spio
from scipy import signal
import matplotlib.pyplot as plt

wd = '/home/joaco/Desktop/joaco/facultad/BCI/Guías/GTP1'
os.chdir(wd)

#%% DEFINICINO FUNCIONES

def index(data, valor):
	index = np.where(np.asarray(data) == valor)[0][0]
	return index

def butter_bandpass_filter(data, frecuencias, sampling_freq, order, axis):
	frecuencias = [frecuencias[i]/(sampling_freq/2) for i in range(len(frecuencias))]
	b, a = signal.butter(order, frecuencias, btype='band')
	y = signal.filtfilt(b, a, data, axis = axis, padlen = None)
	return y


def find_indexes_MI(nombre, stim_codes, wanted_channels, path = 'Datos/MI_S01'):
	
	data = spio.loadmat('{}/{}.mat'.format(path, nombre))
	sample_time = data['sampleTime']
	sampling_freq = data['samplingFreq'][0][0]
	stims = data['stims']
	muestras = np.transpose(data['samples'])
	original_channels = [data['channelNames'][0][i][0]for i in range(len(data['channelNames'][0]))]
	
	if wanted_channels:
		samples = []
		channels = []
		for i, channel in enumerate(original_channels):
			if channel in wanted_channels:
				samples.append(muestras[i])
				channels.append(channel)
		samples = np.transpose(samples)
	else:
		samples = np.transpose(muestras)
		channels = original_channels
		
	stim_indexes = []
	stim_labels = []
	for i in range(len(stims[:,0])):
		if (any (stims[i,1] == stim_codes[j] for j in range(len(stim_codes)))):
			stim_indexes.append(int(stims[i,0]*sampling_freq))
			stim_labels.append(stims[i,1])
			
	return stim_indexes, stim_labels, samples, channels, sampling_freq


def extract_trial_MI(nombres, stim_codes, wanted_channels, orden_filtro, frecuencias, downsample):
	trials = []
	stim_labels_final = []
	
	for nombre in nombres:
		stim_indexes, stim_labels, samples, channels, sampling_freq = find_indexes_MI(nombre, stim_codes, wanted_channels)
		stim_labels_final.extend(stim_labels)
		
		if orden_filtro:
			for i in stim_indexes:
				trials.append(butter_bandpass_filter(samples[i-3*sampling_freq:i+5*sampling_freq], frecuencias, sampling_freq, orden_filtro, axis = 0))
				
		else:
			for i in stim_indexes:
				trials.append(samples[i-3*sampling_freq:i+5*sampling_freq])
				
	if downsample > 1:
		for i in range(len(trials)):
			trials[i] = trials[i][::int(downsample)]
	
	return trials, stim_labels_final, channels, sampling_freq
	
	
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
	sampling_freq = 256
	
	trials =[]
	stim_labels_final = []
	for nombre in nombres: 
		stim_indexes, stim_labels, samples = find_indexes_P300(nombre)
		stim_labels_final.extend(stim_labels)
		
		for i in stim_indexes:
			trials.append(butter_bandpass_filter(samples[i:i+sampling_freq], frecuencias, sampling_freq, orden_filtro, axis = 0))
	
	if downsample > 1:
		for i in range(len(trials)):
			trials[i] = trials[i][::int(downsample)]
		
	return trials, stim_labels_final


def get_indexes(stim_labels, stim_code):
	indexes = []
	for i in range(len(stim_labels)):
		if stim_labels[i] == stim_code:
			indexes.append(i)
	return indexes


def channel_means(trials, stim_indexes):
	channel = []
	for col in range(len(trials[0][0])):
		col = 0
		lista_cols = []
		for fila in range(len(trials[0][:,0])):
			lista_filas = []
			for trial in stim_indexes:
				lista_filas.append(trials[trial][fila,col])
			lista_cols.append(np.mean(lista_filas))
		channel.append(lista_cols)
	return channel
		

def dynamic_plot(trials, stim_indexes_1, stim_indexes_2):
	fig = plt.figure()
	ax1 = fig.add_subplot(1,1,1)
	vector = np.arange(2, 302, 2)
	
	for i in vector:
		channel_stim_1 = channel_means(trials, stim_indexes_1[:i])
		channel_stim_2 = channel_means(trials, stim_indexes_2[:i])
		
		ax1.clear()
		ax1.grid()
		ax1.plot(channel_stim_1[0], label = 'Target stimulus')
		ax1.plot(channel_stim_2[0], label = 'Non taget stimulus')
		ax1.set_title('{}'.format(i))
		plt.legend()
		plt.pause(0.05)

#%% P300
		
nombres = ['ACS11_bis', 'ACS12_bis', 'ACS13_bis']
orden_filtro = 3
frecuencias = [1, 12]
downsample = 1
trials_P300, stim_labels_P300 = extract_trial_P300(nombres, orden_filtro, frecuencias, downsample)

target_indexes = get_indexes(stim_labels_P300, 'target')
non_target_indexes = get_indexes(stim_labels_P300, 'non_target')	
		
dynamic_plot(trials_P300, target_indexes, non_target_indexes)

#%% MI

nombres = ['S01_FILT_S1R1', 'S01_FILT_S1R2', 'S01_FILT_S1R3', 'S01_FILT_S1R4']
stim_codes = [770, 772]
wanted_channels = '’F5’;’F3’;’F1’;’Fz’;’F2’;’F4’;’F6’;’FC5’;’FC3’;’FC1’;’FCz’;’FC2’;’FC4’;’FC6’;’C5’;’C3’;’C1’;’Cz’;’C2’;’C4’;’C6’;’CP5’;’CP3’;’CP1’;’CPz’;’CP2’;’CP4’;’CP6’'.replace(';',',').replace(' ', '').replace('’', '').split(',')   
orden_filtro = 2
frecuencias = [0.1, 40]
downsample = 1
trials_MI, stim_labels, channels, sampling_freq = extract_trial_MI(nombres, stim_codes, wanted_channels, orden_filtro, frecuencias, downsample)

tiempo = [i/sampling_freq-3 for i in range(len(trials_MI[0]))]
plot_channels = ['C3']
frecuencies = [[8, 11], [26, 30]]
for channel in plot_channels:
	for frecuencia in frecuencies:
		for stimulus in stim_codes:
			trials = [abs(butter_bandpass_filter(trials_MI[i], frecuencia, 512, 2, axis = 0)) for i in range(len(trials_MI))]
			indexes = get_indexes(stim_labels, stimulus)
			channels_final = channel_means(trials, indexes)
			plt.figure()
			plt.grid()
			plt.title('Channel: {} - Stimulus: {} - Frecuencies: {}-{}'.format(channel, stimulus,frecuencia[0], frecuencia[1]))
			plt.plot(tiempo, channels_final[index(channels, channel)])
			plt.xlabel('time [s]')
			plt.ylabel('Tension [?]')

#%% SSVEP
		
from numpy.fft import fft, fftfreq

nombres = ['OFFLINE_subj1']
labels_file_name = 'labels'
stim_codes = [1, 2, 3, 4]
occipital_channels = ['EOGL', 'EOGR', 'EOGC']	
sampling_freq = 1000


def SSVEP(nombres, labels_file_name, stim_codes, occipital_channels, sampling_freq, path = 'Datos/SSVEP'):
			
	###---------------- LOAD DATA --------------------###
			
	labels = spio.loadmat('{}/{}.mat'.format(path, labels_file_name))
	channel_names = [labels['labels'][i][0][0] for i in range(len(labels['labels']))]
	
	for nombre in nombres:
		data = spio.loadmat('{}/{}.mat'.format(path, nombre))
		stim = [data['circle_order'][i][0] for i in range(len(data['circle_order']))]
	
	###----------- GET STIMULUS INDEXES -------------###
		stim_indexes = {}
		for i in stim_codes:
			stim_indexes[i] = [j for j in range(len(stim)) if stim[j] == i]
		
	###----------- GET DESIRED CHANNELS -------------###	
		indices_occipitales = []
		
		for j in range(len(occipital_channels)):
			for i in range(len(channel_names)):
				if (channel_names[i] == occipital_channels[j]):
					indices_occipitales.append(i)
		
		data_occipitales = {}
		for i, channel in zip(indices_occipitales, occipital_channels):
			data_occipitales[channel] = np.transpose(data['data_matrix'][i])
	
	###----------- FOURIER TRANSFORM DATA -----------###
		espectro_data = {}
		for channel in occipital_channels:
			espectro_data[channel] = fft(data_occipitales[channel], axis = 1)
			
		frequencies = fftfreq(len(espectro_data[occipital_channels[0]][0]), d=1/sampling_freq)
	
	###----- MEAN FOR STIMULUS TYPE AND CHANNEL -----###
		mean_espectro = {}
		for channel in occipital_channels:
			canal = {}
			for i in stim_indexes.keys():
				estimulo = []
				for k in range(len(espectro_data[channel][0])):
					muestra = []
					for j in stim_indexes[i]:
						muestra.append(espectro_data[channel][j][k])
					estimulo.append(np.mean(muestra))
				canal[i] = estimulo
			mean_espectro[channel] = canal
	
	return mean_espectro, frequencies

mean_espectro, frequencies = SSVEP(nombres, labels_file_name, stim_codes, occipital_channels, sampling_freq)
		
plot_frequencies = frequencies[index(frequencies, 4): index(frequencies, 16)]

plot_spectrum = {}
for channel in occipital_channels:
	channel_data = {}
	for stim in stim_codes:
		channel_data[stim] = mean_espectro[channel][stim][index(frequencies, 4): index(frequencies, 16)]
	plot_spectrum[channel] = channel_data

###------------------ PLOT --------------------###
for i in stim_codes:
	plt.figure()
	plt.grid()
	plt.title('Estímulo: {}'.format(i))
	plt.xlabel('Frecuencia [Hz]')
	plt.ylabel('Amplitud')
	plt.plot(plot_frequencies, np.abs(plot_spectrum['EOGL'][i]))
	
		
	
		
		
		
		
		
	