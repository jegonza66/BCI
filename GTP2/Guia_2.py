#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 12:18:06 2019

@author: joaco
"""
import os
import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
import mne
from scipy import signal
import operator


wd = '/home/joaco/Desktop/joaco/facultad/BCI/Guías/GTP2'
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

def kl(p, q):
    """Kullback-Leibler divergence D(P || Q) for discrete distributions
    Parameters
    ----------
    p, q : array-like, dtype=float, shape=n
    Discrete probability distributions.
    """
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)

    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

#%% PCA
from mne.decoding import UnsupervisedSpatialFilter
from sklearn.decomposition import PCA

#GET DATA
data = spio.loadmat('Datos/DataEEG_EOG_bis.mat')
sampling_freq = data['sf'][0][0]
channel_names = [data['ChannelNames'][i][0][0]for i in range(len(data['ChannelNames']))]
tipos = ['eeg' for i in range(len(channel_names)-3)]+['eog' for i in range(3)]
info = mne.create_info(ch_names = channel_names[:], sfreq = sampling_freq, ch_types = tipos[:], montage = 'standard_1020')
channel_pos = []
for i in range(len(info['chs'])):
	channel_pos.append(info['chs'][i]['loc'][:2])
	
datos = data['Datos']
datos = np.swapaxes(datos,0,2)
datos_eeg = datos[:,:-3,:]
tiempo = [i/sampling_freq for i in range(len(datos[0][0]))]

#APLICAR PCA
sk_pca = PCA()
pca = UnsupervisedSpatialFilter(sk_pca, average = False)
data_pca = pca.fit_transform(datos_eeg)

#PLOT PRINCIPAL COMPONENTS
W = sk_pca.components_
M = np.linalg.inv(W)
M = M.transpose()

fig = plt.figure(figsize=(15, 25))
for i in range(len(M)):
	ax1 = fig.add_subplot(3,9,(i+1))
	ax1.set_title('PCA {}'.format(i))
	mne.viz.plot_topomap(M[i], np.array(channel_pos)[:-3], axes = ax1)
fig.tight_layout()
fig.suptitle('Principal components')

#PLOT RAW DATA AND PRINCIPAL COMPONENTS FOR 1 TRIAL EVERY CHANNEL
fig1 = plt.figure(figsize=(10,20))
plt.subplots_adjust(left = 0.07, bottom=0.08, right=0.95, top=0.93, wspace = 0.17, hspace = 0.3)
fig2 = plt.figure(figsize=(10,20))
plt.subplots_adjust(left = 0.07, bottom=0.08, right=0.95, top=0.93, wspace = 0.17, hspace = 0.3)
for i in range(len(datos[0])):
	ax1 = fig1.add_subplot(13,2,(i+1))
	ax1.grid()
	ax1.set_ylabel('{}'.format(channel_names[i]))
	ax1.plot(tiempo, datos[0][i])
	ax1.set_xlim([0,2])
	ax1.tick_params(axis='both', which='major', labelsize=8)
	if i < len(datos[0])-2:
		ax1.set_xticklabels([])
	else:
		ax1.set_xlabel('time[s]')
fig1.suptitle('Raw data')

for i in range(len(data_pca[0])):	
	ax2 = fig2.add_subplot(13,2,i+1)
	ax2.grid()
	ax2.set_ylabel('PCA {}'.format(i))
	ax2.plot(tiempo, data_pca[0][i])

	ax2.set_xlim([0,2])
	ax2.tick_params(axis='both', which='major', labelsize=8)
	if i < len(data_pca[0])-2:
		ax2.set_xticklabels([])
	else:
		ax2.set_xlabel('time[s]')
fig2.suptitle('Principal components')

#%% ICA
from mne.preprocessing import ICA
from mne.decoding import UnsupervisedSpatialFilter
from sklearn.decomposition import FastICA

data = spio.loadmat('Datos/DataEEG_EOG_bis.mat')
sampling_freq = data['sf'][0][0]
datos = data['Datos']
datos = np.swapaxes(datos,0,2)
datos_eeg = datos[:,:-3,:]
tiempo = [i/sampling_freq for i in range(len(datos[0][0]))]
channel_names = [data['ChannelNames'][i][0][0]for i in range(len(data['ChannelNames']))]
tipos = ['eeg' for i in range(len(data['ChannelNames'])-3)]+['eog' for i in range(3)]
info = mne.create_info(ch_names = channel_names, sfreq = sampling_freq, ch_types = tipos, montage = 'standard_1020')
channel_pos = []
for i in range(len(info['chs'])):
	channel_pos.append(info['chs'][i]['loc'][:2])
	
# MNE 
#eeg = mne.EpochsArray(datos, info)
#eeg.set_montage('standard_1020')
#eeg_temp = eeg.copy()
#
#ica = ICA(method = 'fastica', random_state = 1)
#ica.fit(eeg_temp)

#PLOT INDEPENDENT COMPONENTS MAL PUES ICA HACE PCA ANTES, ENTONCES LA MATRIZ VUELVE AL ESPACIO
#PCA EN VEZ DEL RAW

#UnMix = ica.unmixing_matrix_
#Mix = ica.mixing_matrix_
#
#fig = plt.figure(figsize=(15, 25))
#for i in range(len(Mix)):
#	ax1 = fig.add_subplot(3,9,(i+1))
#	ax1.set_title('ICA {}'.format(i))
#	mne.viz.plot_topomap(UnMix[i], np.array(channel_pos)[:-3], axes = ax1)
#fig.tight_layout()
#fig.suptitle('Independent components mne')

# PLOT ICA COMPONENTS BIEN
#ica.plot_components()

# MEDIANTE SKLEARN FASTICA 
sk_ica = FastICA()
ICA_transformer = UnsupervisedSpatialFilter(sk_ica, average = False)
data_ICA = ICA_transformer.fit_transform(datos_eeg)
Mix = sk_ica.mixing_.transpose()

fig = plt.figure(figsize=(15, 25))
for i in range(len(Mix)):
	ax1 = fig.add_subplot(3,9,(i+1))
	ax1.set_title('ICA {}'.format(i))
	mne.viz.plot_topomap(Mix[i], np.array(channel_pos)[:-3], axes = ax1)
fig.tight_layout()
fig.suptitle('Independent components sklearn')

###-------- PLOT AND INDEPENDENT COMPONENTS -------###
fig2 = plt.figure(figsize=(10,20))
plt.subplots_adjust(left = 0.1, bottom=0.1, right=0.95, top=0.93, wspace = 0.25, hspace = 0.3)
for i in range(len(data_ICA[0])):
    ax2 = fig2.add_subplot(11,2,i+1)
    ax2.grid()
    ax2.set_ylabel('ICA{}'.format(i))
    ax2.plot(tiempo, data_ICA[150][i])
    ax2.set_xlim([0,2])
    ax2.tick_params(axis='both', which='major', labelsize=8)
    if (i < len(data_ICA[0])-2):
        ax2.set_xticklabels([])
    else:
        ax2.set_xlabel('time[s]')
fig2.suptitle('Independent components')

#EXCLUDE ARTIFACTS AND PLOT FILTERED AND RAW DATA 
exclude_channels = [1,3,5,15]
data_ICA[:, exclude_channels, :] = 0
reconstructed_data = ICA_transformer.inverse_transform(data_ICA)

fig1 = plt.figure(figsize=(10,20))
plt.subplots_adjust(left = 0.07, bottom=0.08, right=0.95, top=0.93, wspace = 0.17, hspace = 0.3)
for i in range(len(reconstructed_data[0])):
	ax1 = fig1.add_subplot(11,2,(i+1))
	ax1.grid()
	ax1.set_ylabel('{}'.format(channel_names[i]))
	ax1.plot(tiempo, reconstructed_data[0][i], label= 'filtered data')
	ax1.plot(tiempo, datos[0][i], label = 'raw data')
	ax1.set_xlim([0,2])
	ax1.tick_params(axis='both', which='major', labelsize=8)
	if (i < len(reconstructed_data)-2):
		ax1.set_xticklabels([])
	else:
		ax1.set_xlabel('time[s]')
plt.legend(bbox_to_anchor=(1, 0.5))
fig1.suptitle('Filtered data')

#%% CSP
from mne.decoding import CSP

###------- LOAD DATA --------###	
data = spio.loadmat('Datos/DataEEG_EOG_bis.mat')
sampling_freq = data['sf'][0][0]
datos = data['Datos']
datos = np.swapaxes(datos,0,2)
tiempo = [i/sampling_freq for i in range(len(datos[0][0]))]
datos_filt = butter_bandpass_filter(datos, [8, 30], sampling_freq, 2, axis = 2)

labels = data['etiquetas'][0]
channel_names = [data['ChannelNames'][i][0][0]for i in range(len(data['ChannelNames']))]

###----- SET MNE DATA FORMAT -----###
tipos = ['eeg' for i in range(len(channel_names)-3)]+['eog' for i in range(3)]
info = mne.create_info(ch_names = channel_names[:-3], sfreq = sampling_freq, ch_types = tipos[:-3])
eeg = mne.EpochsArray(datos_filt[:,:-3,:], info)
eeg.set_montage('standard_1020')
eeg_temp = eeg.get_data()

###------ CSP ------###
csp = CSP(n_components=22, log=None, transform_into='csp_space')

# plot CSP patterns estimated on full data for visualization
eeg_CSP = csp.fit_transform(eeg_temp, labels)
csp.plot_patterns(eeg.info, components = [0,1,2,3,18,19,20,21], ch_type='eeg', units='Patterns (AU)', size=1.5)

# GRAFICO DOS CLASES TRANSFORMADAS PARA VER DISTINTA VARIANZA EN UN MISMO CANAL		
canales = [0,-1,1,-2,2,-3,3,-4]
fig = plt.figure(figsize=(15,25))
k = 1
for j,canal in enumerate(canales):
	ax1 = fig.add_subplot(4, 2, k)
	ax1.grid()
	ax1.set_xlim([0,2])
	ax1.set_title('Channel{}'.format(canal))
	for i in [0,1]:
		plt.plot(tiempo, eeg_CSP[i][canal], label = 'Class {}, var = {:.2f}'.format(labels[i], np.var(eeg_CSP[i][canal])))
	ax1.legend()
	k += 1
	if j < len(canales)-2:
		ax1.set_xticklabels([])
	else:
		ax1.set_xlabel('time[s]')
	
#GRAFICO LA MISMA CLASE EN LOS 3 PARES DE CANALES PARA VER CAMBIO DE VARIANZA
for i in [0,1]:
	fig = plt.figure()
	fig.suptitle('Class {}'.format(labels[i]))
	k=0
	for j in range(3):
		ax1 = fig.add_subplot(3,1,(j+1))
		ax1.grid()
		ax1.set_xlabel('time [s]')
		ax1.set_ylabel('filtered signal')
		for canal in [0+k, -(1+k)]:
			ax1.plot(tiempo, eeg_CSP[i][canal], label = 'channel {}, var = {:.2f}'.format(canal, np.var(eeg_CSP[i][canal])))
		ax1.legend()
		k+=1
	
#VEO EL PROMEDIO DE LA VARIANZA DE CADA CLASE PARA CADA CANAL
indexes_1 = get_indexes(labels, 1)
indexes_2 = get_indexes(labels, 2)

varianza_final_1 = []
varianza_final_2 = []
for i in range(len(eeg_CSP[0])):
	var_1 = []
	for index in indexes_1:
		var_1.append(np.var(eeg_CSP[index][i]))	
		
	var_2 = []
	for index in indexes_2:
		var_2.append(np.var(eeg_CSP[index][i]))
		
	varianza_final_1.append(np.mean(var_1))
	varianza_final_2.append(np.mean(var_2))

dif_var = [varianza_final_1[i]-varianza_final_2[i] for i in range(len(varianza_final_1))]

fig = plt.figure()
ax = fig.add_subplot(2,1,1)
ax.plot(dif_var)
ax.grid()
ax.set_ylabel('Var clase 1 - var clase 2')
ax1 = fig.add_subplot(2,1,2)	
ax1.plot(abs(np.array(dif_var)))
ax1.grid()
ax1.set_xlabel('Canal')
ax1.set_ylabel('Valor absoluto')
fig.suptitle('Varianzas por canal')

#%% 3.4
from mne.decoding import CSP

data = spio.loadmat('Datos/DataEEG_EOG_bis.mat')
sampling_freq = data['sf'][0][0]
datos = data['Datos']
tiempo = [i/sampling_freq for i in range(len(datos[0][0]))]
datos = np.swapaxes(datos,0,2)

datos_filt = {}
frecuencias = [[4,8], [8, 12], [13, 30]]
for frecuencia in frecuencias:
	datos_filt[str(frecuencia)] = butter_bandpass_filter(datos, frecuencia, sampling_freq, 2, axis = 2)

labels = data['etiquetas'][0]
channel_names = [data['ChannelNames'][i][0][0]for i in range(len(data['ChannelNames']))]

features = []
###----- SET MNE DATA FORMAT -----###
for frecuencia  in range(len(frecuencias)):
	tipos = ['eeg' for i in range(len(channel_names)-3)]+['eog' for i in range(3)]
	info = mne.create_info(ch_names = channel_names[:-3], sfreq = sampling_freq, ch_types = tipos[:-3])
	eeg = mne.EpochsArray(datos_filt[str(frecuencias[frecuencia])][:,:-3,:], info)
	eeg.set_montage('standard_1020')
	eeg_temp = eeg.get_data()
	
###------ CSP CON AVERAGE POWER Y LO -> QUE DEVUELVE LAS FEATURES EN VEZ DE LOS Z------###
	csp = CSP(n_components=22, log=True, transform_into='average_power')
	eeg_CSP = csp.fit_transform(eeg_temp, labels)
	
###---- ME QUEDO LOS PRIMEROS Y UTIMOS 3 CANALES QUE SON LAS FEATURES QUE MAS DIFERENCIAN LAS CLASES---###
	for i in range(len(eeg_CSP)):
		features.append(np.concatenate((eeg_CSP[i,:3], eeg_CSP[i,-3:])))

###---- ARMO LA LISTA DE FEATURES FINAL DE 18 P CADA TRIAL ----###
features_final = []
for i in range(len(eeg_CSP)):
	features_final.append(np.concatenate((features[i], features[i+288], features[i+2*288])))

###----ARMO HISTOGRAMAS PARA HACER KULLBACK ----###
indexes_1 = get_indexes(labels, 1)
indexes_2 = get_indexes(labels, 2)

features_final = np.transpose(np.array(features_final))

hist_1 = []
edges_1 = []
hist_2 = []
edges_2 = []
for i in range(len(features_final)):
	hist, edges = np.histogram(features_final[i][indexes_1], bins = 9, density = True)
	hist_1.append(hist)
	edges_1.append(edges)
	hist, edges = np.histogram(features_final[i][indexes_2], bins = 9, density = True)
	hist_2.append(hist)
	edges_2.append(edges)

Kullback = []
for i in range(len(hist_1)):
	p = hist_1[i]*np.diff(edges_1[i])
	q = hist_2[i]*np.diff(edges_2[i])
	Kullback.append([i,kl(p,q)])
	
Kullback.sort(key = operator.itemgetter(1), reverse = True)
	
plt.figure()
plt.grid()
plt.title('Kullback-Leibler')
plt.plot([Kullback[i][1]for i in range(len(Kullback))], '.')

# CONSERVO LAS FEATURES QUE ME DAN UNA DIFERENCIA ENTER CLASES MAYOR A 0.1 SEGUN KULLBACK
features_discriminantes = []
i=0
while(Kullback[i][1]>0.1):
	features_discriminantes.append(Kullback[i][0])
	i+=1

#%% 3.5
import mne
from mne.decoding import UnsupervisedSpatialFilter
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from mne.decoding import CSP

###------- LOAD DATA --------###	
data = spio.loadmat('Datos/DataEEG_EOG_bis.mat')
#locations = spio.loadmat('Datos/EMapAll.mat')
#locations = locations['EMapAll']
sampling_freq = data['sf'][0][0]
datos = data['Datos']
datos = np.swapaxes(datos,0,2)
tiempo = [i/sampling_freq for i in range(len(datos[0][0]))]
labels = data['etiquetas'][0]

###---- SELECT TRIAL FOR PLOTTING THEN FILTER TRIAL IN FRECUENCIES AND CAR ----###
trial_1 = np.transpose(datos[0, :, :])
trial_1_CAR = np.array(np.transpose(datos[0, :, :]))

for i in range(len(trial_1_CAR)):
	promedio = np.mean(trial_1_CAR[i])
	for j in range(len(trial_1_CAR[i])):
		trial_1_CAR[i,j] -= promedio

###---- SET MNE DATA FORMAT ----###
channel_names = [data['ChannelNames'][i][0][0]for i in range(len(data['ChannelNames']))]
tipos = ['eeg' for i in range(len(channel_names)-3)]+['eog' for i in range(3)]
info = mne.create_info(ch_names = channel_names[:], sfreq = sampling_freq, ch_types = tipos[:], montage = 'standard_1020')
channel_pos = []
for i in range(len(info['chs'])):
	channel_pos.append(info['chs'][i]['loc'][:2])

###---- PLOT EEG RAW DATA OVER 1 TRIAL ----###
	
#DEFINIR PCA
pca = PCA()

fig1 = plt.figure(figsize = (15,10))
ax1 = fig1.add_subplot(2,3,1)
ax2 = fig1.add_subplot(2,3,2)
ax3 = fig1.add_subplot(2,3,3)
ax4 = fig1.add_subplot(2,3,4)
ax5 = fig1.add_subplot(2,3,5)

for i in range(200):
	ax1.clear()
	ax1.set_title('Trial 1 - CAR EEG data')
	mne.viz.plot_topomap(trial_1_CAR[i][:-3], np.array(channel_pos[:-3]), axes = ax1)
	ax2.clear()
	ax2.set_title('Trial 1 - RAW EEG data')
	mne.viz.plot_topomap(trial_1[i][:-3], np.array(channel_pos[:-3]), axes = ax2)
	
	#APLICAR PCA
	data_pca = pca.fit_transform(trial_1[i])
	W = pca.components_
	M = np.linalg.inv(W)
	M = M.transpose()
	
	ax3.clear()
	mne.viz.plot_topomap(M[i][:-3], np.array(channel_pos)[:-3], axes = ax3)
	plt.pause(0.05)
	