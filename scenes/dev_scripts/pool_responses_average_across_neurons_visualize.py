#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 17:57:16 2019

@author: cesar
"""

import h5py

import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import shutil
import glob
import optparse
import os
import json
import pandas as pd
import numpy as np
import pylab as pl

import seaborn as sns
sns.set_style("ticks")
sns.set()
sns.set_color_codes()

def findOccurrences(s, ch):
    return [i for i, letter in enumerate(s) if letter == ch]

aggregate_root = '/n/coxfs01/cechavarria/2p-aggregate/scenes'

#sess_list = ['JC097_20190621','JC085_20190624','JC097_20190628','JC097_20190704','JC085_20190712']
#area = 'V1'

#sess_list = ['JC080_20190619','JC091_20190621','JC091_20190628','JC097_20190702','JC091_20190703','JC097_20190708']
#area = 'LM'

sess_list = ['JC091_20190625','JC091_20190701','JC091_20190705']
area = 'LI'


response_type = 'norm_df'
filter_crit = 'zscore'
filter_thresh = 1

# filter_crit = 'split_half_R'
# filter_thresh = .6

if ('norm' in response_type) or ('std' in response_type):
    i1 = findOccurrences(response_type,'_')[-1]
    fetch_data = response_type[i1+1:]
else:
    fetch_data = response_type
print(fetch_data)

#define paths
aggregate_file_dir = os.path.join(aggregate_root,area,'files')
fig_out_dir = os.path.join(aggregate_root,area,'figures')

if not os.path.isdir(fig_out_dir):
        os.makedirs(fig_out_dir)
        

#pool all neurons


#put things into pandas df for plotting
resp_dfs = []

animalids = []
sess_count = []
active_cell_count = 0
total_cell_count = 0

ylabel = 'Average Response'

for sess_idx, sess in enumerate(sess_list):
#    sess_idx = 0
#    sess = sess_list[sess_idx]
    print(sess)
    i1 = findOccurrences(sess,'_')[0]
    animalid = sess[0:i1]
    session = sess[i1+1:]
    
    
    if animalid not in animalids:
        animalids.append(animalid)
        sess_count.append(0)
    
    animal_idx = animalids.index(animalid)
    sess_count[animal_idx] = sess_count[animal_idx]+1
    
    #load data
    aggregate_file_dir = os.path.join(aggregate_root,area,'files','trial_responses')
    data_array_fn = '%s_%s_trial_response_array.hdf5'%(animalid, session)
    data_array_filepath = os.path.join(aggregate_file_dir, data_array_fn)
    data_grp = h5py.File(data_array_filepath, 'r')
    
    cell_rois = data_grp.attrs['s2p_cell_rois']
    
    curr_slice = 'Slice01'#hard,coding for now
    
    #unpack
    response_matrix = np.array(data_grp['/'.join([curr_slice, 'responses' ,fetch_data])])
    
    filter_crit_matrix_trials = np.array(data_grp['/'.join([curr_slice, 'responses' ,filter_crit])])
    
    #considering only cell rois
    response_matrix = response_matrix[:,:,cell_rois]
    ntrials,nconfigs,nrois = response_matrix.shape
    
    
    if filter_crit == 'zscore':
        filter_crit_matrix_trials = filter_crit_matrix_trials[:,:,cell_rois]
        filter_crit_matrix_mean = np.squeeze(np.mean(filter_crit_matrix_trials,0))
    elif filter_crit == 'simple_pval' or filter_crit == 'paired_pval' or filter_crit == 'perm_p' or 'stat' in filter_crit:
        filter_crit_matrix_mean = filter_crit_matrix_trials[:,cell_rois]
    elif filter_crit == 'split_half_R':
        filter_crit_matrix_mean = filter_crit_matrix_trials[cell_rois]
    
    data_grp.close()
    
    #avg over trials
    
    mean_response_matrix = np.mean(response_matrix,0)
    #split data into half by selecting odd or even trials
    mean_response_matrix_half1 = np.mean(response_matrix[0:ntrials:2,:,:],0)
    mean_response_matrix_half2 = np.mean(response_matrix[1:ntrials:2,:,:],0)
    
    #consider a config active if at least one of theversion of an image evoked a response above threshol
    
    if filter_crit == 'zscore' or 'tstat' in filter_crit:
        thresh = filter_thresh
        thresh_matrix = filter_crit_matrix_mean>thresh
    elif filter_crit == 'simple_pval' or filter_crit == 'paired_pval' or filter_crit == 'perm_p':
        thresh = filter_thresh
        thresh_matrix = np.logical_and(filter_crit_matrix_mean<thresh,filter_crit_matrix_mean>0)
    elif filter_crit == 'split_half_R':
        thresh = filter_thresh
        thresh_matrix= filter_crit_matrix_mean>thresh
        thresh_matrix = np.expand_dims(thresh_matrix,0)
        thresh_matrix = np.tile(thresh_matrix,(nconfigs,1))
    
    filter_matrix = np.ones((thresh_matrix.shape))*np.nan
    active_rois_per_config = np.nansum(thresh_matrix,1)
    for ridx in range(nrois):
        for idx in range(0,thresh_matrix.shape[0],3):
                if np.sum(thresh_matrix[idx:idx+3,ridx])>0:
                    filter_matrix[idx:idx+3,ridx] = 1
    
    #figure out some activity details
    active_cell_idx = np.nansum(filter_matrix,0)>0
    num_active_rois = np.nansum(np.nansum(filter_matrix,0)>0)
    frac_active_rois = num_active_rois/float(len(cell_rois))
    print('# active rois = %i'%(num_active_rois))
    print('frac active rois = %.04f'%(frac_active_rois))
    
    #normalize across configs within cell, if necessary
    if 'norm' in response_type:
        norm_response_array = np.empty((nconfigs,nrois))
    
        for ridx in range(nrois):
            norm_response_array[:,ridx] = mean_response_matrix[:,ridx]/np.nanmax(mean_response_matrix[:,ridx])
        mean_response_matrix = norm_response_array
    if 'std' in response_type:#standardise response by z-scoring across configs
        std_response_array = np.empty((nconfigs,nrois))
    
        for ridx in range(nrois):
            std_response_array[:,ridx] = (mean_response_matrix[:,ridx]-np.nanmean(mean_response_matrix[:,ridx]))/np.nanstd(mean_response_matrix[:,ridx])
        mean_response_matrix = std_response_array
    
    filt_response_array = mean_response_matrix*filter_matrix
    
    #average over images
    filter_response_cond_per_neuron = np.zeros((3,nrois))
    for cidx in range(3):
        filter_response_cond_per_neuron[cidx,:] = np.nanmean(filt_response_array[cidx:nconfigs:3,:],0)
        
    
    total_cell_count = total_cell_count + active_cell_idx.shape[0]
    active_cell_count = active_cell_count + np.nansum(active_cell_idx)
    #SHIFTING ORDER OF CONDITIONS - for intuitive arrangement
    sess_response = filter_response_cond_per_neuron[[0,2,1],:]
    
    
    
    if sess_idx == 0:
        response_array = sess_response[:,active_cell_idx]
    else:
        response_array = np.hstack((response_array,sess_response[:,active_cell_idx]))

      
frac_active_cells = active_cell_count/float(total_cell_count)


response_per_cond_mean = np.nanmean(response_array,1)
response_per_cond_se = np.nanstd(response_array,1)/np.sqrt(active_cell_count)

bar_loc = np.zeros((3,))
width = 0.4         # the width of the bars
xloc = 1
count = 0

for j in range(3):
    bar_loc[count] = xloc
    xloc = xloc + width
    count = count+1

fig = plt.figure(figsize=(8,8))
plt.bar(bar_loc[0],response_per_cond_mean[0],width,color = 'b',yerr = response_per_cond_se[0])
plt.bar(bar_loc[1],response_per_cond_mean[1],width,color = 'r',yerr = response_per_cond_se[1])
plt.bar(bar_loc[2],response_per_cond_mean[2],width,color = 'g',yerr = response_per_cond_se[2])

axes = plt.gca()
xmin, xmax = axes.get_xlim()
ymin, ymax = axes.get_ylim()
plt.axhline(y=0, xmin=xmin, xmax= xmax, linewidth=1, color='k',linestyle = '-')

xtick_loc = []
xtick_label = []

plt.xticks(xtick_loc,xtick_label)
plt.xlabel('Condition',fontsize = 15)
plt.ylabel('Average Response',fontsize = 15)
plt.suptitle('Average %s Across Neurons'%(response_type),fontsize = 15)


plt.text(bar_loc[0]-.25, ymax, 'n=%i, f=%.04f' % (active_cell_count, frac_active_cells), fontsize=10)


fig_fn = '%s_avg_response_per_cond_across_neurons_%s_thresh_%s_%.02f.png'%(area,response_type,filter_crit,filter_thresh)

fig_file_path = os.path.join(fig_out_dir, fig_fn)
plt.savefig(fig_file_path)
#plt.close()