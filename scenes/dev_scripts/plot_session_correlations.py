#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 14:35:19 2019

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
import scipy.stats as stats
import seaborn as sns
sns.set_style("ticks")
sns.set()
sns.set_color_codes()

def findOccurrences(s, ch):
    return [i for i, letter in enumerate(s) if letter == ch]

def normalize_across_rows(in_array):

    min_val = np.min(in_array,0)
    max_val = np.max(in_array,0)

    min_mat = np.tile(min_val,(in_array.shape[0],1))
    max_mat = np.tile(max_val,(in_array.shape[0],1))
    range_mat = max_mat - min_mat

    out_array = np.true_divide(in_array-min_mat,range_mat)

    out_array[np.isnan(out_array)] = 0
    
    return out_array


def get_upper_triangle_values(in_array):
    mask = np.ones(in_array.shape)
    tmp = np.tril(mask,0)
    out_val = in_array[np.where(tmp==0)]
    
    return out_val

def correlate_split_half_RDM(half1_array,half2_array):
    R_all_half1 = np.corrcoef(half1_array)
    R_all_half2 = np.corrcoef(half2_array)

    all_half1_values = get_upper_triangle_values(R_all_half1)
    all_half2_values = get_upper_triangle_values(R_all_half2)
    
    R = np.corrcoef(all_half1_values,all_half2_values)
    
    return R[0,1]

def corrfunc(x, y, **kws):
    (r, p) = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.2f} ".format(r),
                xy=(.1, .9), xycoords=ax.transAxes)

class struct: pass

opts = struct()
opts.rootdir = '/n/coxfs01/2p-data'
opts.animalid = 'JC097'
opts.session = '20190705'
opts.acquisition = 'FOV1_zoom4p0x'
opts.traceid = 'traces004'
opts.combined_run = 'scenes_combined'

animal = opts.animalid
session = opts.session
response_type = 'df'
filter_crit = 'zscore'
filter_thresh = 1

traceid = '%s_s2p'%(opts.traceid)
#% Set up paths:    
acquisition_dir = os.path.join(opts.rootdir, opts.animalid, opts.session, opts.acquisition)
traceid_dir = os.path.join(acquisition_dir, opts.combined_run,'traces',traceid)

run_dir = traceid_dir.split('/traces')[0]
trace_arrays_dir = os.path.join(traceid_dir,'files')
paradigm_dir = os.path.join(acquisition_dir, opts.combined_run, 'paradigm')


responses_dir = os.path.join(acquisition_dir, opts.combined_run,'filtered_responses',traceid)

fig_out_dir = os.path.join(acquisition_dir, opts.combined_run,'responses',traceid,'figures')
if not os.path.exists(fig_out_dir):
    os.makedirs(fig_out_dir)
    


if 'norm' in response_type:
    i1 = findOccurrences(response_type,'_')[-1]
    fetch_data = response_type[i1+1:]
else:
    fetch_data = response_type

#get file with trial-by-trial responses
responses_dir = os.path.join(acquisition_dir, opts.combined_run,'responses',traceid)
data_array_dir = os.path.join(responses_dir, 'data_arrays')

resp_array_fn = 'trial_response_array.hdf5'
resp_array_filepath = os.path.join(data_array_dir, resp_array_fn)
data_grp = h5py.File(resp_array_filepath, 'r')

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

def get_self_correlation(data_half1,data_half2,zscore_dim=0,cfg_dim=0):
    nconfigs = data_half1.shape[cfg_dim]
    
    #z-score a
    data_half1_zscore = stats.zscore(data_half1,zscore_dim)
    data_half2_zscore = stats.zscore(data_half2,zscore_dim)

    data_half1_zscore[np.isnan(data_half1_zscore)]=0
    data_half2_zscore[np.isnan(data_half2_zscore)]=0
    
    self_R = np.zeros((nconfigs,))
    for cfg_idx in range(nconfigs):
        if cfg_dim == 0:
            self_R[cfg_idx] = np.corrcoef(data_half1_zscore[cfg_idx,:],data_half2_zscore[cfg_idx,:])[0,1]
        elif cfg_dim ==1:
            self_R[cfg_idx] = np.corrcoef(data_half1_zscore[:,cfg_idx],data_half2_zscore[:,cfg_idx])[0,1]
    return self_R

    
    


dset_response = mean_response_matrix
dset_response_active = mean_response_matrix[:,active_cell_idx]

dset_response_half1 = mean_response_matrix_half1
dset_response_half2 = mean_response_matrix_half2

dset_response_active_half1 = mean_response_matrix_half1[:,active_cell_idx]
dset_response_active_half2 = mean_response_matrix_half2[:,active_cell_idx]

#z-score and get correlation
dset_response_zscore = stats.zscore(dset_response,0)
dset_response_active_zscore = stats.zscore(dset_response_active,0)

dset_response_zscore[np.isnan(dset_response_zscore)]=0
dset_response_active_zscore[np.isnan(dset_response_active_zscore)]=0

##----------COMPARING POP PROFILE------------

R_dset_zscore = np.corrcoef(dset_response_zscore)
R_dset_active_zscore = np.corrcoef(dset_response_active_zscore)

#get self correlation
self_R = get_self_correlation(dset_response_half1,dset_response_half2)
self_R_active = get_self_correlation(dset_response_active_half1,dset_response_active_half2)

#replace diagonal values
nconfigs = self_R.size
for cfg_idx in range(nconfigs):
    R_dset_zscore[cfg_idx,cfg_idx] = self_R[cfg_idx]
    R_dset_active_zscore[cfg_idx,cfg_idx] = self_R_active[cfg_idx]


plt.figure(figsize=(12, 10))
ax = sns.heatmap(R_dset_zscore,center = 0,annot=False, cmap = 'RdBu_r',vmax = 1)
fig_name = '%s_%s_all_zscore_%s_RSA.png'%(animal,session,response_type)
fig_fn = os.path.join(fig_out_dir,fig_name)
fig = ax.get_figure()
fig.savefig(fig_fn) 
# plt.close()


plt.figure(figsize=(12, 10))
ax = sns.heatmap(R_dset_active_zscore,center = 0,annot=False, cmap = 'RdBu_r',vmax = 1)
fig_name = '%s_%s_active_zscore_%s_thresh_%.02f_RSA.png'%(animal,session,response_type,filter_thresh)
fig_fn = os.path.join(fig_out_dir,fig_name)
fig = ax.get_figure()
fig.savefig(fig_fn) 
# plt.close()
   


  ##----------COMPARING NEURONS----------- 
R_cells_zscore = np.corrcoef(np.transpose(dset_response_zscore))
R_cells_active_zscore = np.corrcoef(np.transpose(dset_response_active_zscore))


#get self correlation
self_R = get_self_correlation(dset_response_half1,dset_response_half2,0,1)
self_R_active = get_self_correlation(dset_response_active_half1,dset_response_active_half2,0,1)

#replace diagonal values
nconfigs = self_R.size
for cfg_idx in range(nconfigs):
    R_cells_zscore[cfg_idx,cfg_idx] = self_R[cfg_idx]
    
nconfigs = self_R_active.size
for cfg_idx in range(nconfigs):
    R_cells_active_zscore[cfg_idx,cfg_idx] = self_R_active[cfg_idx]

plt.figure(figsize=(12, 10))
ax = sns.heatmap(R_cells_zscore,center = 0,annot=False, cmap = 'RdBu_r',vmax = 1)
fig_name = '%s_%s_zscore_%s_cell_correlation.png'%(animal,session,response_type)
fig_fn = os.path.join(fig_out_dir,fig_name)
fig = ax.get_figure()
fig.savefig(fig_fn) 

plt.figure(figsize=(12, 10))
ax = sns.heatmap(R_cells_active_zscore,center = 0,annot=False, cmap = 'RdBu_r',vmax = 1)
fig_name = '%s_%s_active_zscore_%s_thresh_%.02f_cell_correlation.png'%(animal,session,response_type,filter_thresh)
fig_fn = os.path.join(fig_out_dir,fig_name)
fig = ax.get_figure()
fig.savefig(fig_fn) 

#
#get average non-self correlation value for each cell
non_self = np.ones(R_cells_zscore.shape)
di =  np.diag_indices(non_self.shape[0])
non_self[di] = np.nan

R_non_self_mean = np.nanmean(R_cells_zscore*non_self,1)


#we also have self_R

non_self = np.ones(R_cells_active_zscore.shape)
di =  np.diag_indices(non_self.shape[0])
non_self[di] = np.nan

R_non_self_active_mean = np.nanmean(R_cells_active_zscore*non_self,1)

print(R_non_self_active_mean)
#we also have self_R_active


###-----checking trial-by-trial correlations

#get trial correlation for all cells


ntrials = response_matrix.shape[0]*response_matrix.shape[1]
trial_dset = np.reshape(response_matrix,(ntrials,response_matrix.shape[2]))
trial_active_dset = trial_dset[:,active_cell_idx]

R_trials_cell = np.corrcoef(np.transpose(trial_dset))
R_trials_cell_active = np.corrcoef(np.transpose(trial_active_dset))

plt.figure(figsize=(12, 10))
ax = sns.heatmap(R_trials_cell,center = 0,annot=False, cmap = 'RdBu_r',vmax = 1)
fig_name = '%s_%s_%s_cell_trial_correlation.png'%(animal,session,response_type)
fig_fn = os.path.join(fig_out_dir,fig_name)
fig = ax.get_figure()
fig.savefig(fig_fn) 


plt.figure(figsize=(12, 10))
ax = sns.heatmap(R_trials_cell_active,center = 0,annot=False, cmap = 'RdBu_r',vmax = 1)
fig_name = '%s_%s_active_%s_cell_trial_correlation.png'%(animal,session,response_type)
fig_fn = os.path.join(fig_out_dir,fig_name)
fig = ax.get_figure()
fig.savefig(fig_fn) 


#save correlation values to file

data_array_dir = os.path.join(responses_dir, 'data_arrays')
data_array_fn = 'correlation_matrices_%s_responses_thresh_%s_%i.hdf5'%(response_type,filter_crit, filter_thresh)
data_array_filepath = os.path.join(data_array_dir, data_array_fn)
data_grp = h5py.File(data_array_filepath, 'w')


stim_all = data_grp.create_dataset('stimulus_correlation_all_cells', R_cells_zscore.shape, R_cells_zscore.dtype)
stim_all[...] = R_cells_zscore

stim_active = data_grp.create_dataset('stimulus_correlation_active_cells', R_cells_active_zscore.shape, R_cells_active_zscore.dtype)
stim_active[...] = R_cells_active_zscore

trial_all = data_grp.create_dataset('trial_correlation_all_cells', R_trials_cell.shape, R_trials_cell.dtype)
trial_all[...] = R_trials_cell

trial_active = data_grp.create_dataset('trial_correlation_active_cells', R_trials_cell_active.shape, R_trials_cell_active.dtype)
trial_active[...] = R_trials_cell_active

data_grp.close()
