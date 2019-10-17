#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 16:55:20 2019

@author: cesar
"""

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
opts.combined_run = 'scenes_combined'

animal = opts.animalid
session = opts.session
response_type = 'df'
filter_crit = 'zscore'
filter_thresh = 1


acquisition_dir = os.path.join(opts.rootdir, opts.animalid, opts.session, opts.acquisition)

fig_out_dir = os.path.join(acquisition_dir, opts.combined_run,'neuropil_correction_figures')
if not os.path.exists(fig_out_dir):
    os.makedirs(fig_out_dir)

#--- FOR JC097_20190621 testing   
#traceid_list = ['traces002_s2p','traces003_s2p','traces004_s2p','traces005_s2p','traces006_s2p']
#np_correction = np.array([0,0.35,.70,1.05,1.40])

#traceid_list = ['traces008_s2p','traces009_s2p','traces010_s2p','traces011_s2p','traces012_s2p','traces013_s2p','traces020_s2p']
#traceid_list = ['traces014_s2p','traces015_s2p','traces016_s2p','traces017_s2p','traces018_s2p','traces019_s2p','traces021_s2p']
traceid_list = ['traces001_s2p','traces002_s2p','traces003_s2p','traces004_s2p','traces005_s2p','traces006_s2p','traces007_s2p']
np_correction = np.array([0,0.25,.50,.75,1.00,1.25,1.50])

R_trial_non_self_med_med = np.zeros((len(traceid_list),))
R_trial_non_self_med_sd = np.zeros((len(traceid_list),))
R_trial_non_self_min_med = np.zeros((len(traceid_list),))
R_trial_non_self_min_sd = np.zeros((len(traceid_list),))
R_trial_non_self_max_med = np.zeros((len(traceid_list),))
R_trial_non_self_max_sd = np.zeros((len(traceid_list),))

R_stim_non_self_med_med = np.zeros((len(traceid_list),))
R_stim_non_self_med_sd = np.zeros((len(traceid_list),))
R_stim_non_self_min_med = np.zeros((len(traceid_list),))
R_stim_non_self_min_sd = np.zeros((len(traceid_list),))
R_stim_non_self_max_med = np.zeros((len(traceid_list),))
R_stim_non_self_max_sd = np.zeros((len(traceid_list),))

R_stim_self_med_med = np.zeros((len(traceid_list),))
R_stim_self_med_sd = np.zeros((len(traceid_list),))


R_stim_active_non_self_med_med = np.zeros((len(traceid_list),))
R_stim_active_non_self_med_sd = np.zeros((len(traceid_list),))

R_stim_active_self_med_med = np.zeros((len(traceid_list),))
R_stim_active_self_med_sd = np.zeros((len(traceid_list),))

num_active_cells = np.zeros((len(traceid_list),))

for count, traceid in enumerate(traceid_list):
    #count = 0 
    #traceid = 'traces002_s2p'
    
    
    responses_dir = os.path.join(acquisition_dir, opts.combined_run,'responses',traceid)
    data_array_dir = os.path.join(responses_dir, 'data_arrays')
    data_array_fn = 'correlation_matrices_%s_responses_thresh_%s_%i.hdf5'%(response_type,filter_crit, filter_thresh)
    data_array_filepath = os.path.join(data_array_dir, data_array_fn)
    data_grp = h5py.File(data_array_filepath, 'r')
    
    R_cell_trial = np.array(data_grp['trial_correlation_all_cells'])
    
    R_cell_stim = np.array(data_grp['stimulus_correlation_all_cells'])
    
    R_active_stim = np.array(data_grp['stimulus_correlation_active_cells'])

    
    data_grp.close()
    
    #get non-self correlation for trial-by-trial correlation
    non_self = np.ones(R_cell_trial.shape)
    di =  np.diag_indices(non_self.shape[0])
    non_self[di] = np.nan
    
    R_trial_non_self_med_per_cell = np.nanmedian(R_cell_trial*non_self,1)
    R_trial_non_self_min_per_cell = np.nanmin(R_cell_trial*non_self,1)
    R_trial_non_self_max_per_cell = np.nanmax(R_cell_trial*non_self,1)
    
    R_trial_non_self_med_med[count] = np.median(R_trial_non_self_med_per_cell)
    R_trial_non_self_med_sd[count] = np.std(R_trial_non_self_med_per_cell)
    
    R_trial_non_self_min_med[count] = np.median(R_trial_non_self_min_per_cell)
    R_trial_non_self_min_sd[count] = np.std(R_trial_non_self_min_per_cell)
    
    R_trial_non_self_max_med[count] = np.median(R_trial_non_self_max_per_cell)
    R_trial_non_self_max_sd[count] = np.std(R_trial_non_self_max_per_cell)
    
    #get non-self correlation for stimulus-by-stimulus correlation
    non_self = np.ones(R_cell_stim.shape)
    
    di =  np.diag_indices(non_self.shape[0])
    non_self[di] = np.nan
    
    R_stim_non_self_med_per_cell = np.nanmedian(R_cell_stim*non_self,1)
    R_stim_non_self_min_per_cell = np.nanmin(R_cell_stim*non_self,1)
    R_stim_non_self_max_per_cell = np.nanmax(R_cell_stim*non_self,1)
    
    R_stim_non_self_med_med[count] = np.median(R_stim_non_self_med_per_cell)
    R_stim_non_self_med_sd[count] = np.std(R_stim_non_self_med_per_cell)
    
    R_stim_non_self_min_med[count] = np.median(R_stim_non_self_min_per_cell)
    R_stim_non_self_min_sd[count] = np.std(R_stim_non_self_min_per_cell)
    
    R_stim_non_self_max_med[count] = np.median(R_stim_non_self_max_per_cell)
    R_stim_non_self_max_sd[count] = np.std(R_stim_non_self_max_per_cell)
    
    #for active cells
    non_self = np.ones(R_active_stim.shape)
    di =  np.diag_indices(non_self.shape[0])
    non_self[di] = np.nan
    
    R_stim_active_non_self_med_per_cell = np.nanmedian(R_active_stim*non_self,1)
    R_stim_active_non_self_med_med[count] = np.median(R_stim_active_non_self_med_per_cell)
    R_stim_active_non_self_med_sd[count] = np.std(R_stim_active_non_self_med_per_cell)

    
    #get self correlation for stimulus-by-stimulus correlation
    selfM = np.ones(R_cell_stim.shape)*np.nan
    di =  np.diag_indices(selfM.shape[0])
    selfM[di] = 1
    
    num_active_cells[count] = non_self.shape[0]
    
    R_stim_self_med_per_cell = np.nanmedian(R_cell_stim*selfM,1)

    R_stim_self_med_med[count] = np.median(R_stim_self_med_per_cell)
    R_stim_self_med_sd[count] = np.std(R_stim_self_med_per_cell)
    
    #for active cells
    selfM = np.ones(R_active_stim.shape)*np.nan
    di =  np.diag_indices(selfM.shape[0])
    
    selfM[di] = 1
    
    R_stim_active_self_med_per_cell = np.nanmedian(R_active_stim*selfM,1)

    R_stim_active_self_med_med[count] = np.median(R_stim_active_self_med_per_cell)
    R_stim_active_self_med_sd[count] = np.std(R_stim_active_self_med_per_cell)
    


#plot trial correlations
plt.figure(figsize=(10, 10))
plt.errorbar(np_correction,R_trial_non_self_med_med,R_trial_non_self_med_sd)
plt.xlabel('Neuropil Coefficient')
plt.ylabel('Median Correlation')
plt.suptitle(' Median Trial Correlation')
fig_name = '%s_%s_%s_median_trial_correlation_vs_np_coefficient_all_cells.png'%(animal,session,response_type)
fig_fn = os.path.join(fig_out_dir,fig_name)
plt.savefig(fig_fn) 

#plt.figure(figsize=(10, 10))
#plt.errorbar(np_correction,R_trial_non_self_min_med,R_trial_non_self_min_sd)
#
#plt.figure(figsize=(10, 10))
#plt.errorbar(np_correction,R_trial_non_self_max_med,R_trial_non_self_max_sd)

plt.figure(figsize=(10, 10))
plt.errorbar(np_correction,R_stim_non_self_med_med,R_stim_non_self_med_sd)
plt.errorbar(np_correction,R_stim_self_med_med,R_stim_self_med_sd)
plt.legend(['non-self','self'])
plt.xlabel('Neuropil Coefficient')
plt.ylabel('Median Correlation')
plt.suptitle(' Median Stimulus Correlation | All Cells')
fig_name = '%s_%s_%s_median_stim_correlation_vs_np_coefficient_all_cells.png'%(animal,session,response_type)
fig_fn = os.path.join(fig_out_dir,fig_name)
plt.savefig(fig_fn) 

plt.figure(figsize=(10, 10))
plt.errorbar(np_correction,R_stim_active_non_self_med_med,R_stim_active_non_self_med_sd)
plt.errorbar(np_correction,R_stim_active_self_med_med,R_stim_active_self_med_sd)
plt.legend(['non-self','self'])
plt.xlabel('Neuropil Coefficient')
plt.ylabel('Median Correlation')
plt.suptitle(' Median Stimulus Correlation | Active Cells')
fig_name = '%s_%s_%s_median_trial_correlation_vs_np_coefficient_active_cells_thresh_%s_%.02f.png'%(animal,session,response_type,filter_crit,filter_thresh)
fig_fn = os.path.join(fig_out_dir,fig_name)
plt.savefig(fig_fn) 

plt.figure(figsize=(10, 10))
plt.plot(np_correction,num_active_cells)
plt.xlabel('Neuropil Coefficient')
plt.ylabel('Cell Count')
plt.suptitle(' Median Stimulus Correlation | Active Cells')
fig_name = '%s_%s_%s_num_active_cells_vs_np_coefficient_active_cells_thresh_%s_%.02f.png'%(animal,session,response_type,filter_crit,filter_thresh)
fig_fn = os.path.join(fig_out_dir,fig_name)
plt.savefig(fig_fn) 

