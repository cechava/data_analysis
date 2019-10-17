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



dset_list = ['V1_JC085_20190624','V1_JC085_20190712','V1_JC097_20190621','V1_JC097_20190628',\
             'V1_JC097_20190704',\
             'LM_JC080_20190619','LM_JC091_20190621','LM_JC091_20190703','LM_JC091_20190628',\
             'LM_JC097_20190702','LM_JC097_20190708','LM_JC085_20190704',\
             'LI_JC091_20190625','LI_JC091_20190701','LI_JC091_20190705'
            ]


response_type = 'df_f'

filter_crit = 'zscore'
filter_thresh = 1

if ('norm' in response_type) or ('std' in response_type):
    i1 = findOccurrences(response_type,'_')[-1]
    fetch_data = response_type[i1+1:]
else:
    fetch_data = response_type
print(fetch_data)

#define paths
aggregate_root = '/n/coxfs01/cechavarria/2p-aggregate/scenes'
fig_base_dir = os.path.join(aggregate_root,'RSA_per_sessions','figures')


#get responses
area_list = ['V1','LM','LI']
animalid_list = []
sess_count = []
areaid = np.zeros((len(dset_list,)))
animalid = np.zeros((len(dset_list,)))
sessid = np.zeros((len(dset_list,)))


fig_out_dir = os.path.join(fig_base_dir,'per_session')
if not os.path.isdir(fig_out_dir):
        os.makedirs(fig_out_dir)

#for dset_idx, dset in enumerate(dset_list):
dset_idx = 0
dset = dset_list[dset_idx]

#figure out some indexes
i0 = findOccurrences(dset,'_')[0]
i1 = findOccurrences(dset,'_')[1]

area = dset[0:i0]
animal = dset[i0+1:i1]
session = dset[i1+1:]

if animal not in animalid_list:
    animalid_list.append(animal)
    sess_count.append(0)

sess_count[animalid_list.index(animal)] = sess_count[animalid_list.index(animal)]+1

areaid[dset_idx] = area_list.index(area)
animalid[dset_idx] = animalid_list.index(animal)
sessid[dset_idx] = sess_count[animalid_list.index(animal)]-1

#load data
aggregate_file_dir = os.path.join(aggregate_root,area,'files','trial_responses')
data_array_fn = '%s_%s_trial_response_array.hdf5'%(animal, session)
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


dset_response = mean_response_matrix
dset_response_active = dset_response[:,active_cell_idx]

dset_response_active_half1 = dset_response[:,active_cell_idx]
dset_response_active_half2 = dset_response[:,active_cell_idx]

#z-score and get correlation
dset_response_zscore = stats.zscore(dset_response,0)
dset_response_active_zscore = stats.zscore(dset_response_active,0)

dset_response_zscore[np.isnan(dset_response_zscore)]=0
dset_response_active_zscore[np.isnan(dset_response_active_zscore)]=0

R_dset_zscore = np.corrcoef(dset_response_zscore)
R_dset_active_zscore = np.corrcoef(dset_response_active_zscore)


plt.figure(figsize=(12, 10))
ax = sns.heatmap(R_dset_active_zscore,center = 0,annot=False, cmap = 'RdBu_r')
fig_name = '%s_%s_%s_active_zscore_%s_RSA.png'%(animal,session,area,response_type)
fig_fn = os.path.join(fig_out_dir,fig_name)

fig = ax.get_figure()
fig.savefig(fig_fn) 
   # plt.close()
   
R_cells_zscore = np.corrcoef(np.transpose(dset_response_zscore))

plt.figure(figsize=(12, 10))
ax = sns.heatmap(R_cells_zscore,center = 0,annot=False, cmap = 'RdBu_r')
fig_name = '%s_%s_%s_zscore_%s_cell_correlation.png'%(animal,session,area,response_type)
fig_fn = os.path.join(fig_out_dir,fig_name)
fig = ax.get_figure()
fig.savefig(fig_fn) 

plt.figure(figsize=(12, 10))
ax = sns.heatmap(R_cells_zscore[17:25,14:25],center = 0,annot=False, cmap = 'RdBu_r')

   

R_cells_active_zscore = np.corrcoef(np.transpose(dset_response_active_zscore))

plt.figure(figsize=(12, 10))
ax = sns.heatmap(R_cells_active_zscore,center = 0,annot=False, cmap = 'RdBu_r')
fig_name = '%s_%s_%s_active_zscore_%s_cell_correlation.png'%(animal,session,area,response_type)
fig_fn = os.path.join(fig_out_dir,fig_name)
fig = ax.get_figure()
fig.savefig(fig_fn) 


plt.figure(figsize=(12, 10))
ax = sns.heatmap(R_cells_active_zscore[14:20,14:20],center = 0,annot=False, cmap = 'RdBu_r')
fig_name = '%s_%s_%s_active_zscore_%s_cell_correlation_subset.png'%(animal,session,area,response_type)
fig_fn = os.path.join(fig_out_dir,fig_name)
fig = ax.get_figure()
fig.savefig(fig_fn) 




R_cells_zscore = np.corrcoef(np.transpose(dset_response_active_zscore))


plt.figure(figsize=(12, 10))
ax = sns.heatmap(R_cells_zscore,center = 0,annot=False, cmap = 'RdBu_r')


print(R_cells_active_zscore[14:20,14:20])


#np.where(active_cell_idx)[0][14:20]

orig_unit = np.array([17, 20, 21, 22, 23, 24])


resp_dfs = []
for unit_count, unit in enumerate(orig_unit):
    print(unit)
    mdf = pd.DataFrame({'cell_%i'%(int(unit)): dset_response[:,unit]})

    resp_dfs.append(mdf)
resp_dfs = pd.concat(resp_dfs, axis=1)



g = sns.pairplot(resp_dfs)
g.map(corrfunc)
fig_name = '%s_%s_%s_%s_cell_subset_trial_average_pariplot.png'%(animal,session,area,response_type)
fig_fn = os.path.join(fig_out_dir,fig_name)
g.savefig(fig_fn) 



#orig_unit = np.array([17, 20, 21, 22, 23, 24])
#
#unit_list = np.arange(14,20)
#
#resp_dfs = []
#for unit_count, unit in enumerate(unit_list):
#    print(unit)
#    mdf = pd.DataFrame({'cell_%i'%(int(orig_unit[unit_count])): dset_response_active[:,unit]})
#
#    resp_dfs.append(mdf)
#resp_dfs = pd.concat(resp_dfs, axis=1)
#
#
#
#g = sns.pairplot(resp_dfs)
#g.map(corrfunc)
#fig_name = '%s_%s_%s_%s_cell_subset_trial_average_pariplot.png'%(animal,session,area,response_type)
#fig_fn = os.path.join(fig_out_dir,fig_name)
#g.savefig(fig_fn) 




#get average non-self correlation value for each cell
non_self = np.ones(R_cells_zscore.shape)
di =  np.diag_indices(non_self.shape[0])
non_self[di] = np.nan

R_non_self_mean = np.nanmean(R_cells_zscore*non_self,1)
filter_crit_max = np.nanmax(filter_crit_matrix_mean,0)

g = sns.jointplot(filter_crit_max,R_non_self_mean)
g.annotate(stats.pearsonr)
fig_name = '%s_%s_%s_%s_nonself_R_vs_max_axroe.png'%(animal,session,area,response_type)
fig_fn = os.path.join(fig_out_dir,fig_name)
g.savefig(fig_fn) 


#get trial correlation for all cells
trial_dset = np.reshape(response_matrix,(1500,139))
trial_active_dset = trial_dset[:,active_cell_idx]

R_trials_cell_active = np.corrcoef(np.transpose(trial_active_dset))

plt.figure(figsize=(12, 10))
ax = sns.heatmap(R_trials_cell_active,center = 0,annot=False, cmap = 'RdBu_r')
fig_name = '%s_%s_%s_active_%s_cell_trial_correlation.png'%(animal,session,area,response_type)
fig_fn = os.path.join(fig_out_dir,fig_name)
fig = ax.get_figure()
fig.savefig(fig_fn) 

print(R_trials_cell_active[14:20,14:20])


#np.where(active_cell_idx)[0][14:20]

orig_unit = np.array([17, 20, 21, 22, 23, 24])

unit_list = np.arange(14,20)

resp_dfs = []
for unit_count, unit in enumerate(unit_list):
    mdf = pd.DataFrame({'cell_%i'%(int(orig_unit[unit_count])): trial_active_dset[:,unit]})

    resp_dfs.append(mdf)
resp_dfs = pd.concat(resp_dfs, axis=1)


g = sns.pairplot(resp_dfs)
g.map(corrfunc)
fig_name = '%s_%s_%s_%s_cell_all_trial_pariplot.png'%(animal,session,area,response_type)
fig_fn = os.path.join(fig_out_dir,fig_name)
g.savefig(fig_fn) 



sess_trial_dset = np.reshape(response_matrix[0:5,:,:],(300,139))
sess_trial_active_dset = sess_trial_dset[:,active_cell_idx]

R_sess_trials_cell_active = np.corrcoef(np.transpose(sess_trial_active_dset))

plt.figure(figsize=(12, 10))
ax = sns.heatmap(R_sess_trials_cell_active,center = 0,annot=False, cmap = 'RdBu_r')
fig_name = '%s_%s_%s_active_%s_cell_trial_correlation.png'%(animal,session,area,response_type)
#fig_fn = os.path.join(fig_out_dir,fig_name)
#fig = ax.get_figure()
#fig.savefig(fig_fn) 

print(R_sess_trials_cell_active[14:20,14:20])


#np.where(active_cell_idx)[0][14:20]

orig_unit = np.array([17, 20, 21, 22, 23, 24])

unit_list = np.arange(14,20)

resp_dfs = []
for unit_count, unit in enumerate(unit_list):
    mdf = pd.DataFrame({'cell_%i'%(int(orig_unit[unit_count])): sess_trial_active_dset[:,unit]})

    resp_dfs.append(mdf)
resp_dfs = pd.concat(resp_dfs, axis=1)


g = sns.pairplot(resp_dfs)
g.map(corrfunc)
#fig_name = '%s_%s_%s_%s_cell_all_trial_pariplot.png'%(animal,session,area,response_type)
#fig_fn = os.path.join(fig_out_dir,fig_name)
#g.savefig(fig_fn) 



#----------------------
#clusteringgg
#import scipy.cluster.hierarchy as sch
#
#X = R_trials_cell_active
#d = sch.distance.pdist(X)   # vector of ('55' choose 2) pairwise distances
#L = sch.linkage(d, method='complete')
#ind = sch.fcluster(L, 0.5*d.max(), 'distance')
#columns = [df.columns.tolist()[i] for i in list((np.argsort(ind)))]
#df = df.reindex_axis(columns, axis=1)
#
#plot_corr(df, size=18)
$---------


