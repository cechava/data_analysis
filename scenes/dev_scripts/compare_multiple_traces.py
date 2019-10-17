#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 17:09:04 2019

@author: cesar
"""

import h5py

import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import matplotlib

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

import re
#sys.path.append('/n/coxfs01/cechavarria/repos/2p-pipeline/')
#from pipeline.python.paradigm import align_acquisition_events as acq
#from pipeline.python.traces.utils import get_frame_info
#from pipeline.python.paradigm import utils as util
##
#from pipeline.python.utils import natural_keys, replace_root, print_elapsed_time

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

class struct: pass

opts = struct()
opts.rootdir = '/n/coxfs01/2p-data'
opts.animalid = 'JC085'
opts.session = '20190624'
opts.acquisition = 'FOV1_zoom4p0x'
traceid = 'traces001_s2p'
run = 'scenes_run1'
s2p=True
combined = False
iti_pre = 1.0
iti_post = 1.95




#% Set up paths:    
acquisition_dir = os.path.join(opts.rootdir, opts.animalid, opts.session, opts.acquisition)

traceid_dir = os.path.join(acquisition_dir, run, 'traces',traceid)

trace_arrays_dir = os.path.join(traceid_dir,'files')
trace_fig_dir = os.path.join(traceid_dir,'figures','np_subtraced_traces')
if not os.path.exists(trace_fig_dir):
        os.makedirs(trace_fig_dir)
        
# Get SCAN IMAGE info for run:
run_dir = traceid_dir.split('/traces')[0]
run = os.path.split(run_dir)[-1]
with open(os.path.join(run_dir, '%s.json' % run), 'r') as fr:
    scan_info = json.load(fr)
all_frames_tsecs = np.array(scan_info['frame_tstamps_sec'])
nslices_full = len(all_frames_tsecs) / scan_info['nvolumes']
nslices = len(scan_info['slices'])
if scan_info['nchannels']==2:
    all_frames_tsecs = np.array(all_frames_tsecs[0::2])

#    if nslices_full > nslices:
#        # There are discard frames per volume to discount
#        subset_frame_tsecs = []
#        for slicenum in range(nslices):
#            subset_frame_tsecs.extend(frame_tsecs[slicenum::nslices_full])
#        frame_tsecs = np.array(sorted(subset_frame_tsecs))
print("N tsecs:", len(all_frames_tsecs))
framerate = scan_info['frame_rate']
volumerate = scan_info['volume_rate']
nvolumes = scan_info['nvolumes']
nfiles = scan_info['ntiffs']





# Load MW info to get stimulus details:
paradigm_dir = os.path.join(acquisition_dir, run, 'paradigm')
mw_fpath = [os.path.join(paradigm_dir, m) for m in os.listdir(paradigm_dir) if 'trials_' in m and m.endswith('json')][0]
with open(mw_fpath,'r') as m:
    mwinfo = json.load(m)
pre_iti_sec = round(mwinfo[list(mwinfo.keys())[0]]['iti_dur_ms']/1E3) 
nframes_iti_full = int(round(pre_iti_sec * volumerate))


trial_list = sorted(mwinfo.keys(), key=natural_keys)
stim_on_frames = []
stim_off_frames = []
for trial_idx, trial_key in enumerate(trial_list):
    stim_on_frames.append(mwinfo[trial_key]['frame_stim_on'])
    stim_off_frames.append(mwinfo[trial_key]['frame_stim_off'])
    
stim_on_frames = np.array(stim_on_frames)
stim_off_frames = np.array(stim_off_frames)

trace_files = [f for f in os.listdir(trace_arrays_dir) if 'File' in f and f.endswith('hdf5')]

file_idx = 0
trace_file = trace_files[0]
trace_fn = os.path.join(trace_arrays_dir,trace_file)

rawfile = h5py.File(trace_fn, 'r')
raw_df = rawfile['Slice01']['traces']['np_subtracted'][:]

stim_on_frames = []
trials_per_file = 25
for tridx in range(trials_per_file):
    tridx_all = tridx+(trials_per_file*file_idx)
    trial_key = 'trial%05d' % (tridx_all+1)
    stim_on_frames.append(mwinfo[trial_key]['frame_stim_on'])
stim_on_frames = np.array(stim_on_frames)

fig=plt.figure(figsize = (20, 5))
axes = plt.gca()
ymin, ymax = axes.get_ylim()
for f in stim_on_frames:
    plt.axvline(f, ymin=ymin, ymax = ymax, linewidth=1, color='k')
plt.plot(raw_df[:,10])