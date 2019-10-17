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
import cPickle as pkl
import pandas as pd
import numpy as np
import pylab as pl
import seaborn as sns
import pprint
pp = pprint.PrettyPrinter(indent=4)
sys.path.append('/n/coxfs01/cechavarria/repos/2p-pipeline/')
from pipeline.python.paradigm import align_acquisition_events as acq
from pipeline.python.traces.utils import get_frame_info
from pipeline.python.paradigm import utils as util
from pipeline.python.utils import natural_keys, replace_root, print_elapsed_time

def findOccurrences(s, ch):
    return [i for i, letter in enumerate(s) if letter == ch]

def load_TID(run_dir, trace_id, auto=False):
    run = os.path.split(run_dir)[-1]
    trace_dir = os.path.join(run_dir, 'traces')
    tmp_tid_dir = os.path.join(trace_dir, 'tmp_tids')
    tracedict_path = os.path.join(trace_dir, 'traceids_%s.json' % run)

    print "Loading params for TRACE SET, id %s" % trace_id
    with open(tracedict_path, 'r') as f:
        tracedict = json.load(f)
    TID = tracedict[trace_id]
    pp.pprint(TID)
    return TID




parser = optparse.OptionParser()

# PATH opts:
parser.add_option('-D', '--root', action='store', dest='rootdir', default='/n/coxfs01/2p-data', 
                    help='data root dir (root project dir containing all animalids) [default: /n/coxfs01/2pdata if --slurm]')
parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')
parser.add_option('-S', '--session', action='store', dest='session', default='', help='session (format: YYYMMDD_ANIMALID')
parser.add_option('-A', '--acq', action='store', dest='acquisition', default='FOV1', help="acquisition (ex: 'FOV1_zoom3x') [default: FOV1]")
parser.add_option('-R', '--run', action='store', dest='run', default='', help="run containing tiffs to be processed (ex: gratings_phasemod_run1)")
parser.add_option('-T', '--traceid', action='store', dest='traceid', default='traces001', help="Traceid")

(options, args) = parser.parse_args(options)


s2p =True
combined = False
iti_pre = 1.0
iti_post = 1.50
stim_dur = 1.0



#% Set up paths:    
acquisition_dir = os.path.join(optsE.rootdir, optsE.animalid, optsE.session, optsE.acquisition)
if s2p:
    traceid_dir = os.path.join(acquisition_dir, run, 'traces', '%s_s2p'%(traceid))
else:
    traceid_dir = util.get_traceid_from_acquisition(acquisition_dir, run, traceid)
    TID = load_TID(run_dir, traceid)
run_dir = traceid_dir.split('/traces')[0]
trace_arrays_dir = os.path.join(traceid_dir,'files')


# Get SCAN IMAGE info for run:

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
print "N tsecs:", len(all_frames_tsecs)
framerate = scan_info['frame_rate']
volumerate = scan_info['volume_rate']
nvolumes = scan_info['nvolumes']
nfiles = scan_info['ntiffs']
pre_frames = int(iti_pre*volumerate)
post_frames = int(iti_post*volumerate)
stim_frames = int(stim_dur*volumerate)



# Load MW info to get stimulus details:
paradigm_dir = os.path.join(acquisition_dir, run, 'paradigm')
mw_fpath = [os.path.join(paradigm_dir, m) for m in os.listdir(paradigm_dir) if 'trials_' in m and m.endswith('json')][0]
with open(mw_fpath,'r') as m:
    mwinfo = json.load(m)
pre_iti_sec = round(mwinfo[mwinfo.keys()[0]]['iti_dur_ms']/1E3) 
nframes_iti_full = int(round(pre_iti_sec * volumerate))
trial_list = sorted(mwinfo.keys(), key=natural_keys)


# Create outfile:
if s2p:
    parsedtraces_fn = 'parsedtraces_s2p.hdf5'
else:
    parsedtraces_fn = 'parsedtraces_%s.hdf5' % (TID['trace_hash'])
parsedtraces_filepath = os.path.join(traceid_dir, 'files', parsedtraces_fn)
file_grp = h5py.File(parsedtraces_filepath, 'w')

##save attributes
file_grp.attrs['source_dir'] = trace_arrays_dir
file_grp.attrs['framerate'] = framerate
file_grp.attrs['volumerate'] = volumerate
file_grp.attrs['nvolumes'] = nvolumes
file_grp.attrs['nfiles'] = nfiles
file_grp.attrs['iti_pre'] = iti_pre
file_grp.attrs['iti_post'] = iti_post
file_grp.attrs['pre_frames'] = pre_frames
file_grp.attrs['post_frames'] = post_frames
file_grp.attrs['stim_frames'] = stim_frames

# to get baseline index with [0:pre_frames]
# to get stim period do [pre_frames:pre_frames+stim_frames+1]

#get parsed traces

curr_slice = 'Slice01'#hard-coding planar data for now
fid = 0
for trial_idx, trial_key in enumerate(trial_list):
# trial_idx = 25
# trial_key = 'trial00026'

    if trial_idx > 0:
        if not(fid == mwinfo[trial_key]['block_idx']):
            print(fid)
            #update fid and reload
            fid = mwinfo[trial_key]['block_idx']
            trace_file = [f for f in os.listdir(trace_arrays_dir) if 'File%03d'%(fid+1) in f and f.endswith('hdf5')][0]
            trace_fn = os.path.join(trace_arrays_dir,trace_file)

            rawfile = h5py.File(trace_fn, 'r')
            

            raw_df = rawfile[curr_slice]['traces']['raw'][:]
            sub_df = rawfile[curr_slice]['traces']['np_subtracted'][:]
            np_df = rawfile[curr_slice]['traces']['neuropil'][:]
    else:
        #load file
        trace_file = [f for f in os.listdir(trace_arrays_dir) if 'File%03d'%(fid+1) in f and f.endswith('hdf5')][0]
        trace_fn = os.path.join(trace_arrays_dir,trace_file)

        rawfile = h5py.File(trace_fn, 'r')
        if 's2p_cell_rois' in rawfile.attrs.keys():
                file_grp.attrs['s2p_cell_rois'] = rawfile.attrs['s2p_cell_rois']
        raw_df = rawfile[curr_slice]['traces']['raw'][:]
        sub_df = rawfile[curr_slice]['traces']['np_subtracted'][:]
        np_df = rawfile[curr_slice]['traces']['neuropil'][:]

        #get dimension sizes for arrays
        ntrials = len(mwinfo)
        nrois = raw_df.shape[1]
        print('ROIS: %i'%(nrois))

        #use first trial to figure out frame number
        idx_on = mwinfo[trial_key]['frame_stim_on']
        idx_off = mwinfo[trial_key]['frame_stim_off']
        ntpts = (idx_off-idx_on)+pre_frames+post_frames


        #initialize empty arrays
        parsed_traces_raw = np.empty((ntrials,ntpts,nrois))
        parsed_traces_raw[:] = np.nan
        parsed_traces_sub = np.empty((ntrials,ntpts,nrois))
        parsed_traces_sub[:] = np.nan
        parsed_traces_np = np.empty((ntrials,ntpts,nrois))
        parsed_traces_np[:] = np.nan


    idx0 = (mwinfo[trial_key]['frame_stim_on']-pre_frames)-(nvolumes*fid)
    idx1 = mwinfo[trial_key]['frame_stim_off']+post_frames-(nvolumes*fid)
    #discard time points post-stim offset if there are extra frames in this trial(stim onset alignment is what is most importnt)
    idx_diff = (idx1-idx0)-ntpts
    if idx_diff>0:
        idx1 = idx1-idx_diff
    trial_frames = idx1-idx0

    #get psth
    trial_df_raw = raw_df[idx0:idx1,:]
    trial_df_sub = sub_df[idx0:idx1,:]
    trial_df_np = np_df[idx0:idx1,:]

    parsed_traces_raw[trial_idx,0:trial_frames,:]=trial_df_raw
    parsed_traces_sub[trial_idx,0:trial_frames,:]=trial_df_sub
    parsed_traces_np[trial_idx,0:trial_frames,:]=trial_df_np

#get parsed trace time stamps
tstamps_indices = np.arange(parsed_traces_raw.shape[1])                          
volumeperiod = 1/volumerate
curr_tstamps = (tstamps_indices*volumeperiod)
curr_tstamps = curr_tstamps - iti_pre

#save arrays to file

fset = file_grp.create_dataset('/'.join([curr_slice, 'frames_tsec']), curr_tstamps.shape, curr_tstamps.dtype)
fset[...] = curr_tstamps

tset = file_grp.create_dataset('/'.join([curr_slice, 'frames_indices']), tstamps_indices.shape, tstamps_indices.dtype)
tset[...] = tstamps_indices 


raw_parsed = file_grp.create_dataset('/'.join([curr_slice, 'traces', 'raw']), parsed_traces_raw.shape, parsed_traces_raw.dtype)
raw_parsed[...] = parsed_traces_raw

sub_parsed = file_grp.create_dataset('/'.join([curr_slice, 'traces', 'np_subtracted']), parsed_traces_sub.shape, parsed_traces_sub.dtype)
sub_parsed[...] = parsed_traces_sub

np_parsed = file_grp.create_dataset('/'.join([curr_slice, 'traces', 'neuropil']), parsed_traces_np.shape, parsed_traces_np.dtype)
np_parsed[...] = parsed_traces_np

file_grp.close()

print('Done Parsing Traces!')
print('Saved all info to: %s'%(parsedtraces_filepath))

#open stimulus condition file
stimconfig_fn = 'trial_conditions.hdf5'
stimconfig_filepath = os.path.join(paradigm_dir, 'files', stimconfig_fn)
file_grp = h5py.File(stimconfig_filepath, 'w')
file_grp.attrs['ntrials'] = len(trial_list)

#get stimulus conidtion info for each trial
trial_list = sorted(mwinfo.keys(), key=natural_keys)
trial_img = np.zeros((len(trial_list),))
trial_cond = np.zeros((len(trial_list),))

for trial_idx, trial_key in enumerate(trial_list):
# trial_key = trial_list[0]
# trial_idx = 0

    stim_name = mwinfo[trial_key]['stimuli']['stimulus']
    
    if 'mag' in stim_name:
        trial_cond[trial_idx] = 1
    elif 'tex' in stim_name:
        trial_cond[trial_idx] = 2

    #get image name
    i1 = findOccurrences(stim_name,'.')[-1]
    i0 = findOccurrences(stim_name,'_')[-1]

    trial_img[trial_idx] = int(stim_name[i0+1:i1])
print np.unique(trial_img)
new_img_array = np.zeros(trial_img.shape)
for img_idx, img_id in enumerate(np.unique(trial_img)):
    new_img_array[np.where(trial_img == img_id)[0]]=img_idx
trial_img = new_img_array

config_count = 0
trial_config = np.zeros(len(trial_list),)
for img_id in np.unique(trial_img):
    for cond in np.unique(trial_cond):
        found_idx = np.intersect1d(np.where(trial_cond==cond)[0],np.where(trial_img==img_id)[0])
        trial_config[found_idx]=config_count
        config_count = config_count+1


#save arrays to file
imgset = file_grp.create_dataset('trial_img', trial_img.shape, trial_img.dtype)
imgset[...] = trial_img

configset = file_grp.create_dataset('trial_config', trial_config.shape, trial_config.dtype)
configset[...] = trial_config

condset = file_grp.create_dataset('trial_cond', trial_cond.shape, trial_cond.dtype)

condset[...] = trial_cond

file_grp.close()

print('Got Trial Conditions !')
print('Saved all info to: %s'%(stimconfig_filepath))