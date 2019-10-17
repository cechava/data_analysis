
import h5py
import matplotlib
matplotlib.use('Agg')
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
import scipy.stats as stats
pp = pprint.PrettyPrinter(indent=4)
sys.path.append('/n/coxfs01/cechavarria/repos/2p-pipeline/')
from pipeline.python.utils import natural_keys, replace_root, print_elapsed_time

def get_comma_separated_args(option, opt, value, parser):
  setattr(parser.values, option.dest, value.split(','))


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


def parse_trials(opts):
    #hard-coding some parameters
    s2p =True
    combined = False
    iti_pre = 1.0
    iti_post = 1.50
    stim_dur = 1.0


    for run_count,run in enumerate(opts.run_list):
        print(run)

        #% Set up paths:    
        acquisition_dir = os.path.join(opts.rootdir, opts.animalid, opts.session, opts.acquisition)
        if s2p:
            traceid_dir = os.path.join(acquisition_dir, run, 'traces', '%s_s2p'%(opts.traceid))
        else:
            traceid_dir = util.get_traceid_from_acquisition(acquisition_dir, run, opts.traceid)
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


            if trial_idx > 0:
                if not(fid == mwinfo[trial_key]['block_idx']):
                    print('fid: %i'% (fid))
                    #update fid and reload
                    fid = mwinfo[trial_key]['block_idx']
                    trace_file = [f for f in os.listdir(trace_arrays_dir) if 'File%03d'%(fid+1) in f and f.endswith('hdf5')][0]
                    trace_fn = os.path.join(trace_arrays_dir,trace_file)

                    rawfile = h5py.File(trace_fn, 'r')
                    motion_offset = np.array(rawfile.attrs['motion_offset'])


                    raw_df = rawfile[curr_slice]['traces']['raw'][:]
                    sub_df = rawfile[curr_slice]['traces']['np_subtracted'][:]
                    np_df = rawfile[curr_slice]['traces']['neuropil'][:]
                    print(run_count,fid)
                    mean_pix_fid[fid,:] = np.squeeze(np.mean(np.squeeze(np.mean(sub_df,0)),0))
            else:
                #load file
                trace_file = [f for f in os.listdir(trace_arrays_dir) if 'File%03d'%(fid+1) in f and f.endswith('hdf5')][0]
                trace_fn = os.path.join(trace_arrays_dir,trace_file)

                rawfile = h5py.File(trace_fn, 'r')
                
                motion_offset = np.array(rawfile.attrs['motion_offset'])
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
                if run_count == 0:
                    idx_on = mwinfo[trial_key]['frame_stim_on']
                    idx_off = mwinfo[trial_key]['frame_stim_off']
                    ntpts = (idx_off-idx_on)+pre_frames+post_frames
                    print('ntpts: %i', ntpts)

                    mean_pix_fid = np.empty((nfiles,nrois))


                #initialize empty arrays
                trial_fid = np.empty((ntrials,))
                parsed_traces_raw = np.empty((ntrials,ntpts,nrois))
                parsed_traces_raw[:] = np.nan
                parsed_traces_sub = np.empty((ntrials,ntpts,nrois))
                parsed_traces_sub[:] = np.nan
                parsed_traces_np = np.empty((ntrials,ntpts,nrois))
                parsed_traces_np[:] = np.nan
                parsed_motion = np.empty((ntrials,ntpts))
                parsed_motion[:] = np.nan


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
            parsed_motion[trial_idx,0:trial_frames] = motion_offset[idx0:idx1]
            trial_fid[trial_idx] = fid

        #get parsed trace time stamps
        tstamps_indices = np.arange(parsed_traces_raw.shape[1])                          
        volumeperiod = 1/volumerate
        curr_tstamps = (tstamps_indices*volumeperiod)
        curr_tstamps = curr_tstamps - iti_pre

        #save arrays to file
        motion = file_grp.create_dataset('trial_motion', parsed_motion.shape, parsed_motion.dtype)
        motion[...] = parsed_motion

        fset = file_grp.create_dataset('/'.join([curr_slice, 'frames_tsec']), curr_tstamps.shape, curr_tstamps.dtype)
        fset[...] = curr_tstamps

        tset = file_grp.create_dataset('/'.join([curr_slice, 'frames_indices']), tstamps_indices.shape, tstamps_indices.dtype)
        tset[...] = tstamps_indices 

        fidset = file_grp.create_dataset('/'.join([curr_slice, 'trial_fid']), trial_fid.shape, trial_fid.dtype)
        fidset[...] = trial_fid 

        pixset = file_grp.create_dataset('/'.join([curr_slice, 'mean_pix_fid']), mean_pix_fid.shape, mean_pix_fid.dtype)
        pixset[...] = mean_pix_fid 



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
                trial_cond[trial_idx] = 2
            elif 'tex' in stim_name:
                trial_cond[trial_idx] = 1

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

def combine_trials(opts):

    s2p = True

    #Set up paths
    acquisition_dir = os.path.join(opts.rootdir, opts.animalid, opts.session, opts.acquisition)

    if s2p:
        combined_traceid_file_dir = os.path.join(acquisition_dir,opts.combined_run,'traces','%s_s2p'%(opts.traceid),'files')
    else:
        combined_traceid_file_dir = os.path.join(acquisition_dir,opts.combined_run,'traces',opts.traceid,'files')
    if not os.path.exists(combined_traceid_file_dir):
        os.makedirs(combined_traceid_file_dir)
        
    combined_paradigm_dir = os.path.join(acquisition_dir,opts.combined_run,'paradigm','files')
    if not os.path.exists(combined_paradigm_dir):
        os.makedirs(combined_paradigm_dir)

    for run_idx,run in enumerate(opts.run_list):
        print(run)
    # run_idx = 0
    # run = 'scenes_run1'


        #% Set up paths:   
        if s2p:
            traceid_dir = os.path.join(acquisition_dir, run, 'traces', '%s_s2p'%(opts.traceid))
        else:
            traceid_dir = util.get_traceid_from_acquisition(acquisition_dir, run, opts.traceid)
        run_dir = traceid_dir.split('/traces')[0]
        trace_arrays_dir = os.path.join(traceid_dir,'files')
        paradigm_dir = os.path.join(acquisition_dir, run, 'paradigm')


        #read file
        parsedtraces_filepath = glob.glob(os.path.join(traceid_dir, 'files','parsedtraces*'))[0]
        file_grp = h5py.File(parsedtraces_filepath, 'r')



        curr_slice = 'Slice01'#hard-code planar data for now
        
        trial_fid = np.array(file_grp['Slice01']['trial_fid'])
        trial_run = np.ones(trial_fid.shape)*(run_idx+1)
        mean_pix_fid = np.array(file_grp['Slice01']['mean_pix_fid'])
        #get motion
        motion_run = np.array(file_grp['trial_motion'])
        #get raw pixel value arrays
        pix_raw_run = np.array(file_grp[curr_slice]['traces']['raw'])
        pix_cell_run = np.array(file_grp[curr_slice]['traces']['np_subtracted'])
        pix_np_run = np.array(file_grp[curr_slice]['traces']['neuropil'])

        print(pix_raw_run.shape)



        #save attributes and stack traces
        if run_idx ==0:
             # Create outfile:
            combined_array_fn = 'parsedtraces.hdf5'
            combined_array_filepath = os.path.join(combined_traceid_file_dir, combined_array_fn)
            combined_grp = h5py.File(combined_array_filepath, 'w')
            
            combined_grp.attrs['source_dir'] = file_grp.attrs['source_dir']
            combined_grp.attrs['framerate'] = file_grp.attrs['framerate']
            combined_grp.attrs['volumerate'] = file_grp.attrs['volumerate']
            combined_grp.attrs['nvolumes'] = file_grp.attrs['nvolumes']
            combined_grp.attrs['nfiles'] = file_grp.attrs['nfiles']
            combined_grp.attrs['iti_pre'] = file_grp.attrs['iti_pre']
            combined_grp.attrs['iti_post'] = file_grp.attrs['iti_post']
            combined_grp.attrs['pre_frames'] = file_grp.attrs['pre_frames']
            combined_grp.attrs['post_frames'] = file_grp.attrs['post_frames'] 
            combined_grp.attrs['stim_frames'] = file_grp.attrs['stim_frames']
            combined_grp.attrs['nrois'] = pix_cell_run.shape[2]
            curr_tstamps = np.array(file_grp['Slice01']['frames_tsec'])
            tstamps_indices = np.array(file_grp['Slice01']['frames_tsec'])
            
            
            
            if 's2p_cell_rois' in file_grp.attrs.keys():
                combined_grp.attrs['s2p_cell_rois'] = file_grp.attrs['s2p_cell_rois']

            pix_raw_combo = pix_raw_run
            pix_cell_combo = pix_cell_run
            pix_np_combo = pix_np_run
            trial_fid_combo = trial_fid
            trial_run_combo = trial_run
            mean_pix_fid_combo = mean_pix_fid
            motion_run_combo = motion_run
        else:
            #for now, lop off timepoints if doesn't match what we already have, if volumerate fast enough. this is negligible
            if pix_raw_run.shape[1]>pix_raw_combo.shape[1]:
                extra_frames = pix_raw_run.shape[1]-pix_raw_combo.shape[1]
                pix_raw_run = pix_raw_run[:,:-extra_frames,:]
                pix_cell_run = pix_cell_run[:,:-extra_frames,:]
                pix_np_run = pix_np_run[:,:-extra_frames,:]
                motion_run = motion_run[:,:-extra_frames]
            
            pix_raw_combo = np.vstack((pix_raw_combo,pix_raw_run))
            pix_cell_combo = np.vstack((pix_cell_combo,pix_cell_run))
            pix_np_combo = np.vstack((pix_np_combo,pix_np_run))

            motion_run_combo = np.vstack((motion_run_combo,motion_run))
            
            trial_fid_combo = np.hstack((trial_fid_combo,trial_fid))
            trial_run_combo = np.hstack((trial_run_combo,trial_run))
            mean_pix_fid_combo = np.vstack((mean_pix_fid_combo,mean_pix_fid))
            
        file_grp.close()

    #save combined traces to file
    #save arrays to file

    motion = combined_grp.create_dataset('motion', motion_run_combo.shape, motion_run_combo.dtype)
    motion[...] = motion_run_combo

    fset = combined_grp.create_dataset('/'.join([curr_slice, 'frames_tsec']), curr_tstamps.shape, curr_tstamps.dtype)
    fset[...] = curr_tstamps

    tset = combined_grp.create_dataset('/'.join([curr_slice, 'frames_indices']), tstamps_indices.shape, tstamps_indices.dtype)
    tset[...] = tstamps_indices 

    fidset = combined_grp.create_dataset('/'.join([curr_slice, 'trial_fid']), trial_fid_combo.shape, trial_fid_combo.dtype)
    fidset[...] = trial_fid_combo

    runset = combined_grp.create_dataset('/'.join([curr_slice, 'trial_run']), trial_run_combo.shape, trial_run_combo.dtype)
    runset[...] = trial_run_combo

    pixset = combined_grp.create_dataset('/'.join([curr_slice, 'mean_pix']), mean_pix_fid_combo.shape, mean_pix_fid_combo.dtype)
    pixset[...] = mean_pix_fid_combo

    raw_combined = combined_grp.create_dataset('/'.join([curr_slice, 'traces', 'raw']), pix_raw_combo.shape, pix_raw_combo.dtype)
    raw_combined[...] = pix_raw_combo

    cell_combined = combined_grp.create_dataset('/'.join([curr_slice, 'traces', 'np_subtracted']), pix_cell_combo.shape, pix_cell_combo.dtype)
    cell_combined[...] = pix_cell_combo

    np_combined = combined_grp.create_dataset('/'.join([curr_slice, 'traces', 'neuropil']), pix_np_combo.shape, pix_np_combo.dtype)
    np_combined[...] = pix_np_combo

    combined_grp.close()

    print('Done Combining Traces!')
    print('Saved all info to: %s'%(combined_array_filepath))

    #for run in runlist
    for run_idx,run in enumerate(opts.run_list):
        print(run)
    # run = 'scenes_run1'
    # run_idx = 0

        #open stimulus condition file
        stimconfig_fn = 'trial_conditions.hdf5'
        paradigm_dir = os.path.join(acquisition_dir, run, 'paradigm')
        stimconfig_filepath = os.path.join(paradigm_dir, 'files', stimconfig_fn)
        run_config_grp = h5py.File(stimconfig_filepath, 'r')

        config_run = np.array(run_config_grp['trial_config'])
        cond_run = np.array(run_config_grp['trial_cond'])
        img_run = np.array(run_config_grp['trial_img'])

        if run_idx ==0:
            #open output file
            combo_config_filepath = os.path.join(combined_paradigm_dir, stimconfig_fn)
            combo_config_grp = h5py.File(combo_config_filepath, 'w')

            combo_config_grp.attrs['ntrials'] = run_config_grp.attrs['ntrials']

            config_combo = config_run
            cond_combo = cond_run
            img_combo = img_run

        else:
            config_combo = np.hstack((config_combo,config_run))
            cond_combo = np.hstack((cond_combo,cond_run))
            img_combo = np.hstack((img_combo,img_run))

        run_config_grp.close()

    print('Combined Trial Conditions !')



    #save arrays to file
    imgcombo = combo_config_grp.create_dataset('trial_img', img_combo.shape, img_combo.dtype)
    imgcombo[...] = img_combo

    configcombo = combo_config_grp.create_dataset('trial_config', config_combo.shape, config_combo.dtype)
    configcombo[...] = config_combo

    condcombo = combo_config_grp.create_dataset('trial_cond', cond_combo.shape, cond_combo.dtype)
    condcombo[...] = cond_combo

    combo_config_grp.close()
    print('Saved all info to: %s'%(combo_config_filepath))


def evaluate_trials(opts):
    traceid = '%s_s2p'%(opts.traceid)

    #% Set up paths:    
    acquisition_dir = os.path.join(opts.rootdir, opts.animalid, opts.session, opts.acquisition)

    traceid_dir = os.path.join(acquisition_dir, opts.combined_run,'traces',traceid)

    run_dir = traceid_dir.split('/traces')[0]
    trace_arrays_dir = os.path.join(traceid_dir,'files')
    paradigm_dir = os.path.join(acquisition_dir, opts.combined_run, 'paradigm')

    data_array_dir = os.path.join(traceid_dir, 'data_arrays')
    if not os.path.exists(data_array_dir):
        os.makedirs(data_array_dir)

    #read file
    parsedtraces_filepath = glob.glob(os.path.join(traceid_dir, 'files','parsedtraces*'))[0]
    file_grp = h5py.File(parsedtraces_filepath, 'r')

    #get motion info
    motion = np.array(file_grp['motion'])


    pre_frames = file_grp.attrs['pre_frames']
    post_frames = file_grp.attrs['post_frames']
    stim_frames = file_grp.attrs['stim_frames']
    #to get baseline index with [0:pre_frames]
    #to get stim period do [pre_frames:pre_frames+stim_frames+1]

    curr_slice = 'Slice01'#hard-code planar data for now
    #get raw pixel value arrays
    pix_raw_array = np.array(file_grp[curr_slice]['traces']['raw'])
    pix_cell_array = np.array(file_grp[curr_slice]['traces']['np_subtracted'])
    pix_np_array = np.array(file_grp[curr_slice]['traces']['neuropil'])

    mean_f_raw = np.nanmean(np.nanmean(pix_raw_array,0),0)
    mean_f_cell = np.nanmean(np.nanmean(pix_cell_array,0),0)
    mean_f_np = np.nanmean(np.nanmean(pix_np_array,0),0)


    #open file for storage
    # Create outfile:
    data_array_fn = 'processed_config_traces.hdf5'
    data_array_filepath = os.path.join(traceid_dir, 'data_arrays', data_array_fn)
    data_grp = h5py.File(data_array_filepath, 'w')

    #save attributes
    data_grp.attrs['source_dir'] = file_grp.attrs['source_dir']
    data_grp.attrs['framerate'] = file_grp.attrs['framerate']
    data_grp.attrs['volumerate'] = file_grp.attrs['volumerate']
    data_grp.attrs['nvolumes'] = file_grp.attrs['nvolumes']
    data_grp.attrs['nfiles'] = file_grp.attrs['nfiles']
    data_grp.attrs['iti_pre'] = file_grp.attrs['iti_pre']
    data_grp.attrs['iti_post'] = file_grp.attrs['iti_post']
    data_grp.attrs['pre_frames'] = file_grp.attrs['pre_frames']
    data_grp.attrs['post_frames'] = file_grp.attrs['post_frames'] 
    data_grp.attrs['stim_frames'] = file_grp.attrs['stim_frames']
    data_grp.attrs['nrois'] = pix_cell_array.shape[2]
    data_grp.attrs['frames_tsec'] = file_grp['Slice01']['frames_tsec']
    data_grp.attrs['frames_tsec'] = file_grp['Slice01']['frames_tsec']
    data_grp.attrs['trial_fid'] = file_grp['Slice01']['trial_fid']
    data_grp.attrs['trial_run'] = file_grp['Slice01']['trial_run']
    if 's2p_cell_rois' in file_grp.attrs.keys():
        data_grp.attrs['s2p_cell_rois'] = file_grp.attrs['s2p_cell_rois']




    file_grp.close()

    ntrials,ntpts,nrois = pix_cell_array.shape
    print('ROIs:%i'%(nrois))
    print('Trials:%i'%(ntrials))

    f_cell_array = np.zeros(pix_cell_array.shape)
    f_raw_array = np.zeros(pix_raw_array.shape)
    f_np_array = np.zeros(pix_np_array.shape)

    df_cell_array = np.zeros(pix_cell_array.shape)
    df_raw_array = np.zeros(pix_raw_array.shape)
    df_np_array = np.zeros(pix_np_array.shape)

    df_f_cell_array = np.zeros(pix_cell_array.shape)
    df_f_raw_array = np.zeros(pix_raw_array.shape)
    df_f_np_array = np.zeros(pix_np_array.shape)

    zscore_cell_array = np.zeros(pix_cell_array.shape)
    zscore_raw_array = np.zeros(pix_raw_array.shape)
    zscore_np_array = np.zeros(pix_np_array.shape)


    for tidx in range(ntrials):
    #tidx = 0
    #tidx = 100
        for ridx in range(nrois):
    #ridx = 3

            #get trial timecourse
            pix_raw = np.squeeze(pix_raw_array[tidx,:,ridx])
            pix_cell = np.squeeze(pix_cell_array[tidx,:,ridx].squeeze())
            pix_np = np.squeeze(pix_np_array[tidx,:,ridx].squeeze())


            #get baseline
            base_raw = pix_raw[0:pre_frames]
            stim_raw = pix_raw[pre_frames:pre_frames+stim_frames+1]
            base_cell = pix_cell[0:pre_frames]
            stim_cell = pix_cell[pre_frames:pre_frames+stim_frames+1]
            base_np = pix_np[0:pre_frames]
            stim_np = pix_np[pre_frames:pre_frames+stim_frames+1]
            
            #get raw fluoresence value
            f_raw = pix_raw
            f_cell = pix_cell
            f_np = pix_np

            #calculate df
            df_raw = pix_raw - np.mean(base_raw)
            df_cell = pix_cell - np.mean(base_cell)
            df_np = pix_np - np.mean(base_np)


            #calculate df/f
            df_f_raw = df_raw/mean_f_raw[ridx]
            df_f_cell = df_cell/mean_f_cell[ridx]
            df_f_np = df_np/mean_f_np[ridx]

            #calculate z-score
            zscore_raw = (pix_raw - np.mean(base_raw))/np.std(base_raw)
            zscore_cell = (pix_cell - np.mean(base_cell))/np.std(base_cell)
            zscore_np = (pix_np - np.mean(base_np))/np.std(base_np)

            #store in array
            f_raw_array[tidx,:,ridx] = f_raw
            df_raw_array[tidx,:,ridx] = df_raw
            df_f_raw_array[tidx,:,ridx] = df_f_raw
            zscore_raw_array[tidx,:,ridx] = zscore_raw

            f_cell_array[tidx,:,ridx] = f_cell
            df_cell_array[tidx,:,ridx] = df_cell
            df_f_cell_array[tidx,:,ridx] = df_f_cell
            zscore_cell_array[tidx,:,ridx] = zscore_cell

            f_np_array[tidx,:,ridx] = f_np
            df_np_array[tidx,:,ridx] = df_np
            df_f_np_array[tidx,:,ridx] = df_f_np
            zscore_np_array[tidx,:,ridx] = zscore_np

    #load trial info
    stimconfig_fn = 'trial_conditions.hdf5'
    stimconfig_filepath = os.path.join(paradigm_dir, 'files', stimconfig_fn)
    config_grp = h5py.File(stimconfig_filepath, 'r')

    ntrials = config_grp.attrs['ntrials']
    trial_config = np.array(config_grp['trial_config'])
    trial_cond = np.array(config_grp['trial_cond'])
    trial_img = np.array(config_grp['trial_img'])

    config_grp.close()

    for cfg_idx in np.unique(trial_config):
    #cfg_idx = 11

        cfg_key = 'config%03d'%(cfg_idx)
        print(cfg_key)
        tidx = np.where(trial_config == cfg_idx)[0]
        print(len(tidx))

        #print(trial_img[tidx[0]],trial_cond[tidx[0]])

        #get relevant traces
        f_raw_config_trace = f_raw_array[tidx,:,:]
        f_cell_config_trace = f_cell_array[tidx,:,:]
        f_np_config_trace = f_np_array[tidx,:,:]
        
        df_raw_config_trace = df_raw_array[tidx,:,:]
        df_raw_config_trace_mean = np.squeeze(np.nanmean(df_raw_config_trace,0))
        df_raw_config_trace_se = stats.sem(df_raw_config_trace,0)

        zscore_raw_config_trace = zscore_raw_array[tidx,:,:]
        zscore_raw_config_trace_mean = np.squeeze(np.nanmean(zscore_raw_config_trace,0))
        zscore_raw_config_trace_se = stats.sem(zscore_raw_config_trace,0)

        df_f_raw_config_trace = df_f_raw_array[tidx,:,:]
        df_f_raw_config_trace_mean = np.squeeze(np.nanmean(df_f_raw_config_trace,0))
        df_f_raw_config_trace_se = stats.sem(df_f_raw_config_trace,0)

        df_cell_config_trace = df_cell_array[tidx,:,:]
        df_cell_config_trace_mean = np.squeeze(np.nanmean(df_cell_config_trace,0))
        df_cell_config_trace_se = stats.sem(df_cell_config_trace,0)

        zscore_cell_config_trace = zscore_cell_array[tidx,:,:]
        zscore_cell_config_trace_mean = np.squeeze(np.nanmean(zscore_cell_config_trace,0))
        zscore_cell_config_trace_se = stats.sem(zscore_cell_config_trace,0)

        df_f_cell_config_trace = df_f_cell_array[tidx,:,:]
        df_f_cell_config_trace_mean = np.squeeze(np.nanmean(df_f_cell_config_trace,0))
        df_f_cell_config_trace_se = stats.sem(df_f_cell_config_trace,0)

        df_np_config_trace = df_np_array[tidx,:,:]
        df_np_config_trace_mean = np.squeeze(np.nanmean(df_np_config_trace,0))
        df_np_config_trace_se = stats.sem(df_np_config_trace,0)

        zscore_np_config_trace = zscore_np_array[tidx,:,:]
        zscore_np_config_trace_mean = np.squeeze(np.nanmean(zscore_np_config_trace,0))
        zscore_np_config_trace_se = stats.sem(zscore_np_config_trace,0)

        df_f_np_config_trace = df_f_np_array[tidx,:,:]
        df_f_np_config_trace_mean = np.squeeze(np.nanmean(df_f_np_config_trace,0))
        df_f_np_config_trace_se = stats.sem(df_f_np_config_trace,0)

        motion_config_trace = motion[tidx,:]



        #save motion
        motion_set = data_grp.create_dataset('/'.join([cfg_key, 'motion']), motion_config_trace.shape, motion_config_trace.dtype)
        motion_set[...] = motion_config_trace

        #save config specifics
        img_dset = data_grp.create_dataset('/'.join([curr_slice, cfg_key, 'img']),(1,),dtype = int)
        img_dset[...] = trial_img[tidx[0]]

        cond_dset = data_grp.create_dataset('/'.join([curr_slice, cfg_key, 'scene_cond']),(1,),dtype = int)
        cond_dset[...] = trial_cond[tidx[0]]


        #save config responses
        raw_f_trace_dset = data_grp.create_dataset('/'.join([curr_slice, cfg_key, 'f', 'trace','raw']), f_raw_config_trace.shape, f_raw_config_trace.dtype)
        raw_f_trace_dset[...] = f_raw_config_trace
        
        cell_f_trace_dset = data_grp.create_dataset('/'.join([curr_slice, cfg_key, 'f', 'trace','np_subtracted']), f_cell_config_trace.shape, f_cell_config_trace.dtype)
        cell_f_trace_dset[...] = f_cell_config_trace

        np_f_trace_dset = data_grp.create_dataset('/'.join([curr_slice, cfg_key, 'f', 'trace','np']), f_np_config_trace.shape, f_np_config_trace.dtype)
        np_f_trace_dset[...] = f_np_config_trace


        raw_df_trace_dset = data_grp.create_dataset('/'.join([curr_slice, cfg_key, 'df', 'trace','raw']), df_raw_config_trace.shape, df_raw_config_trace.dtype)
        raw_df_trace_dset[...] = df_raw_config_trace

        raw_df_mean_dset = data_grp.create_dataset('/'.join([curr_slice, cfg_key, 'df', 'trace_mean','raw']), df_raw_config_trace_mean.shape, df_raw_config_trace_mean.dtype)
        raw_df_mean_dset[...] = df_raw_config_trace_mean

        raw_df_se_dset = data_grp.create_dataset('/'.join([curr_slice, cfg_key, 'df', 'trace_se','raw']), df_raw_config_trace_se.shape, df_raw_config_trace_se.dtype)
        raw_df_se_dset[...] = df_raw_config_trace_se

        cell_df_trace_dset = data_grp.create_dataset('/'.join([curr_slice, cfg_key, 'df', 'trace','np_subtracted']), df_cell_config_trace.shape, df_cell_config_trace.dtype)
        cell_df_trace_dset[...] = df_cell_config_trace

        cell_df_mean_dset = data_grp.create_dataset('/'.join([curr_slice, cfg_key, 'df', 'trace_mean','np_subtracted']), df_cell_config_trace_mean.shape, df_cell_config_trace_mean.dtype)
        cell_df_mean_dset[...] = df_cell_config_trace_mean

        cell_df_se_dset = data_grp.create_dataset('/'.join([curr_slice, cfg_key, 'df', 'trace_se','np_subtracted']), df_cell_config_trace_se.shape, df_cell_config_trace_se.dtype)
        cell_df_se_dset[...] = df_cell_config_trace_se

        np_df_trace_dset = data_grp.create_dataset('/'.join([curr_slice, cfg_key, 'df', 'trace','neuropil']), df_np_config_trace.shape, df_np_config_trace.dtype)
        np_df_trace_dset[...] = df_np_config_trace

        np_df_mean_dset = data_grp.create_dataset('/'.join([curr_slice, cfg_key, 'df', 'trace_mean','neuropil']), df_np_config_trace_mean.shape, df_np_config_trace_mean.dtype)
        np_df_mean_dset[...] = df_np_config_trace_mean

        np_df_se_dset = data_grp.create_dataset('/'.join([curr_slice, cfg_key, 'df', 'trace_se','neuropil']), df_np_config_trace_se.shape, df_np_config_trace_se.dtype)
        np_df_se_dset[...] = df_np_config_trace_se
      


        raw_df_f_trace_dset = data_grp.create_dataset('/'.join([curr_slice, cfg_key, 'df_f', 'trace','raw']), df_f_raw_config_trace.shape, df_f_raw_config_trace.dtype)
        raw_df_f_trace_dset[...] = df_f_raw_config_trace

        raw_df_f_mean_dset = data_grp.create_dataset('/'.join([curr_slice, cfg_key, 'df_f', 'trace_mean','raw']), df_f_raw_config_trace_mean.shape, df_f_raw_config_trace_mean.dtype)
        raw_df_f_mean_dset[...] = df_f_raw_config_trace_mean

        raw_df_f_se_dset = data_grp.create_dataset('/'.join([curr_slice, cfg_key, 'df_f', 'trace_se','raw']), df_f_raw_config_trace_se.shape, df_f_raw_config_trace_se.dtype)
        raw_df_f_se_dset[...] = df_f_raw_config_trace_se

        cell_df_f_trace_dset = data_grp.create_dataset('/'.join([curr_slice, cfg_key, 'df_f', 'trace','np_subtracted']), df_f_cell_config_trace.shape, df_f_cell_config_trace.dtype)
        cell_df_f_trace_dset[...] = df_f_cell_config_trace

        cell_df_f_mean_dset = data_grp.create_dataset('/'.join([curr_slice, cfg_key, 'df_f', 'trace_mean','np_subtracted']), df_f_cell_config_trace_mean.shape, df_f_cell_config_trace_mean.dtype)
        cell_df_f_mean_dset[...] = df_f_cell_config_trace_mean

        cell_df_f_se_dset = data_grp.create_dataset('/'.join([curr_slice, cfg_key, 'df_f', 'trace_se','np_subtracted']), df_f_cell_config_trace_se.shape, df_f_cell_config_trace_se.dtype)
        cell_df_f_se_dset[...] = df_f_cell_config_trace_se

        np_df_f_trace_dset = data_grp.create_dataset('/'.join([curr_slice, cfg_key, 'df_f', 'trace','neuropil']), df_f_np_config_trace.shape, df_f_np_config_trace.dtype)
        np_df_f_trace_dset[...] = df_f_np_config_trace

        np_df_f_mean_dset = data_grp.create_dataset('/'.join([curr_slice, cfg_key, 'df_f', 'trace_mean','neuropil']), df_f_np_config_trace_mean.shape, df_f_np_config_trace_mean.dtype)
        np_df_f_mean_dset[...] = df_f_np_config_trace_mean

        np_df_f_se_dset = data_grp.create_dataset('/'.join([curr_slice, cfg_key, 'df_f', 'trace_se','neuropil']), df_f_np_config_trace_se.shape, df_f_np_config_trace_se.dtype)
        np_df_f_se_dset[...] = df_f_np_config_trace_se


        raw_zscore_trace_dset = data_grp.create_dataset('/'.join([curr_slice, cfg_key, 'zscore', 'trace','raw']), zscore_raw_config_trace.shape, zscore_raw_config_trace.dtype)
        raw_zscore_trace_dset[...] = zscore_raw_config_trace

        raw_zscore_mean_dset = data_grp.create_dataset('/'.join([curr_slice, cfg_key, 'zscore', 'trace_mean','raw']), zscore_raw_config_trace_mean.shape, zscore_raw_config_trace_mean.dtype)
        raw_zscore_mean_dset[...] = zscore_raw_config_trace_mean

        raw_zscore_se_dset = data_grp.create_dataset('/'.join([curr_slice, cfg_key, 'zscore', 'trace_se','raw']), zscore_raw_config_trace_se.shape, zscore_raw_config_trace_se.dtype)
        raw_zscore_se_dset[...] = zscore_raw_config_trace_se

        cell_zscore_trace_dset = data_grp.create_dataset('/'.join([curr_slice, cfg_key, 'zscore', 'trace','np_subtracted']), zscore_cell_config_trace.shape, zscore_cell_config_trace.dtype)
        cell_zscore_trace_dset[...] = zscore_cell_config_trace

        cell_zscore_mean_dset = data_grp.create_dataset('/'.join([curr_slice, cfg_key, 'zscore', 'trace_mean','np_subtracted']), zscore_cell_config_trace_mean.shape, zscore_cell_config_trace_mean.dtype)
        cell_zscore_mean_dset[...] = zscore_cell_config_trace_mean

        cell_zscore_se_dset = data_grp.create_dataset('/'.join([curr_slice, cfg_key, 'zscore', 'trace_se','np_subtracted']), zscore_cell_config_trace_se.shape, zscore_cell_config_trace_se.dtype)
        cell_zscore_se_dset[...] = zscore_cell_config_trace_se

        np_zscore_trace_dset = data_grp.create_dataset('/'.join([curr_slice, cfg_key, 'zscore', 'trace','neuropil']), zscore_np_config_trace.shape, zscore_np_config_trace.dtype)
        np_zscore_trace_dset[...] = zscore_np_config_trace

        np_zscore_mean_dset = data_grp.create_dataset('/'.join([curr_slice, cfg_key, 'zscore', 'trace_mean','neuropil']), zscore_np_config_trace_mean.shape, zscore_np_config_trace_mean.dtype)
        np_zscore_mean_dset[...] = zscore_np_config_trace_mean

        np_zscore_se_dset = data_grp.create_dataset('/'.join([curr_slice, cfg_key, 'zscore', 'trace_se','neuropil']), zscore_np_config_trace_se.shape, zscore_np_config_trace_se.dtype)
        np_zscore_se_dset[...] = zscore_np_config_trace_se

    data_grp.close()

def get_trial_responses(opts):

    traceid = '%s_s2p'%(opts.traceid)
    motion_thresh = int(opts.motion_thresh)
    #% Set up paths:    
    acquisition_dir = os.path.join(opts.rootdir, opts.animalid, opts.session, opts.acquisition)

    traceid_dir = os.path.join(acquisition_dir, opts.combined_run,'traces', traceid)

    run_dir = traceid_dir.split('/traces')[0]
    trace_arrays_dir = os.path.join(traceid_dir,'files')
    paradigm_dir = os.path.join(acquisition_dir, opts.combined_run, 'paradigm')

    #output directory
    responses_dir = os.path.join(acquisition_dir, opts.combined_run,'responses', traceid)
    data_array_dir = os.path.join(responses_dir, 'data_arrays')
        


    #open file to read
    data_array_fn = 'processed_config_traces.hdf5'
    data_array_filepath = os.path.join(traceid_dir, 'data_arrays', data_array_fn)
    print(data_array_filepath)
    data_grp = h5py.File(data_array_filepath, 'r')

    frames_tsec = data_grp.attrs['frames_tsec']
    nrois = data_grp.attrs['nrois']
    print('ROIs:%i'%(nrois))


    if 's2p_cell_rois' in data_grp.attrs.keys():
        cell_rois = data_grp.attrs['s2p_cell_rois']
    else:
        cell_rois = np.arange(nrois)

    curr_slice = 'Slice01'#hard,coding for now
    stim_period0 = data_grp.attrs['pre_frames']
    stim_period1 = data_grp.attrs['pre_frames']+data_grp.attrs['stim_frames']+1

    trial_fid = np.array(data_grp.attrs['trial_fid'])
    trial_run = np.array(data_grp.attrs['trial_run'])



    config_img = np.zeros((len(data_grp[curr_slice].keys())))
    config_cond = np.zeros((len(data_grp[curr_slice].keys())))
    for cfg_count,cfg_key in enumerate(data_grp[curr_slice].keys()):
        config_img[cfg_count] = np.array(data_grp['/'.join([curr_slice,cfg_key,'img'])])[0]+1
        config_cond[cfg_count] = np.array(data_grp['/'.join([curr_slice,cfg_key,'scene_cond'])])[0]



    trial_filter0 = []
    for cfg_count,cfg_key in enumerate(data_grp[curr_slice].keys()):
        motion_trace = np.array(data_grp['/'.join([cfg_key,'motion'])])
        config_trial_filter = np.sum(np.abs(motion_trace)>motion_thresh,1)<5#keep if less than 5 frames passed threshold
        trial_filter0.append(config_trial_filter)
    trial_filter0 = np.array(trial_filter0)  
    nconfigs,ntrials = trial_filter0.shape

    #count number of non-excluded trials
    good_trial_count = np.sum(trial_filter0,1)
    #minimum is how many trials we will mach
    trials_to_keep = np.nanmin(good_trial_count)

    #create a final trial_filter
    trial_filter = np.zeros(trial_filter0.shape)
    trial_filter[good_trial_count==trials_to_keep,:] = trial_filter0[good_trial_count==trials_to_keep,:]
    trial_filter[good_trial_count>trials_to_keep,0:trials_to_keep] = 1
    trial_filter = trial_filter.astype('bool')

    print('**Matching all configs to have %i trials**'%(trials_to_keep))
        

    # cfg_key = 'config006'
    # cfg_count = 0 
    # print(np.array(data_grp[curr_slice][cfg_key]['img']))
    # print(np.array(data_grp[curr_slice][cfg_key]['scene_cond']))

    for cfg_count,cfg_key in enumerate(data_grp[curr_slice].keys()):
      #  print(cfg_key)
        motion_trace = data_grp['/'.join([cfg_key,'motion'])]
        
        trial_traces_f = np.array(data_grp['/'.join([curr_slice,cfg_key,'f', 'trace','np_subtracted'])])[trial_filter[cfg_count,:],:,:]
        trial_traces_df = np.array(data_grp['/'.join([curr_slice,cfg_key,'df', 'trace','np_subtracted'])])[trial_filter[cfg_count,:],:,:]
        trial_traces_df_f = np.array(data_grp['/'.join([curr_slice,cfg_key,'df_f', 'trace','np_subtracted'])])[trial_filter[cfg_count,:],:,:]
        trial_traces_zscore = np.array(data_grp['/'.join([curr_slice,cfg_key,'zscore', 'trace','np_subtracted'])])[trial_filter[cfg_count,:],:,:]

        trial_response_df = np.squeeze(np.mean(trial_traces_df[:,stim_period0:stim_period1,:],1))
        trial_response_df_f = np.squeeze(np.mean(trial_traces_df_f[:,stim_period0:stim_period1,:],1))
        trial_response_zscore = np.squeeze(np.mean(trial_traces_zscore[:,stim_period0:stim_period1,:],1))

        trial_baseline_f = np.squeeze(np.mean(trial_traces_f[:,0:stim_period0,:],1))
        trial_response_f = np.squeeze(np.mean(trial_traces_f[:,stim_period0:stim_period1,:],1))
        trial_response_f_zscore = np.true_divide(trial_response_f-np.mean(trial_baseline_f,0),\
                   np.std(trial_baseline_f,0))

        if cfg_count == 0:
            response_matrix_df = trial_response_df
            response_matrix_df_f = trial_response_df_f
            response_matrix_zscore = trial_response_zscore
            response_matrix_f_zscore = trial_response_f_zscore
            response_matrix_f = trial_response_f
            baseline_matrix_f = trial_baseline_f
        else:
            response_matrix_df = np.dstack((response_matrix_df,trial_response_df))
            response_matrix_df_f = np.dstack((response_matrix_df_f,trial_response_df_f))
            response_matrix_zscore = np.dstack((response_matrix_zscore,trial_response_zscore))
            response_matrix_f_zscore = np.dstack((response_matrix_f_zscore,trial_response_f_zscore))
            response_matrix_f = np.dstack((response_matrix_f,trial_response_f))
            baseline_matrix_f = np.dstack((baseline_matrix_f,trial_baseline_f))

    response_matrix_f = np.swapaxes(response_matrix_f,1,2) 
    baseline_matrix_f = np.swapaxes(baseline_matrix_f,1,2) 
    response_matrix_df = np.swapaxes(response_matrix_df,1,2) 
    response_matrix_df_f = np.swapaxes(response_matrix_df_f,1,2) 
    response_matrix_zscore = np.swapaxes(response_matrix_zscore,1,2) 
    response_matrix_f_zscore = np.swapaxes(response_matrix_f_zscore,1,2)



    #perform permutation test
    nreps = 100
    ntrials,nconfigs,nrois = response_matrix_f.shape
    ks_stat = np.empty((nconfigs,nrois))
    ks_p = np.empty((nconfigs,nrois))
    perm_tstat = np.empty((nconfigs,nrois))
    perm_p = np.empty((nconfigs,nrois))
    paired_tstat = np.empty((nconfigs,nrois))
    paired_p = np.empty((nconfigs,nrois))
    simple_tstat = np.empty((nconfigs,nrois))
    simple_p = np.empty((nconfigs,nrois))

    for cidx in range(nconfigs):
        for ridx in range(nrois):
            #get true change in fluoresence
            true_df = np.squeeze(response_matrix_df[:,cidx,ridx])

            #repeat permutation a bunch of times
            for rep in range(nreps):
                shuffle_all = np.random.permutation(np.hstack((np.squeeze(baseline_matrix_f[:,cidx,ridx]),np.squeeze(response_matrix_f[:,cidx,ridx]))))

                shuffle_base = shuffle_all[0:ntrials]
                shuffle_stim = shuffle_all[ntrials:]

                if rep == 0:
                    shuffle_df = shuffle_stim - shuffle_base
                else:
                    shuffle_df = np.hstack((shuffle_df,shuffle_stim - shuffle_base))

            #performs some stats and strone
            ks_stat[cidx,ridx], ks_p[cidx,ridx] = stats.ks_2samp(true_df,shuffle_df)
            perm_tstat[cidx,ridx], perm_p[cidx,ridx] = stats.ttest_ind(true_df,shuffle_df,equal_var = False)#using this test since the sampled come from same cell

            #get ttest stats by comparing df distribution to 0
            simple_tstat[cidx,ridx], simple_p[cidx,ridx] = stats.ttest_1samp(np.squeeze(response_matrix_df[:,cidx,ridx]),0)

            #get ttest stats by comparing pixel values between baseline and stim period
            paired_tstat[cidx,ridx], paired_p[cidx,ridx] = stats.ttest_rel(np.squeeze(response_matrix_f[:,cidx,ridx]),np.squeeze(baseline_matrix_f[:,cidx,ridx]))

    #sign your p-values
    simple_p = np.sign(simple_tstat)*simple_p
    paired_p = np.sign(paired_tstat)*paired_p

    #do split-half correlation to assess reliability of each cell
    nreps = 100

    split_size = int(np.floor(ntrials/2))
    R_cells = np.zeros((nreps,nrois))


    for rep in range(nreps):

        #randomly split
        rand_trials = np.random.permutation(ntrials)
        half1 = response_matrix_df[rand_trials[0:split_size]]
        half2 = response_matrix_df[rand_trials[split_size:-1]]

        #get mean response across trials
        half1_mean = np.squeeze(np.mean(half1,0))
        half2_mean = np.squeeze(np.mean(half2,0))

        #get cell split-half correlation
        for ridx in range(nrois):
            R_tmp = np.corrcoef(np.squeeze(half1_mean[:,ridx]),np.squeeze(half2_mean[:,ridx]))
            R_cells[rep,ridx] = R_tmp[0,1]
    split_half_R = np.nanmean(R_cells,0)
    #save to array

    #open file for storage
    # Create outfile:
    resp_array_fn = 'trial_response_array_motion_%i.hdf5'%(motion_thresh)
    resp_array_filepath = os.path.join(data_array_dir, resp_array_fn)
    print('Saving to: %s'%(resp_array_filepath))
    resp_grp = h5py.File(resp_array_filepath, 'w')

    #copy attributes
    for att_key in data_grp.attrs.keys():
        resp_grp.attrs[att_key] = data_grp.attrs[att_key]
    resp_grp.attrs['config_img'] = config_img
    resp_grp.attrs['config_cond'] = config_cond
    resp_grp.attrs['motion_thresh'] = motion_thresh

    data_grp.close()

    #save response
    motion_dset =  resp_grp.create_dataset('motion_filter',trial_filter.shape,trial_filter.dtype)
    motion_dset[...] = trial_filter

    f_zscore_dset = resp_grp.create_dataset('/'.join([curr_slice, 'responses' ,'f_zscore']), response_matrix_f_zscore.shape, response_matrix_f_zscore.dtype)
    f_zscore_dset[...] = response_matrix_f_zscore

    df_dset = resp_grp.create_dataset('/'.join([curr_slice, 'responses' ,'df']), response_matrix_df.shape, response_matrix_df.dtype)
    df_dset[...] = response_matrix_df

    df_f_dset = resp_grp.create_dataset('/'.join([curr_slice, 'responses' ,'df_f']), response_matrix_df_f.shape, response_matrix_df_f.dtype)
    df_f_dset[...] = response_matrix_df_f

    zscore_dset = resp_grp.create_dataset('/'.join([curr_slice, 'responses' ,'zscore']), response_matrix_zscore.shape, response_matrix_zscore.dtype)
    zscore_dset[...] = response_matrix_zscore

    kstat_dset = resp_grp.create_dataset('/'.join([curr_slice, 'responses' ,'ks_stat']), ks_stat.shape, ks_stat.dtype)
    kstat_dset[...] = ks_stat

    ksp_dset = resp_grp.create_dataset('/'.join([curr_slice, 'responses' ,'ks_p']), ks_stat.shape, ks_stat.dtype)
    ksp_dset[...] = ks_p

    perm_stat_dset = resp_grp.create_dataset('/'.join([curr_slice, 'responses' ,'perm_tstat']), perm_tstat.shape, perm_tstat.dtype)
    perm_stat_dset[...] = perm_tstat

    perm_p_dset = resp_grp.create_dataset('/'.join([curr_slice, 'responses' ,'perm_p']), perm_p.shape, perm_p.dtype)
    perm_p_dset[...] = perm_p

    simple_stat_dset = resp_grp.create_dataset('/'.join([curr_slice, 'responses' ,'simple_tstat']), simple_tstat.shape, simple_tstat.dtype)
    simple_stat_dset[...] = simple_tstat


    simple_p_dset = resp_grp.create_dataset('/'.join([curr_slice, 'responses' ,'simple_pval']), simple_p.shape, simple_p.dtype)
    simple_p_dset[...] = simple_p

    paired_stat_dset = resp_grp.create_dataset('/'.join([curr_slice, 'responses' ,'paired_tstat']), paired_tstat.shape, paired_tstat.dtype)
    paired_stat_dset[...] = paired_tstat


    paired_p_dset = resp_grp.create_dataset('/'.join([curr_slice, 'responses' ,'paired_pval']), paired_p.shape, paired_p.dtype)
    paired_p_dset[...] = paired_p

    r_dset = resp_grp.create_dataset('/'.join([curr_slice, 'responses' ,'split_half_R']), split_half_R.shape, split_half_R.dtype)
    r_dset[...] = split_half_R

    resp_grp.close()

class Struct():
    pass

def extract_options(options):
    parser = optparse.OptionParser()

    # PATH opts:
    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/n/coxfs01/2p-data', help='source dir (root project dir containing all expts) [default: /n/coxfs01/2p-data]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')
    parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID') 
    parser.add_option('-A', '--acq', action='store', dest='acquisition', default='', help="acquisition folder (ex: 'FOV1_zoom3x')")
    parser.add_option('-R', '--run', action='store', dest='run', default='', help='name of s2p run to process') 
    parser.add_option('-Y', '--analysis', action='store', dest='analysis', default='', help='Analysis to process. [ex: suite2p_analysis001]')
    parser.add_option('-T', '--traceid', action='store', dest='traceid', default='', help="(ex: traces001_s2p)")
    parser.add_option('-r', '--run_list', action='callback', dest='run_list', default='',type='string',callback=get_comma_separated_args, help='comma-separated names of run dirs containing tiffs to be processed (ex: run1, run2, run3)')
    parser.add_option('-C', '--combined_run', action='store', dest='combined_run', default='', help='name of combo run') 
    parser.add_option('-m', '--motion_thresh', action='store', dest='motion_thresh', default='5', help='threshold for motion to exclude trials') 

    (options, args) = parser.parse_args() 

    return options




#-----------------------------------------------------
#           MAIN SET OF ACTIONS
#-----------------------------------------------------

def main(options): 
    
    options = extract_options(options)

    # print('----- Parsing trials -----')
    # parse_trials(options)
    # print('----- Combining trials ----')
    # combine_trials(options)
    # print('----- Evaluating trials ----')
    # evaluate_trials(options)
    print('----- Getting mean response for trials ----')
    get_trial_responses(options)
    

    
#%%

if __name__ == '__main__':
    main(sys.argv[1:])
