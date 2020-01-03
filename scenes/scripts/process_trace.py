
import h5py
import matplotlib
matplotlib.use('Agg')
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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

def get_trial_stat(base,stim):
    #get KS signed tstat and pvl
    
    diff_sign = np.sign(np.mean(stim)-np.mean(base))
    stat,pval = stats.ks_2samp(base,stim)
    stat = diff_sign*stat
    pval = diff_sign*pval
    
    return stat,pval

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

def plot_motion(opts):
    traceid = '%s_s2p'%(opts.traceid)
    #hard-coding some parameters
    iti_pre = 1.0
    iti_post = 1.95

    trials_per_file = 25

    acquisition_dir = os.path.join(opts.rootdir, opts.animalid, opts.session, opts.acquisition)


    fig_base_dir = os.path.join(opts.rootdir, opts.animalid, opts.session,'registration_figures')
    reg_features = ['offset','motion']
    

    for run in opts.run_list:
        print(run)

        traceid_dir = os.path.join(acquisition_dir, run, 'traces',traceid)

        trace_arrays_dir = os.path.join(traceid_dir,'files')
      
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



        trace_files = [f for f in os.listdir(trace_arrays_dir) if 'File' in f and f.endswith('hdf5')]


        #file_idx = 0
        #trace_file = trace_files[0]
        for file_idx,trace_file in enumerate(trace_files):
            trace_fn = os.path.join(trace_arrays_dir,trace_file)

            rawfile = h5py.File(trace_fn, 'r')

            stim_on_frames = []

            for tridx in range(trials_per_file):
                tridx_all = tridx+(trials_per_file*file_idx)
                trial_key = 'trial%05d' % (tridx_all+1)
                stim_on_frames.append(mwinfo[trial_key]['frame_stim_on'])
            stim_on_frames = np.array(stim_on_frames)-(nvolumes*file_idx)

            #open stimulus condition file
            stimconfig_fn = 'trial_conditions.hdf5'
            paradigm_dir = os.path.join(acquisition_dir, run, 'paradigm')
            stimconfig_filepath = os.path.join(paradigm_dir, 'files', stimconfig_fn)
            run_config_grp = h5py.File(stimconfig_filepath, 'r')

            trial_config = np.array(run_config_grp['trial_config']).astype('int')
            trial_img = np.array(run_config_grp['trial_img']).astype('int')
            trial_cond = np.array(run_config_grp['trial_cond']).astype('int')

            palette=["#4c72b0","#c44e52","#55a868"]

            for idx,reg in enumerate(reg_features):
                reg_metric =  np.array(rawfile.attrs[reg])
                min_reg_metric =  int(rawfile.attrs['min_%s'%reg])
                max_reg_metric =  int(rawfile.attrs['max_%s'%reg])

                fig_dir = os.path.join(fig_base_dir,reg)
                if not os.path.isdir(fig_dir):
                    os.makedirs(fig_dir)

                fig=plt.figure(figsize = (30, 5))


                plt.plot(all_frames_tsecs,reg_metric)
                if reg == 'offset':
                    plt.axhline(y = 10, xmin = 0, xmax = 1, linewidth=1, color='k',linestyle ='--')
                else:
                    plt.axhline(y = 1, xmin = 0, xmax = 1, linewidth=1, color='k',linestyle ='--')
               
                plt.axhline(y = 0, xmin = 0, xmax = 1, linewidth=1, color='k',linestyle ='-')
                axes = plt.gca()
                axes.set_ylim([min_reg_metric,max_reg_metric])
                ymin, ymax = axes.get_ylim()
                for fidx,f in enumerate(all_frames_tsecs[stim_on_frames]):
                    axes.add_patch(patches.Rectangle((f, ymin), 1, ymax-ymin, linewidth=0, fill=True, color=palette[trial_cond[fidx+(file_idx*trials_per_file)]], alpha=0.4));
                    axes.text(f,ymax,'%i'%(trial_img[fidx+(file_idx*trials_per_file)]),fontsize=12);

                plt.xlabel('Time (secs)',fontsize = 14)
                plt.ylabel(reg,fontsize = 14)
                fig.suptitle(reg)
                
                fig_fn = '%s_timecourse_%s_file%03d.png'%(reg,run,file_idx+1)
                plt.savefig(os.path.join(fig_dir,fig_fn))
                plt.close()

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
                    motion = np.array(rawfile.attrs['motion'])
                    offset = np.array(rawfile.attrs['offset'])


                    raw_df = rawfile[curr_slice]['traces']['pixel_value']['raw'][:]
                    sub_df = rawfile[curr_slice]['traces']['pixel_value']['cell'][:]
                    np_df = rawfile[curr_slice]['traces']['pixel_value']['neuropil'][:]

                    df_f =rawfile[curr_slice]['traces']['global_df_f']['cell'][:]
                    spks =rawfile[curr_slice]['traces']['spks']['cell'][:]

                    print(run_count,fid)
                    mean_pix_fid[fid,:] = np.squeeze(np.mean(np.squeeze(np.mean(sub_df,0)),0))
            else:
                #load file
                trace_file = [f for f in os.listdir(trace_arrays_dir) if 'File%03d'%(fid+1) in f and f.endswith('hdf5')][0]
                trace_fn = os.path.join(trace_arrays_dir,trace_file)

                rawfile = h5py.File(trace_fn, 'r')
                
                motion = np.array(rawfile.attrs['motion'])
                offset = np.array(rawfile.attrs['offset'])
                raw_df = rawfile[curr_slice]['traces']['pixel_value']['raw'][:]
                sub_df = rawfile[curr_slice]['traces']['pixel_value']['cell'][:]
                np_df = rawfile[curr_slice]['traces']['pixel_value']['neuropil'][:]

                df_f =rawfile[curr_slice]['traces']['global_df_f']['cell'][:]
                spks =rawfile[curr_slice]['traces']['spks']['cell'][:]


                #copy some attributes
                file_grp.attrs['s2p_cell_rois'] = rawfile.attrs['s2p_cell_rois']
                file_grp.attrs['roi_center_x'] = rawfile.attrs['roi_center_x']
                file_grp.attrs['roi_center_y'] = rawfile.attrs['roi_center_y']
                file_grp.attrs['roi_compact'] = rawfile.attrs['roi_compact']
                file_grp.attrs['roi_skew'] = rawfile.attrs['roi_skew']
                file_grp.attrs['roi_aspect_ratio'] = rawfile.attrs['roi_aspect_ratio']
                file_grp.attrs['roi_radius'] = rawfile.attrs['roi_radius']
                file_grp.attrs['roi_footprint'] = rawfile.attrs['roi_footprint']
                file_grp.attrs['roi_npix'] = rawfile.attrs['roi_npix']




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
                parsed_offset = np.empty((ntrials,ntpts))
                parsed_offset[:] = np.nan
                parsed_motion = np.copy(parsed_offset)

                parsed_traces_df_f = np.empty((ntrials,ntpts,nrois))
                parsed_traces_df_f[:] = np.nan

                parsed_traces_spks = np.empty((ntrials,ntpts,nrois))
                parsed_traces_spks[:] = np.nan


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
            trial_df_f = df_f[idx0:idx1,:]
            trial_spks = spks[idx0:idx1,:]

            parsed_traces_raw[trial_idx,0:trial_frames,:]=trial_df_raw
            parsed_traces_sub[trial_idx,0:trial_frames,:]=trial_df_sub
            parsed_traces_np[trial_idx,0:trial_frames,:]=trial_df_np
            parsed_traces_df_f[trial_idx,0:trial_frames,:]=trial_df_f
            parsed_traces_spks[trial_idx,0:trial_frames,:]=trial_spks


            parsed_offset[trial_idx,0:trial_frames] = offset[idx0:idx1]
            parsed_motion[trial_idx,0:trial_frames] = motion[idx0:idx1]
            trial_fid[trial_idx] = fid

        #get parsed trace time stamps
        tstamps_indices = np.arange(parsed_traces_raw.shape[1])                          
        volumeperiod = 1/volumerate
        curr_tstamps = (tstamps_indices*volumeperiod)
        curr_tstamps = curr_tstamps - iti_pre

        #save arrays to file
        offset = file_grp.create_dataset('trial_offset', parsed_offset.shape, parsed_offset.dtype)
        offset[...] = parsed_offset

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



        raw_parsed = file_grp.create_dataset('/'.join([curr_slice, 'traces', 'pixel_value', 'raw']), parsed_traces_raw.shape, parsed_traces_raw.dtype)
        raw_parsed[...] = parsed_traces_raw

        sub_parsed = file_grp.create_dataset('/'.join([curr_slice, 'traces', 'pixel_value','cell']), parsed_traces_sub.shape, parsed_traces_sub.dtype)
        sub_parsed[...] = parsed_traces_sub

        np_parsed = file_grp.create_dataset('/'.join([curr_slice, 'traces', 'pixel_value','neuropil']), parsed_traces_np.shape, parsed_traces_np.dtype)
        np_parsed[...] = parsed_traces_np

        df_f_parsed = file_grp.create_dataset('/'.join([curr_slice, 'traces', 'global_df_f', 'cell']), parsed_traces_df_f.shape, parsed_traces_df_f.dtype)
        df_f_parsed[...] = parsed_traces_df_f

        spks_parsed = file_grp.create_dataset('/'.join([curr_slice, 'traces', 'spks', 'cell']), parsed_traces_spks.shape, parsed_traces_spks.dtype)
        spks_parsed[...] = parsed_traces_spks

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
        #get offset motion
        offset_run = np.array(file_grp['trial_offset'])
        motion_run = np.array(file_grp['trial_motion'])
        #get raw pixel value arrays
        pix_raw_run = np.array(file_grp[curr_slice]['traces']['pixel_value']['raw'])
        pix_cell_run = np.array(file_grp[curr_slice]['traces']['pixel_value']['cell'])
        pix_np_run = np.array(file_grp[curr_slice]['traces']['pixel_value']['neuropil'])
        df_f_cell_run = np.array(file_grp[curr_slice]['traces']['global_df_f']['cell'])
        spks_cell_run = np.array(file_grp[curr_slice]['traces']['spks']['cell'])

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
            
            
            
            #copy some attributes
            combined_grp.attrs['s2p_cell_rois'] = file_grp.attrs['s2p_cell_rois']
            combined_grp.attrs['roi_center_x'] = file_grp.attrs['roi_center_x']
            combined_grp.attrs['roi_center_y'] = file_grp.attrs['roi_center_y']
            combined_grp.attrs['roi_compact'] = file_grp.attrs['roi_compact']
            combined_grp.attrs['roi_skew'] = file_grp.attrs['roi_skew']
            combined_grp.attrs['roi_aspect_ratio'] = file_grp.attrs['roi_aspect_ratio']
            combined_grp.attrs['roi_radius'] = file_grp.attrs['roi_radius']
            combined_grp.attrs['roi_footprint'] = file_grp.attrs['roi_footprint']
            combined_grp.attrs['roi_npix'] = file_grp.attrs['roi_npix']

            pix_raw_combo = pix_raw_run
            pix_cell_combo = pix_cell_run
            pix_np_combo = pix_np_run
            df_f_cell_combo = df_f_cell_run
            spks_cell_combo = spks_cell_run
            trial_fid_combo = trial_fid
            trial_run_combo = trial_run
            mean_pix_fid_combo = mean_pix_fid
            offset_run_combo = offset_run
            motion_run_combo = motion_run
        else:
            #for now, lop off timepoints if doesn't match what we already have, if volumerate fast enough. this is negligible
            if pix_raw_run.shape[1]>pix_raw_combo.shape[1]:
                extra_frames = pix_raw_run.shape[1]-pix_raw_combo.shape[1]
                pix_raw_run = pix_raw_run[:,:-extra_frames,:]
                pix_cell_run = pix_cell_run[:,:-extra_frames,:]
                pix_np_run = pix_np_run[:,:-extra_frames,:]
                df_f_cell_run = df_f_cell_run[:,:-extra_frames,:]
                spks_cell_run = spks_cell_run[:,:-extra_frames,:]
                offset_run = offset_run[:,:-extra_frames]
                motion_run = motion_run[:,:-extra_frames]
            
            pix_raw_combo = np.vstack((pix_raw_combo,pix_raw_run))
            pix_cell_combo = np.vstack((pix_cell_combo,pix_cell_run))
            pix_np_combo = np.vstack((pix_np_combo,pix_np_run))
            df_f_cell_combo = np.vstack((df_f_cell_combo,df_f_cell_run))
            spks_cell_combo = np.vstack((spks_cell_combo,spks_cell_run))

            offset_run_combo = np.vstack((offset_run_combo,offset_run))
            motion_run_combo = np.vstack((motion_run_combo,motion_run))
            
            trial_fid_combo = np.hstack((trial_fid_combo,trial_fid))
            trial_run_combo = np.hstack((trial_run_combo,trial_run))
            mean_pix_fid_combo = np.vstack((mean_pix_fid_combo,mean_pix_fid))
            
        file_grp.close()

    #save combined traces to file
    #save arrays to file

    offset = combined_grp.create_dataset('offset', offset_run_combo.shape, offset_run_combo.dtype)
    offset[...] = offset_run_combo

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

    raw_combined = combined_grp.create_dataset('/'.join([curr_slice, 'traces', 'pixel_value', 'raw']), pix_raw_combo.shape, pix_raw_combo.dtype)
    raw_combined[...] = pix_raw_combo

    cell_combined = combined_grp.create_dataset('/'.join([curr_slice, 'traces', 'pixel_value', 'cell']), pix_cell_combo.shape, pix_cell_combo.dtype)
    cell_combined[...] = pix_cell_combo

    np_combined = combined_grp.create_dataset('/'.join([curr_slice, 'traces', 'pixel_value', 'neuropil']), pix_np_combo.shape, pix_np_combo.dtype)
    np_combined[...] = pix_np_combo

    df_f_combined = combined_grp.create_dataset('/'.join([curr_slice,'traces', 'global_df_f', 'cell']), df_f_cell_combo.shape, df_f_cell_combo.dtype)
    df_f_combined[...] = df_f_cell_combo

    spks_combined = combined_grp.create_dataset('/'.join([curr_slice,'traces', 'spks', 'cell']), spks_cell_combo.shape, spks_cell_combo.dtype)
    spks_combined[...] = spks_cell_combo

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



    #read file
    parsedtraces_filepath = glob.glob(os.path.join(traceid_dir, 'files','parsedtraces*'))[0]
    file_grp = h5py.File(parsedtraces_filepath, 'r')

    #get motion info
    offset = np.array(file_grp['offset'])
    motion = np.array(file_grp['motion'])


    pre_frames = file_grp.attrs['pre_frames']
    post_frames = file_grp.attrs['post_frames']
    stim_frames = file_grp.attrs['stim_frames']
    #to get baseline index with [0:pre_frames]
    #to get stim period do [pre_frames:pre_frames+stim_frames+1]

    curr_slice = 'Slice01'#hard-code planar data for now
    #get raw pixel value arrays
    pix_raw_array = np.array(file_grp[curr_slice]['traces']['pixel_value']['raw'])
    pix_cell_array = np.array(file_grp[curr_slice]['traces']['pixel_value']['cell'])
    pix_np_array = np.array(file_grp[curr_slice]['traces']['pixel_value']['neuropil'])

    global_df_f_array = np.array(file_grp[curr_slice]['traces']['global_df_f']['cell'])
    spks_array = np.array(file_grp[curr_slice]['traces']['spks']['cell'])

    mean_f_raw = np.nanmean(np.nanmean(pix_raw_array,0),0)
    mean_f_cell = np.nanmean(np.nanmean(pix_cell_array,0),0)
    mean_f_np = np.nanmean(np.nanmean(pix_np_array,0),0)


    #open file for storage
    # Create outfile:
    data_array_fn = 'processed_traces.hdf5'
    data_array_filepath = os.path.join(traceid_dir, 'files', data_array_fn)
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




    #file_grp.close()

    ntrials,ntpts,nrois = pix_cell_array.shape
    print('ROIs:%i'%(nrois))
    print('Trials:%i'%(ntrials))

    #get empty arrays
    f_cell_array = np.zeros(pix_cell_array.shape)
    f_raw_array = np.zeros(pix_raw_array.shape)
    f_np_array = np.zeros(pix_np_array.shape)

    df_cell_array = np.zeros(pix_cell_array.shape)
    df_raw_array = np.zeros(pix_raw_array.shape)
    df_np_array = np.zeros(pix_np_array.shape)

    local_df_f_cell_array = np.zeros(pix_cell_array.shape)
    local_df_f_raw_array = np.zeros(pix_raw_array.shape)
    local_df_f_np_array = np.zeros(pix_np_array.shape)

    zscore_cell_array = np.zeros(pix_cell_array.shape)
    zscore_raw_array = np.zeros(pix_raw_array.shape)
    zscore_np_array = np.zeros(pix_np_array.shape)

    delta_global_df_f_array = np.zeros(pix_cell_array.shape)
    delta_spks_array = np.zeros(pix_cell_array.shape)

    zscore_global_df_f_array = np.zeros(pix_cell_array.shape)
    zscore_spks_array = np.zeros(pix_cell_array.shape)


    trial_stat = np.zeros((ntrials,nrois))
    trial_pval = np.zeros((ntrials,nrois))

    #evalue trials
    for tidx in range(ntrials):
    #tidx = 0
    #tidx = 100
        for ridx in range(nrois):
    #ridx = 3

            #get trial timecourse
            pix_raw = np.squeeze(pix_raw_array[tidx,:,ridx])
            pix_cell = np.squeeze(pix_cell_array[tidx,:,ridx].squeeze())
            pix_np = np.squeeze(pix_np_array[tidx,:,ridx].squeeze())


            #get baseline and stimulus period values
            base_raw = pix_raw[0:pre_frames]
            stim_raw = pix_raw[pre_frames:pre_frames+stim_frames+1]
            base_cell = pix_cell[0:pre_frames]
            stim_cell = pix_cell[pre_frames:pre_frames+stim_frames+1]
            base_np = pix_np[0:pre_frames]
            stim_np = pix_np[pre_frames:pre_frames+stim_frames+1]
            
            base_df_f = global_df_f_array[tidx,0:pre_frames,ridx]
            stim_df_f = global_df_f_array[tidx,pre_frames:pre_frames+stim_frames+1,ridx]
            
            base_spks = spks_array[tidx,0:pre_frames,ridx]
            stim_spks = spks_array[tidx,pre_frames:pre_frames+stim_frames+1,ridx]
            
            #run some stats to determine if responsive trial(KS test)
            trial_stat[tidx,ridx],trial_pval[tidx,ridx] = get_trial_stat(base_cell,stim_cell)
            
            #get raw fluoresence value
            f_raw = pix_raw
            f_cell = pix_cell
            f_np = pix_np

            #calculate df
            df_raw = pix_raw - np.mean(base_raw)
            df_cell = pix_cell - np.mean(base_cell)
            df_np = pix_np - np.mean(base_np)


            #calculate local df/f
            local_df_f_raw = df_raw/mean_f_raw[ridx]
            local_df_f_cell = df_cell/mean_f_cell[ridx]
            local_df_f_np = df_np/mean_f_np[ridx]
            
            #calculate global df_f
            global_df_f = global_df_f_array[tidx,:,ridx] - np.mean(base_df_f)
            zscore_df_f = (global_df_f_array[tidx,:,ridx] - np.mean(base_df_f))/np.std(base_df_f)
            
            #calculate change in spike count
            delta_spks = spks_array[tidx,:,ridx] - np.mean(base_spks)
            zscore_spks = (spks_array[tidx,:,ridx] - np.mean(base_spks)) / np.std(base_spks)

            #calculate z-score
            zscore_raw = (pix_raw - np.mean(base_raw))/np.std(base_raw)
            zscore_cell = (pix_cell - np.mean(base_cell))/np.std(base_cell)
            zscore_np = (pix_np - np.mean(base_np))/np.std(base_np)

            #store in array
            f_raw_array[tidx,:,ridx] = f_raw
            df_raw_array[tidx,:,ridx] = df_raw
            local_df_f_raw_array[tidx,:,ridx] = local_df_f_raw
            zscore_raw_array[tidx,:,ridx] = zscore_raw

            f_cell_array[tidx,:,ridx] = f_cell
            df_cell_array[tidx,:,ridx] = df_cell
            local_df_f_cell_array[tidx,:,ridx] = local_df_f_cell
            zscore_cell_array[tidx,:,ridx] = zscore_cell
            
            delta_global_df_f_array[tidx,:,ridx] = global_df_f
            zscore_global_df_f_array[tidx,:,ridx] = zscore_df_f
            
            delta_spks_array[tidx,:,ridx] = delta_spks
            zscore_spks_array[tidx,:,ridx] = zscore_spks

            f_np_array[tidx,:,ridx] = f_np
            df_np_array[tidx,:,ridx] = df_np
            local_df_f_np_array[tidx,:,ridx] = local_df_f_np
            zscore_np_array[tidx,:,ridx] = zscore_np

    #load trial paradigm info
    stimconfig_fn = 'trial_conditions.hdf5'
    stimconfig_filepath = os.path.join(paradigm_dir, 'files', stimconfig_fn)
    config_grp = h5py.File(stimconfig_filepath, 'r')

    ntrials = config_grp.attrs['ntrials']
    trial_config = np.array(config_grp['trial_config'])
    trial_cond = np.array(config_grp['trial_cond'])
    trial_img = np.array(config_grp['trial_img'])

    config_grp.close()

    #save arrays to file

    #save motion

    offset_set = data_grp.create_dataset('offset', offset.shape, offset.dtype)
    offset_set[...] = offset

    motion_set = data_grp.create_dataset('motion', motion.shape, motion.dtype)
    motion_set[...] = motion

    #save config specifics
    trial_cfg_dset = data_grp.create_dataset('trial_config',trial_config.shape, trial_config.dtype)
    trial_cfg_dset[...] = trial_config

    trial_img_dset = data_grp.create_dataset('trial_img',trial_img.shape, trial_img.dtype)
    trial_img_dset[...] = trial_img

    trial_cond_dset = data_grp.create_dataset('trial_cond',trial_cond.shape, trial_cond.dtype)
    trial_cond_dset[...] = trial_cond



    #save trace arrays


    raw_f_trace_dset = data_grp.create_dataset('/'.join([curr_slice, 'traces','f','raw']),\
                                               f_raw_array.shape, f_raw_array.dtype)
    raw_f_trace_dset[...] = f_raw_array

    cell_f_trace_dset = data_grp.create_dataset('/'.join([curr_slice, 'traces','f','cell']),\
                                                f_cell_array.shape, f_cell_array.dtype)
    cell_f_trace_dset[...] = f_cell_array

    np_f_trace_dset = data_grp.create_dataset('/'.join([curr_slice, 'traces','f','np']),\
                                              f_np_array.shape, f_np_array.dtype)
    np_f_trace_dset[...] = f_np_array

    raw_df_trace_dset = data_grp.create_dataset('/'.join([curr_slice, 'traces','df','raw']),\
                                                df_raw_array.shape, df_raw_array.dtype)
    raw_df_trace_dset[...] = df_raw_array

    cell_df_trace_dset = data_grp.create_dataset('/'.join([curr_slice, 'traces','df','cell']),\
                                                 df_cell_array.shape, df_cell_array.dtype)
    cell_df_trace_dset[...] = df_cell_array

    np_df_trace_dset = data_grp.create_dataset('/'.join([curr_slice, 'traces','df','np']),\
                                               df_np_array.shape, df_np_array.dtype)
    np_df_trace_dset[...] = df_np_array

    raw_df_ftrace_dset = data_grp.create_dataset('/'.join([curr_slice, 'traces','local_df_f','raw']),\
                                                 local_df_f_raw_array.shape, local_df_f_raw_array.dtype)
    raw_df_ftrace_dset[...] = local_df_f_raw_array

    cell_df_ftrace_dset = data_grp.create_dataset('/'.join([curr_slice, 'traces','local_df_f','cell']),\
                                                  local_df_f_cell_array.shape, local_df_f_cell_array.dtype)
    cell_df_ftrace_dset[...] = local_df_f_cell_array

    np_df_ftrace_dset = data_grp.create_dataset('/'.join([curr_slice, 'traces','local_df_f','np']),\
                                                local_df_f_np_array.shape, local_df_f_np_array.dtype)
    np_df_ftrace_dset[...] = local_df_f_np_array


    raw_zscore_trace_dset = data_grp.create_dataset('/'.join([curr_slice, 'traces','zscore_df','raw']),\
                                                    zscore_raw_array.shape, zscore_raw_array.dtype)
    raw_zscore_trace_dset[...] = zscore_raw_array

    cell_zscore_trace_dset = data_grp.create_dataset('/'.join([curr_slice, 'traces','zscore_df','cell']),\
                                                     zscore_cell_array.shape, zscore_cell_array.dtype)
    cell_zscore_trace_dset[...] = zscore_cell_array

    np_zscore_trace_dset = data_grp.create_dataset('/'.join([curr_slice, 'traces','zscore_df','np']),\
                                                   zscore_np_array.shape, zscore_np_array.dtype)
    np_zscore_trace_dset[...] = zscore_np_array

     
    global_df_ftrace_dset = data_grp.create_dataset('/'.join([curr_slice, 'traces','global_df_f','cell']),\
                                                    delta_global_df_f_array.shape, delta_global_df_f_array.dtype)
    global_df_ftrace_dset[...] = delta_global_df_f_array   

    zscore_df_f_trace_dset = data_grp.create_dataset('/'.join([curr_slice, 'traces','zscore_df_f','cell']),\
                                                     zscore_global_df_f_array.shape, zscore_global_df_f_array.dtype)
    zscore_df_f_trace_dset[...] = zscore_global_df_f_array

    spks_trace_dset = data_grp.create_dataset('/'.join([curr_slice, 'traces','delta_spks','cell']),\
                                              delta_spks_array.shape, delta_spks_array.dtype)

    spks_trace_dset[...] = delta_spks_array   

    zscore_spks_trace_dset = data_grp.create_dataset('/'.join([curr_slice, 'traces','zscore_spks','cell']),\
                                                     zscore_spks_array.shape, zscore_spks_array.dtype)
    zscore_spks_trace_dset[...] = zscore_spks_array


    data_grp.close()

def get_trial_responses(opts):
    traceid = '%s_s2p'%(opts.traceid)

    # if opts.motion_thresh is not None:
    #     motion_thresh = int(opts.motion_thresh)
    # else:
    #     motion_thresh = 'None'
        
    #% Set up paths:    
    acquisition_dir = os.path.join(opts.rootdir, opts.animalid, opts.session, opts.acquisition)

    traceid_dir = os.path.join(acquisition_dir, opts.combined_run,'traces', traceid)

    run_dir = traceid_dir.split('/traces')[0]
    trace_arrays_dir = os.path.join(traceid_dir,'files')
    paradigm_dir = os.path.join(acquisition_dir, opts.combined_run, 'paradigm')

    #output directory
    responses_dir = os.path.join(acquisition_dir, opts.combined_run,'responses', traceid)

    responses_file_dir = os.path.join(responses_dir,'files')
    if not os.path.exists(responses_file_dir):os.makedirs(responses_file_dir)

    #open file to read
    trace_fn = 'processed_traces.hdf5'
    trace_filepath = os.path.join(traceid_dir, 'files', trace_fn)
    print('Opening:%s'%(trace_filepath))
    data_grp = h5py.File(trace_filepath, 'r')

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

    #get traces
    trial_traces_df = np.array(data_grp['/'.join([curr_slice,'traces','df','cell'])])
    trial_traces_dspks = np.array(data_grp['/'.join([curr_slice,'traces','delta_spks','cell'])])

    trial_traces_lcl_df_f = np.array(data_grp['/'.join([curr_slice,'traces','local_df_f','cell'])])
    trial_traces_glbl_df_f = np.array(data_grp['/'.join([curr_slice,'traces','global_df_f','cell'])])

    trial_traces_zscore_df = np.array(data_grp['/'.join([curr_slice,'traces','zscore_df','cell'])])
    trial_traces_zscore_spks = np.array(data_grp['/'.join([curr_slice,'traces','zscore_spks','cell'])])

    #get mean during stim period
    trial_mean_df = np.squeeze(np.nanmean(trial_traces_df[:,stim_period0:stim_period1,:],1))
    trial_mean_dspks = np.squeeze(np.nanmean(trial_traces_dspks[:,stim_period0:stim_period1,:],1))
    trial_mean_lcl_df_f = np.squeeze(np.nanmean(trial_traces_lcl_df_f[:,stim_period0:stim_period1,:],1))
    trial_mean_glbl_df_f = np.squeeze(np.nanmean(trial_traces_glbl_df_f[:,stim_period0:stim_period1,:],1))
    trial_mean_zscore_df = np.squeeze(np.nanmean(trial_traces_zscore_df[:,stim_period0:stim_period1,:],1))
    trial_mean_zscore_spks = np.squeeze(np.nanmean(trial_traces_zscore_spks[:,stim_period0:stim_period1,:],1))

    #get max during stim period
    trial_max_df = np.squeeze(np.nanmax(trial_traces_df[:,stim_period0:stim_period1,:],1))
    trial_max_dspks = np.squeeze(np.nanmax(trial_traces_dspks[:,stim_period0:stim_period1,:],1))
    trial_max_lcl_df_f = np.squeeze(np.nanmax(trial_traces_lcl_df_f[:,stim_period0:stim_period1,:],1))
    trial_max_glbl_df_f = np.squeeze(np.nanmax(trial_traces_glbl_df_f[:,stim_period0:stim_period1,:],1))
    trial_max_zscore_df = np.squeeze(np.nanmax(trial_traces_zscore_df[:,stim_period0:stim_period1,:],1))
    trial_max_zscore_spks = np.squeeze(np.nanmax(trial_traces_zscore_spks[:,stim_period0:stim_period1,:],1))



    #get max offset and motion per trial
    trial_max_offset = np.nanmax(np.abs(np.array(data_grp['offset'])),1)
    trial_max_motion = np.nanmax(np.abs(np.array(data_grp['motion'])),1)

    #save arrays

    #open file for storage
    # Create outfile:
    resp_array_fn = 'trial_response_array.hdf5'
    resp_array_filepath = os.path.join(responses_file_dir, resp_array_fn)
    print('Saving to: %s'%(resp_array_filepath))
    resp_grp = h5py.File(resp_array_filepath, 'w')

    #copy attributes
    for att_key in data_grp.attrs.keys():
        resp_grp.attrs[att_key] = data_grp.attrs[att_key]

    resp_grp['trial_config'] = np.array(data_grp['trial_config'])
    resp_grp['trial_cond'] = np.array(data_grp['trial_cond'])
    resp_grp['trial_img'] = np.array(data_grp['trial_img'])

    resp_grp['trial_max_offset'] = trial_max_offset
    resp_grp['trial_max_motion'] = trial_max_motion

    resp_grp['/'.join([curr_slice,'mean_response','df'])] = trial_mean_df
    resp_grp['/'.join([curr_slice,'mean_response','dspks'])] = trial_mean_dspks
    resp_grp['/'.join([curr_slice,'mean_response','local_df_f'])] = trial_mean_lcl_df_f

    resp_grp['/'.join([curr_slice,'mean_response','global_df_f'])] = trial_mean_glbl_df_f
    resp_grp['/'.join([curr_slice,'mean_response','zscore_df'])] = trial_mean_zscore_df
    resp_grp['/'.join([curr_slice,'mean_response','zscore_spks'])] = trial_mean_zscore_spks

    resp_grp['/'.join([curr_slice,'max_response','df'])] = trial_max_df
    resp_grp['/'.join([curr_slice,'max_response','dspks'])] = trial_max_dspks
    resp_grp['/'.join([curr_slice,'max_response','local_df_f'])] = trial_max_lcl_df_f
    resp_grp['/'.join([curr_slice,'max_response','global_df_f'])] = trial_max_glbl_df_f
    resp_grp['/'.join([curr_slice,'max_response','zscore_df'])] = trial_max_zscore_df
    resp_grp['/'.join([curr_slice,'max_response','zscore_spks'])] = trial_max_zscore_spks


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


    print('----- Parsing trials -----')
    parse_trials(options)

    print('----- Plotting Motion -----')
    plot_motion(options)

    print('----- Combining trials ----')
    combine_trials(options)
    print('----- Evaluating trials ----')
    evaluate_trials(options)
    print('----- Getting mean response for trials ----')
    get_trial_responses(options)
    

    
#%%

if __name__ == '__main__':
    main(sys.argv[1:])
