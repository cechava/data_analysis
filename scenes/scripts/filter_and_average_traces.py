import h5py

import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import matplotlib
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

def findOccurrences(s, ch):
    return [i for i, letter in enumerate(s) if letter == ch]

def filter_traces(opts):

    filter_crit = opts.filter_crit
    filter_thresh = int(opts.filter_thresh)

    traceid = '%s_s2p'%(opts.traceid)
    #% Set up paths:    
    acquisition_dir = os.path.join(opts.rootdir, opts.animalid, opts.session, opts.acquisition)
    traceid_dir = os.path.join(acquisition_dir, opts.combined_run,'traces',traceid)

    run_dir = traceid_dir.split('/traces')[0]
    trace_arrays_dir = os.path.join(traceid_dir,'files')
    paradigm_dir = os.path.join(acquisition_dir, opts.combined_run, 'paradigm')

    #output directory for figure
    fig_folder = 'motion_%s_%s_%d'%(opts.motion_thresh,filter_crit,filter_thresh)
    fig_out_dir = os.path.join(traceid_dir,'figures','filtered_traces',fig_folder)
    if not os.path.exists(fig_out_dir):
        os.makedirs(fig_out_dir)

    #read trace file file
    parsedtraces_filepath = glob.glob(os.path.join(traceid_dir, 'files','parsedtraces*'))[0]
    file_grp = h5py.File(parsedtraces_filepath, 'r')

    curr_slice = 'Slice01'#hard-code planar data for now
    pix_cell_array = np.array(file_grp[curr_slice]['traces']['np_subtracted'])
    nrois = pix_cell_array.shape[2]

    if 's2p_cell_rois' in file_grp.attrs.keys():
        cell_rois = file_grp.attrs['s2p_cell_rois']
    else:
        cell_rois = np.arange(nrois)


    

    pre_frames = file_grp.attrs['pre_frames']
    post_frames = file_grp.attrs['post_frames']
    stim_frames = file_grp.attrs['stim_frames']
    #to get baseline index with [0:pre_frames]
    #to get stim period do [pre_frames:pre_frames+stim_frames+1]


    frames_tsec = np.array(file_grp[curr_slice]['frames_tsec'])
    #get raw pixel value arrays



    file_grp.close()

    #get file with trial-by-trial responses
    responses_dir = os.path.join(acquisition_dir, opts.combined_run,'responses',traceid)
    data_array_dir = os.path.join(responses_dir, 'data_arrays')

    resp_array_fn = 'trial_response_array_motion_%s.hdf5'%(opts.motion_thresh)
    resp_array_filepath = os.path.join(data_array_dir, resp_array_fn)
    resp_grp = h5py.File(resp_array_filepath, 'r')


    curr_slice = 'Slice01'#hard,coding for now

    #unpack
    filter_crit_matrix_trials = np.array(resp_grp['/'.join([curr_slice, 'responses' ,filter_crit])])
    filter_crit_matrix_trials = filter_crit_matrix_trials[:,:,cell_rois]


    #consider only cell rois and trials that pass threshold
    pix_cell_array = pix_cell_array[:,:,cell_rois]
    ntrials,ntpts,nrois = pix_cell_array.shape

    resp_grp.close()


    filter_crit_matrix_mean = np.squeeze(np.mean(filter_crit_matrix_trials,0))

    #consider a config active if at least one of theversion of an image evoked a response above threshol
    thresh = filter_thresh
    thresh_matrix = filter_crit_matrix_mean>thresh
    filter_matrix = np.zeros((thresh_matrix.shape))
    for ridx in range(nrois):
        for idx in range(0,thresh_matrix.shape[0],3):
                if np.sum(thresh_matrix[idx:idx+3,ridx])>0:
                    filter_matrix[idx:idx+3,ridx] = 1



    #load trial info
    stimconfig_fn = 'trial_conditions.hdf5'
    stimconfig_filepath = os.path.join(paradigm_dir, 'files', stimconfig_fn)
    config_grp = h5py.File(stimconfig_filepath, 'r')

    trial_config = np.array(config_grp['trial_config'])
    trial_cond = np.array(config_grp['trial_cond'])
    trial_img = np.array(config_grp['trial_img'])

    config_grp.close()

    #for each roi, first average across trials per config, then normalize across configs
    #-should result in higher values, also noticed bigger differences in values

    nconfigs = len(np.unique(trial_config))
    print('ROIs:%i'%(nrois))
    print('Trials:%i'%(ntrials))


    norm_roi_df_array = np.empty((ntpts,nconfigs,nrois))
    norm_array_filt = np.ones((norm_roi_df_array.shape))*np.nan
    config_cond = np.empty((nconfigs,))
    config_img = np.empty((nconfigs,))

    for ridx in range(nrois):
    #ridx = 0
        roi_df_array = np.empty((ntrials,ntpts))
        for tidx in range(ntrials):

            #get trial timecourse
            pix_cell = np.squeeze(pix_cell_array[tidx,:,ridx].squeeze())

            #get baseline
            base_cell = pix_cell[0:pre_frames]

            #calculate df
            df_cell = pix_cell - np.mean(base_cell)

            #store in array
            roi_df_array[tidx,:] = df_cell


        mean_df_array = np.empty((nconfigs,ntpts))
        for cfg_idx in np.unique(trial_config):


            cfg_key = 'config%03d'%(cfg_idx)
            tidx = np.where(trial_config == cfg_idx)[0]
            config_cond[int(cfg_idx)] = trial_cond[tidx][0]
            config_img[int(cfg_idx)] = trial_img[tidx][0]

            mean_df_array[int(cfg_idx),:] = np.nanmean(roi_df_array[tidx,:],0)

        #normalize traces across configs
        norm_roi_df_array[:,:,ridx] = np.transpose(mean_df_array/np.nanmax(mean_df_array.flatten()))
        
        #filter out traces to unresponsive conditions within neuron
        for cfg_idx in np.unique(trial_config):
            cfg_idx = int(cfg_idx)
            if filter_matrix[cfg_idx,ridx]:
                norm_array_filt[:,cfg_idx,ridx] = norm_roi_df_array[:,cfg_idx,ridx]

    #figure out some activity details
    num_active_rois = np.sum(np.sum(filter_matrix,0)>0)
    frac_active_rois = num_active_rois/float(len(cell_rois))
    print('# active rois = %i'%(num_active_rois))
    print('frac active rois = %.04f'%(frac_active_rois))

    active_rois_per_config = np.sum(filter_matrix,1)
    active_rois_per_config = np.expand_dims(active_rois_per_config,1)
    active_rois_per_config_tile = np.matmul(np.ones((ntpts,1)),np.transpose(active_rois_per_config))

    #average across active configs first, then across neuron
    filtered_trace_cond_roi_mean = np.zeros((ntpts,3,nrois))
    active_config = active_rois_per_config>0

    for ridx in range(nrois):
        filtered_traces_tmp = np.array([])
        for cfg_idx in range(0,nconfigs,3):
            if filtered_traces_tmp.size ==0:
                filtered_traces_tmp = norm_array_filt[:,cfg_idx:cfg_idx+3,ridx]
            else:
                filtered_traces_tmp = np.dstack((filtered_traces_tmp ,norm_array_filt[:,cfg_idx:cfg_idx+3,ridx]))
       # print(filtered_traces_tmp.shape)
        filtered_trace_cond_roi_mean[:,:,ridx] = np.nanmean(filtered_traces_tmp,2)

    filtered_trace_cond_mean_neurons = np.nanmean(filtered_trace_cond_roi_mean,2)
    filtered_trace_cond_se_neurons = np.nanstd(filtered_trace_cond_roi_mean,2)/np.sqrt(num_active_rois)

    #put things into pandas df for plotting
    conddfs2 = []
    ylabel = 'Normalized Response'
    #cfg_key = 'config001'
    for cond_count in range(filtered_trace_cond_mean_neurons.shape[1]):
        stim_cond = cond_count
        mean_trace = filtered_trace_cond_mean_neurons[:,cond_count]
        sem_trace = filtered_trace_cond_se_neurons[:,cond_count]
        nreps = num_active_rois
        cdf = pd.DataFrame({'%s' % ylabel: mean_trace,
                            'tsec': frames_tsec,
                            'sem': sem_trace,
                           'fill_minus': mean_trace - sem_trace,
                            'fill_plus': mean_trace + sem_trace,
                            'stim_cond' : [stim_cond for _ in range(len(mean_trace))],
                           'nreps': [nreps for _ in range(len(mean_trace))]
                           })

        conddfs2.append(cdf)
    conddfs2 = pd.concat(conddfs2, axis=0)

    palette = sns.color_palette(["#4c72b0","#c44e52","#55a868"])
    sns.set_palette(palette)

    sns.set_style("ticks")
    trace_labels = ['Original','Corr-Match','sf-Match']

    fig_title = ('Normalized response - avg across active images')
    fig_fn = 'filtered_trace_avg_across_active_cells_thresh_%s_%i.png'%(filter_crit,filter_thresh)
    fig_file_path = os.path.join(fig_out_dir,fig_fn)

    #make figure
    p = sns.FacetGrid(conddfs2, hue='stim_cond', size=10)
    p = p.map(pl.fill_between, "tsec", "fill_minus", "fill_plus", alpha=0.5)
    p = p.map(pl.plot, "tsec", ylabel, lw=1, alpha=1)


    axes = p.ax
    ymin,ymax = axes.get_ylim()
    xmin,xmax = axes.get_xlim()
    start_val = 0.0
    end_val = 1.0 #hard-coding

            #print ri, ci
    p.ax.add_patch(patches.Rectangle((start_val, ymin), end_val, ymax-ymin, linewidth=0, fill=True, color='k', alpha=0.2))
    p.ax.text(-0.999, ymax+(ymax*0), 'n=%i' % num_active_rois, fontsize=10)
    p.ax.axhline(y=0, xmin=xmin, xmax= xmax, linewidth=1, color='k',linestyle = '--')


    #p.ax.set_xticks(())
    #sns.despine(trim=True, offset=0, bottom=True, left=False, ax=p.ax)
    p.ax.set_xlabel('time (s)', fontsize=12)
    p.ax.set_ylabel('%s' % ylabel, fontsize=12)


    pl.legend(bbox_to_anchor=(0, -0.3), loc=2, borderaxespad=0.1, labels=trace_labels, fontsize=8)


    p.savefig(fig_file_path)
    plt.close()


    #average over neurons - per config
    filtered_trace_mean = np.nanmean(norm_array_filt,2)
    filtered_trace_se = np.true_divide(np.nanstd(norm_array_filt,2),np.sqrt(active_rois_per_config_tile))

    #put things into pandas df for plotting
    meandfs = []
    ylabel = 'Normalized Response'
    #cfg_key = 'config001'
    for cfg_count in range(nconfigs):
        cfg_key = 'config%03d'%(cfg_count)
        img = config_img[cfg_count]+1
        stim_cond = config_cond[cfg_count]
        mean_trace = filtered_trace_mean[:,cfg_count]
        sem_trace = filtered_trace_se[:,cfg_count]
        nreps = active_rois_per_config[cfg_count]
        mdf = pd.DataFrame({'%s' % ylabel: mean_trace,
                            'tsec': frames_tsec,
                            'sem': sem_trace,
                           'fill_minus': mean_trace - sem_trace,
                            'fill_plus': mean_trace + sem_trace,
                            'config': [cfg_count for _ in range(len(mean_trace))],
                            'img' : [img for _ in range(len(mean_trace))],
                            'stim_cond' : [stim_cond for _ in range(len(mean_trace))],
                           'nreps': [nreps for _ in range(len(mean_trace))]
                           })

        meandfs.append(mdf)
    meandfs = pd.concat(meandfs, axis=0)

    psth_cols = 5
    sns.set_style("ticks")
    trace_labels = ['Original','Corr-Match','sf-Match']
    fig_title = ('Normalized response per config - avg across active cells')
    fig_fn = 'filtered_trace_per_config_thresh_%s_%i.png'%(filter_crit,filter_thresh)
    fig_file_path = os.path.join(fig_out_dir,fig_fn)

    #make figure
    p = sns.FacetGrid(meandfs, col="img", col_wrap=psth_cols, hue='stim_cond', size=5)
    p = p.map(pl.fill_between, "tsec", "fill_minus", "fill_plus", alpha=0.5)
    p = p.map(pl.plot, "tsec", ylabel, lw=1, alpha=1)
    pl.subplots_adjust(wspace=0.1, hspace=0.8, top=0.85, bottom=0.1, left=0.1)

    axes = p.axes
    ymin,ymax = axes[0].get_ylim()
    xmin,xmax = axes[0].get_xlim()
    start_val = 0.0
    end_val = 1.0 #hard-coding
    for ri in range(p.axes.shape[0]):
            #print ri, ci
            p.axes[ri].add_patch(patches.Rectangle((start_val, ymin), end_val, ymax-ymin, linewidth=0, fill=True, color='k', alpha=0.2))
            p.axes[ri].text(-0.999, ymax+(ymax*0.2), 'n=%i' % active_rois_per_config[ri*3], fontsize=10)
            p.axes[ri].axhline(y=0, xmin=xmin, xmax= xmax, linewidth=1, color='k',linestyle = '--')

            if ri == 0:
              #  p.axes[ri].yaxis.set_major_locator(pl.MaxNLocator(2))
                p.axes[ri].set_xticks(())
                sns.despine(trim=True, offset=0, bottom=True, left=False, ax=p.axes[ri])
                p.axes[ri].set_xlabel('time (s)', fontsize=8)
                p.axes[ri].set_ylabel('%s' % ylabel, fontsize=8)
            else:
                sns.despine(trim=True, offset=0, bottom=True, left=True, ax=p.axes[ri])
                p.axes[ri].tick_params(
                                        axis='both',          # changes apply to the x-axis
                                        which='both',      # both major and minor ticks are affected
                                        bottom='off',      # ticks along the bottom edge are off
                                        left='off',
                                        top='off',         # ticks along the top edge are off
                                        labelbottom='off',
                                        labelleft='off') # labels along the bottom edge are off)
                p.axes[ri].set_xlabel('')
                p.axes[ri].set_ylabel('')
    pl.legend(bbox_to_anchor=(0, -0.3), loc=2, borderaxespad=0.1, labels=trace_labels, fontsize=8)

    p.fig.suptitle(fig_title)
    p.savefig(fig_file_path)
    plt.close()

    #average across images
    active_config = active_rois_per_config>0
    filtered_trace_cond_mean = np.zeros((ntpts,3))
    filtered_traces_tmp = np.array([])
    for cfg_idx in range(0,nconfigs,3):
        if active_config[cfg_idx]:
            if filtered_traces_tmp.size ==0:
                filtered_traces_tmp = filtered_trace_mean[:,cfg_idx:cfg_idx+3]
                print(filtered_traces_tmp.shape)
            else:
                filtered_traces_tmp = np.dstack((filtered_traces_tmp ,filtered_trace_mean[:,cfg_idx:cfg_idx+3]))

    n_active_images = filtered_traces_tmp.shape[2]
    filtered_trace_cond_mean = np.nanmean(filtered_traces_tmp,2)
    filtered_trace_cond_se = np.nanstd(filtered_traces_tmp,2)/np.sqrt(n_active_images)

    #put things into pandas df for plotting
    conddfs = []
    ylabel = 'Normalized Response'
    #cfg_key = 'config001'
    for cond_count in range(filtered_trace_cond_mean.shape[1]):
        stim_cond = cond_count
        mean_trace = filtered_trace_cond_mean[:,cond_count]
        sem_trace = filtered_trace_cond_se[:,cond_count]
        nreps = n_active_images
        cdf = pd.DataFrame({'%s' % ylabel: mean_trace,
                            'tsec': frames_tsec,
                            'sem': sem_trace,
                           'fill_minus': mean_trace - sem_trace,
                            'fill_plus': mean_trace + sem_trace,
                            'config': [cfg_count for _ in range(len(mean_trace))],
                            'img' : [img for _ in range(len(mean_trace))],
                            'stim_cond' : [stim_cond for _ in range(len(mean_trace))],
                           'nreps': [nreps for _ in range(len(mean_trace))]
                           })

        conddfs.append(cdf)
    conddfs = pd.concat(conddfs, axis=0)

    fig_title = ('Normalized response - avg across active images')
    fig_fn = 'filtered_trace_avg_across_images_thresh_%s_%i.png'%(filter_crit,filter_thresh)
    fig_file_path = os.path.join(fig_out_dir,fig_fn)

    #make figure
    p = sns.FacetGrid(conddfs, hue='stim_cond', size=10)
    p = p.map(pl.fill_between, "tsec", "fill_minus", "fill_plus", alpha=0.5)
    p = p.map(pl.plot, "tsec", ylabel, lw=1, alpha=1)


    axes = p.ax
    ymin,ymax = axes.get_ylim()
    xmin,xmax = axes.get_xlim()
    start_val = 0.0
    end_val = 1.0 #hard-coding

            #print ri, ci
    p.ax.add_patch(patches.Rectangle((start_val, ymin), end_val, ymax-ymin, linewidth=0, fill=True, color='k', alpha=0.2))
    p.ax.text(-0.999, ymax+(ymax*0), 'n=%i' % n_active_images, fontsize=10)
    p.ax.axhline(y=0, xmin=xmin, xmax= xmax, linewidth=1, color='k',linestyle = '--')


    #p.ax.set_xticks(())
    #sns.despine(trim=True, offset=0, bottom=True, left=False, ax=p.ax)
    p.ax.set_xlabel('time (s)', fontsize=12)
    p.ax.set_ylabel('%s' % ylabel, fontsize=12)


    pl.legend(bbox_to_anchor=(0, -0.3), loc=2, borderaxespad=0.1, labels=trace_labels, fontsize=8)


    p.savefig(fig_file_path)
    plt.close()

    #save arrays to file
    # Create outfile:
    data_array_fn = 'filtered_norm_df_traces_thresh_%s_%i.hdf5'%(filter_crit, filter_thresh)
    data_array_filepath = os.path.join(traceid_dir, 'data_arrays', data_array_fn)
    data_grp = h5py.File(data_array_filepath, 'w')

    data_grp.attrs['frames_tsec'] = frames_tsec
    data_grp.attrs['s2p_cell_rois'] = cell_rois
    data_grp.attrs['frames_tsec'] = frames_tsec
    data_grp.attrs['nconfigs'] = nconfigs
    data_grp.attrs['nrois'] = nrois


    img_dset = data_grp.create_dataset('config_img', config_img.shape, config_img.dtype)
    img_dset[...] = config_img

    cond_dset = data_grp.create_dataset('config_cond',config_img.shape, config_img.dtype)
    cond_dset[...] = config_cond

    nrois_dset = data_grp.create_dataset('/'.join([curr_slice, 'n_active_rois' ]), num_active_rois.shape, num_active_rois.dtype)
    nrois_dset[...] = num_active_rois

    frois_dset = data_grp.create_dataset('/'.join([curr_slice, 'frac_active_rois' ]), frac_active_rois.shape, frac_active_rois.dtype)
    frois_dset[...] = frac_active_rois

    passrois_dset = data_grp.create_dataset('/'.join([curr_slice, 'active_rois_per_config' ]), active_rois_per_config.shape, active_rois_per_config.dtype)
    passrois_dset[...] = active_rois_per_config



    filt_trace_cfg_mean_neu_dset = data_grp.create_dataset('/'.join([curr_slice, 'filtered_trace_per_cond_mean_across_neurons' ]), filtered_trace_cond_mean_neurons.shape, filtered_trace_cond_mean_neurons.dtype)
    filt_trace_cfg_mean_neu_dset[...] = filtered_trace_cond_mean_neurons

    filt_trace_cfg_se_neu_dset = data_grp.create_dataset('/'.join([curr_slice, 'filtered_trace_per_cond_se_across_neurons' ]), filtered_trace_cond_se_neurons.shape, filtered_trace_cond_se_neurons.dtype)
    filt_trace_cfg_se_neu_dset[...] = filtered_trace_cond_se_neurons

    filt_trace_cond_mean_dset = data_grp.create_dataset('/'.join([curr_slice, 'filtered_trace_per_cond_per_neuron_mean_across_images' ]), filtered_trace_cond_roi_mean.shape, filtered_trace_cond_roi_mean.dtype)
    filt_trace_cond_mean_dset[...] = filtered_trace_cond_roi_mean

    filt_trace_cfg_mean_cfg_dset = data_grp.create_dataset('/'.join([curr_slice, 'filtered_trace_mean_per_config_acrosss_neurons' ]), filtered_trace_mean.shape, filtered_trace_mean.dtype)
    filt_trace_cfg_mean_cfg_dset[...] = filtered_trace_mean

    filt_trace_cfg_neu_cfg_dset = data_grp.create_dataset('/'.join([curr_slice, 'filtered_trace_mean_per_config_per_neuron' ]), norm_array_filt.shape, norm_array_filt.dtype)
    filt_trace_cfg_neu_cfg_dset[...] = norm_array_filt

    filt_trace_cfg_se_cfg_dset = data_grp.create_dataset('/'.join([curr_slice, 'filtered_trace_se_per_config_across_neurons' ]), filtered_trace_se.shape, filtered_trace_se.dtype)
    filt_trace_cfg_se_cfg_dset[...] = filtered_trace_se

    filt_trace_cfg_mean_img_dset = data_grp.create_dataset('/'.join([curr_slice, 'filtered_trace_per_cond_mean_across_images' ]), filtered_trace_cond_mean.shape, filtered_trace_cond_mean.dtype)
    filt_trace_cfg_mean_img_dset[...] = filtered_trace_cond_mean

    filt_trace_cfg_se_img_dset = data_grp.create_dataset('/'.join([curr_slice, 'filtered_trace_per_cond_se_across_images' ]), filtered_trace_cond_se.shape, filtered_trace_cond_se.dtype)
    filt_trace_cfg_se_img_dset[...] = filtered_trace_cond_se



    data_grp.close()


def extract_options(options):
    parser = optparse.OptionParser()

    # PATH opts:
    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/n/coxfs01/2p-data', help='source dir (root project dir containing all expts) [default: /n/coxfs01/2p-data]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')
    parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID') 
    parser.add_option('-A', '--acq', action='store', dest='acquisition', default='', help="acquisition folder (ex: 'FOV1_zoom3x')")
    parser.add_option('-T', '--traceid', action='store', dest='traceid', default='', help="(ex: traces001_s2p)")
    parser.add_option('-C', '--combined_run', action='store', dest='combined_run', default='', help='name of combo run') 
    parser.add_option('-f', '--filter_crit', action='store', dest='filter_crit', default='zscore', help='criterion to filter traces e.g.zscore') 
    parser.add_option('-t', '--filter_thresh', action='store', dest='filter_thresh', default='zscore', help='cutoff value of filter criterion') 
    parser.add_option('-m', '--motion_thresh', action='store', dest='motion_thresh', default='5', help='threshold for motion to exclude trials') 
    (options, args) = parser.parse_args() 

    return options




#-----------------------------------------------------
#           MAIN SET OF ACTIONS
#-----------------------------------------------------

def main(options): 
    
    options = extract_options(options)

    print('----- filterting traces -----')
    filter_traces(options)


    
#%%

if __name__ == '__main__':
    main(sys.argv[1:])
