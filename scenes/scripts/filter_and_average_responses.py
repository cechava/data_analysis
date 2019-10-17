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

def filter_responses(opts):
    response_type = opts.response_type
    print(response_type)
    filter_crit = opts.filter_crit
    filter_thresh = int(opts.filter_thresh)

    traceid = '%s_s2p'%(opts.traceid)

        #% Set up paths:    
    acquisition_dir = os.path.join(opts.rootdir, opts.animalid, opts.session, opts.acquisition)
    traceid_dir = os.path.join(acquisition_dir, opts.combined_run,'traces',traceid)

    run_dir = traceid_dir.split('/traces')[0]
    trace_arrays_dir = os.path.join(traceid_dir,'files')
    paradigm_dir = os.path.join(acquisition_dir, opts.combined_run, 'paradigm')


    filt_responses_dir = os.path.join(acquisition_dir, opts.combined_run,'filtered_responses',traceid)

        


    if 'norm' in response_type:
        i1 = findOccurrences(response_type,'_')[-1]
        fetch_data = response_type[i1+1:]
    else:
        fetch_data = response_type
    print(fetch_data)

    #get file with trial-by-trial responses
    responses_dir = os.path.join(acquisition_dir, opts.combined_run,'responses',traceid)
    data_array_dir = os.path.join(responses_dir, 'data_arrays')

    resp_array_fn = 'trial_response_array_motion_%s.hdf5'%(opts.motion_thresh)
    resp_array_filepath = os.path.join(data_array_dir, resp_array_fn)
    resp_grp = h5py.File(resp_array_filepath, 'r')

    if 's2p_cell_rois' in resp_grp.attrs.keys():
        cell_rois = resp_grp.attrs['s2p_cell_rois']
    else:
        cell_rois = np.arange(nrois)

    curr_slice = 'Slice01'#hard,coding for now

    #unpack
    response_matrix = np.array(resp_grp['/'.join([curr_slice, 'responses' ,fetch_data])])

        
    if filter_crit == 'zscore':
        filter_crit_matrix_trials = np.array(resp_grp['/'.join([curr_slice, 'responses' ,filter_crit])])
        filter_crit_matrix_trials = filter_crit_matrix_trials[:,:,cell_rois]
        filter_crit_matrix = np.squeeze(np.mean(filter_crit_matrix_trials,0))
    elif filter_crit == 'simple_tstat':
        filter_crit_matrix = np.array(resp_grp['/'.join([curr_slice, 'responses' ,filter_crit])])

    #considering only cell rois
    response_matrix = response_matrix[:,:,cell_rois]
    ntrials,nconfigs,nrois = response_matrix.shape
    filter_crit_matrix = filter_crit_matrix[:,cell_rois]

    resp_grp.close()


    #consider a config active if at least one of theversion of an image evoked a response above threshol
    thresh = filter_thresh
    thresh_matrix = filter_crit_matrix>thresh
    negative_response = filter_crit_matrix<=0#let's ignore negative responses b/c they are hard to interpret
    filter_matrix = np.ones((thresh_matrix.shape))*np.nan
    active_rois_per_config = np.nansum(thresh_matrix,1)
    for ridx in range(nrois):
        for idx in range(0,thresh_matrix.shape[0],3):
                if np.sum(thresh_matrix[idx:idx+3,ridx])>0 and np.sum(negative_response[idx:idx+3,ridx])==0:
                    filter_matrix[idx:idx+3,ridx] = 1
    active_rois_per_img = np.nansum(filter_matrix,1)
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

    config_cond = np.empty((nconfigs,))
    config_img = np.empty((nconfigs,))


    for cfg_idx in np.unique(trial_config):
        tidx = np.where(trial_config == cfg_idx)[0]
        config_cond[int(cfg_idx)] = trial_cond[tidx][0]
        config_img[int(cfg_idx)] = trial_img[tidx][0]

    mean_response_matrix = np.mean(response_matrix,0)
    if 'norm' in response_type:
        norm_response_array = np.empty((nconfigs,nrois))

        for ridx in range(nrois):
            norm_response_array[:,ridx] = mean_response_matrix[:,ridx]/np.nanmax(mean_response_matrix[:,ridx])
        mean_response_matrix = norm_response_array
        
    filt_response_array = mean_response_matrix*filter_matrix


    #figure out some activity details
    active_cell_idx = np.nansum(filter_matrix,0)>0
    num_active_rois = np.nansum(np.nansum(filter_matrix,0)>0)
    frac_active_rois = num_active_rois/float(len(cell_rois))
    print('# active rois = %i'%(num_active_rois))
    print('frac active rois = %.04f'%(frac_active_rois))


    resp_diff = np.ones((30,filt_response_array.shape[1]))*np.nan
    mod_idx = np.ones((30,filt_response_array.shape[1]))*np.nan


    for img_count, cond_idx in enumerate(range(0,nconfigs,3)):
        if filter_matrix[cond_idx,ridx]:
            #original>medium complex
            resp_diff[2*img_count,:] = filt_response_array[cond_idx,:] - filt_response_array[cond_idx+1,:]
            mod_idx[2*img_count,:] = np.true_divide(filt_response_array[cond_idx,:] - filt_response_array[cond_idx+1,:],
                                               filt_response_array[cond_idx,:] + filt_response_array[cond_idx+1,:])       
           
           #medium>low complex
            resp_diff[(2*img_count)+1,:] = filt_response_array[cond_idx+1,:] - filt_response_array[cond_idx+2,:]
            mod_idx[(2*img_count)+1,:] = np.true_divide(filt_response_array[cond_idx+1,:] - filt_response_array[cond_idx+2,:],
                                               filt_response_array[cond_idx+1,:] + filt_response_array[cond_idx+2,:]) 


    #avg config across neurons
    img_response_mean = np.nanmean(filt_response_array,1)
    img_response_se = np.true_divide(np.nanstd(filt_response_array,1),np.sqrt(active_rois_per_img))


    folder_name = 'motion_%s_%s_%i'%(opts.motion_thresh,filter_crit,filter_thresh)

    fig_out_dir = os.path.join(filt_responses_dir,'figures',folder_name)
    if not os.path.exists(fig_out_dir):
        os.makedirs(fig_out_dir)
    print(fig_out_dir)

    #plot mean config across neurons
    bar_loc = np.zeros((img_response_mean.size))
    width = 0.4         # the width of the bars
    gap = .5
    xloc = 1
    count = 0
    for i in range(len(np.unique(trial_img))):
        for j in range(len(np.unique(trial_cond))):
            bar_loc[count] = xloc
            xloc = xloc + width
            count = count+1
        xloc = xloc + gap


    fig = plt.figure(figsize=(20,5))
    plt.bar(bar_loc[0:len(bar_loc):3],img_response_mean[0:len(bar_loc):3],width,color = '#4c72b0',yerr = img_response_se[0:len(bar_loc):3])
    plt.bar(bar_loc[1:len(bar_loc):3],img_response_mean[1:len(bar_loc):3],width,color = '#c44e52',yerr = img_response_se[1:len(bar_loc):3])
    plt.bar(bar_loc[2:len(bar_loc):3],img_response_mean[2:len(bar_loc):3],width,color = '#55a868',yerr = img_response_se[2:len(bar_loc):3])

    axes = plt.gca()
    xmin, xmax = axes.get_xlim()
    ymin, ymax = axes.get_ylim()
    plt.axhline(y=0, xmin=xmin, xmax= xmax, linewidth=1, color='k',linestyle = '-')

    xtick_loc = bar_loc[1:len(bar_loc):3]
    xtick_label = np.unique(trial_img+1).astype('int')

    plt.xticks(xtick_loc,xtick_label.tolist())
    plt.xlabel('Image',fontsize = 15)
    plt.ylabel('Average Response',fontsize = 15)
    plt.suptitle('Average %s Across Neurons'%(response_type),fontsize = 15)

    count = 0
    for idx in bar_loc[1:len(bar_loc):3]:
        plt.text(idx-.25, ymax, 'n=%i' % active_rois_per_img[count], fontsize=10)
        count = count +3
        
    fig_fn = 'avg_across_neurons_%s_thresh_%s_%i.png'%(response_type,filter_crit,filter_thresh)
    fig_file_path = os.path.join(fig_out_dir, fig_fn)
    plt.savefig(fig_file_path)
    plt.close()

    #plot number of response neurons per config
    bar_loc = np.zeros((img_response_mean.size))
    width = 0.4         # the width of the bars
    gap = .5
    xloc = 1
    count = 0
    for i in range(len(np.unique(trial_img))):
        for j in range(len(np.unique(trial_cond))):
            bar_loc[count] = xloc
            xloc = xloc + width
            count = count+1
        xloc = xloc + gap


    fig = plt.figure(figsize=(20,5))
    plt.bar(bar_loc[0:len(bar_loc):3],active_rois_per_config[0:len(bar_loc):3],width,color = '#4c72b0')
    plt.bar(bar_loc[1:len(bar_loc):3],active_rois_per_config[1:len(bar_loc):3],width,color = '#c44e52')
    plt.bar(bar_loc[2:len(bar_loc):3],active_rois_per_config[2:len(bar_loc):3],width,color = '#55a868')

    axes = plt.gca()
    xmin, xmax = axes.get_xlim()
    ymin, ymax = axes.get_ylim()
    plt.axhline(y=0, xmin=xmin, xmax= xmax, linewidth=1, color='k',linestyle = '-')

    xtick_loc = bar_loc[1:len(bar_loc):3]
    xtick_label = np.unique(trial_img+1).astype('int')

    plt.xticks(xtick_loc,xtick_label.tolist())
    plt.xlabel('Image',fontsize = 15)
    plt.ylabel('Number of cells',fontsize = 15)
    plt.suptitle('Number of responseive cells per stim',fontsize = 15)

    count = 0
    for idx in bar_loc[1:len(bar_loc):3]:
        plt.text(idx-.25, ymax, 'n=%i' % active_rois_per_img[count], fontsize=10)
        count = count +3
        
    fig_fn = 'active_neurons_%s_thresh_%s_%i.png'%(response_type,filter_crit,filter_thresh)
    fig_file_path = os.path.join(fig_out_dir, fig_fn)
    plt.savefig(fig_file_path)
    plt.close()

    #averaging difference/ modulation index across active neurons (1 active condition) - group by image
    tmp = np.copy(active_rois_per_img)
    tmp = np.delete(tmp,np.arange(2,nconfigs,3))
    mean_mod_idx = np.nanmean(mod_idx,1)
    se_mod_idx = np.true_divide(np.nanstd(mod_idx,1),np.sqrt(tmp))

    mean_diff = np.nanmean(resp_diff,1)
    se_diff = np.true_divide(np.nanstd(resp_diff,1),np.sqrt(tmp))


    #plot 
    bar_loc = np.zeros((len(np.unique(config_img))*2))
    width = 0.4         # the width of the bars
    gap = .5
    xloc = 1
    count = 0
    for i in range(15):
        for j in range(2):
            bar_loc[count] = xloc
            xloc = xloc + width
            count = count+1
        xloc = xloc + gap


    fig = plt.figure(figsize=(20,5))
    plt.bar(bar_loc[0:len(bar_loc):2],mean_mod_idx[0:mean_mod_idx.size:2],width,color = 'c',yerr=se_mod_idx[0:mean_mod_idx.size:2])
    plt.bar(bar_loc[1:len(bar_loc):2],mean_mod_idx[1:mean_mod_idx.size:2],width,color = 'm',yerr=se_mod_idx[1:mean_mod_idx.size:2])

    plt.legend(['Orig>Med','Med>Low'])

    axes = plt.gca()
    xmin, xmax = axes.get_xlim()
    ymin, ymax = axes.get_ylim()
    plt.axhline(y=0, xmin=xmin, xmax= xmax, linewidth=1, color='k',linestyle = '-')

    xtick_loc = bar_loc[1:len(bar_loc):2]
    xtick_label = np.unique(config_img+1).astype('int')

    plt.xticks(xtick_loc,xtick_label.tolist())
    plt.xlabel('Image',fontsize = 15)
    plt.ylabel('Modulation Index',fontsize = 15)
    fig_fn = 'mod_idx_%s_thresh_%s_%i.png'%(response_type,filter_crit,filter_thresh)
    fig_file_path = os.path.join(fig_out_dir, fig_fn)
    plt.savefig(fig_file_path)
    plt.close

    fig = plt.figure(figsize=(20,5))
    plt.bar(bar_loc[0:len(bar_loc):2],mean_diff[0:mean_mod_idx.size:2],width,color = 'c',yerr=se_diff[0:mean_diff.size:2])
    plt.bar(bar_loc[1:len(bar_loc):2],mean_diff[1:mean_mod_idx.size:2],width,color = 'm',yerr=se_diff[1:mean_diff.size:2])

    plt.legend(['Orig>Med','Med>Low'])

    axes = plt.gca()
    xmin, xmax = axes.get_xlim()
    ymin, ymax = axes.get_ylim()
    plt.axhline(y=0, xmin=xmin, xmax= xmax, linewidth=1, color='k',linestyle = '-')

    xtick_loc = bar_loc[1:len(bar_loc):2]
    xtick_label = np.unique(config_img+1).astype('int')

    plt.xticks(xtick_loc,xtick_label.tolist())
    plt.xlabel('Image',fontsize = 15)
    plt.ylabel('Difference',fontsize = 15)
    fig_fn = 'diff_%s_thresh_%s_%i.png'%(response_type,filter_crit,filter_thresh)
    fig_file_path = os.path.join(fig_out_dir, fig_fn)
    plt.savefig(fig_file_path)
    plt.close


    #average over images
    filter_response_cond_per_neuron = np.zeros((3,nrois))
    for cidx in range(3):
        filter_response_cond_per_neuron[cidx,:] = np.nanmean(filt_response_array[cidx:nconfigs:3,:],0)
     


    filter_response_cond_mean_neuron = np.nanmean(filter_response_cond_per_neuron,1)
    filter_response_cond_se_neuron = np.nanstd(filter_response_cond_per_neuron,1)/np.sqrt(num_active_rois)

    cond_labels = ['Original','Medium Complex','Low Complex']

    bar_loc = np.zeros((3,))
    width = 0.4         # the width of the bars
    xloc = 1
    count = 0

    for j in range(len(np.unique(trial_cond))):
        bar_loc[count] = xloc
        xloc = xloc + width
        count = count+1

    fig = plt.figure(figsize=(8,8))
    plt.bar(bar_loc[0],filter_response_cond_mean_neuron[0],width,color = '#4c72b0',yerr = filter_response_cond_se_neuron[0])
    plt.bar(bar_loc[1],filter_response_cond_mean_neuron[1],width,color = '#c44e52',yerr = filter_response_cond_se_neuron[1])
    plt.bar(bar_loc[2],filter_response_cond_mean_neuron[2],width,color = '#55a868',yerr = filter_response_cond_se_neuron[2])

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


    plt.text(bar_loc[0]-.25, ymax, 'n=%i, f=%.04f' % (num_active_rois, frac_active_rois), fontsize=10)


    fig_fn = 'avg_response_per_cond_across_neurons_%s_thresh_%s_%i.png'%(response_type,filter_crit,filter_thresh)

    fig_file_path = os.path.join(fig_out_dir, fig_fn)
    plt.savefig(fig_file_path)
    plt.close()

    ylabel = 'Average Response'

    resp_dfs = []
    for cidx in range(filter_response_cond_per_neuron.shape[0]):
        response = filter_response_cond_per_neuron[cidx,active_cell_idx]
        cell = np.arange(num_active_rois)
        cond = np.ones((num_active_rois,))*cidx
        mdf = pd.DataFrame({'%s' % ylabel: response,
                            'response': response,
                            'cell': cell,
                            'cond': cond,
                           })

        resp_dfs.append(mdf)
    resp_dfs = pd.concat(resp_dfs, axis=0)

    bar_loc = np.arange(0,3)
    width = 0.5

    palette = sns.color_palette(["#4c72b0","#c44e52","#55a868"])
    sns.set_palette(palette)

    p = sns.catplot(x='cond', y='response', kind="strip", hue = 'cond',data=resp_dfs,size = 10);

    axes = p.ax
    ymin,ymax = axes.get_ylim()
    xmin,xmax = axes.get_xlim()

    for idx in range(len(np.unique(trial_cond))):
        p.ax.hlines(y = filter_response_cond_mean_neuron[idx], xmin=bar_loc[idx]-(width/2), xmax = bar_loc[idx]+(width/2), linewidth=2, color='k',linestyle = '-')
        p.ax.hlines(y = filter_response_cond_mean_neuron[idx] + filter_response_cond_se_neuron[idx], xmin=bar_loc[idx]-(width/2), xmax = bar_loc[idx]+(width/2), linewidth=1, color='k',linestyle = '--')
        p.ax.hlines(y = filter_response_cond_mean_neuron[idx] - filter_response_cond_se_neuron[idx], xmin=bar_loc[idx]-(width/2), xmax = bar_loc[idx]+(width/2), linewidth=1, color='k',linestyle = '--')




    p.ax.set_xticks(())
    p.ax.set_xlabel('Condition',fontsize = 15)
    p.ax.set_ylabel('Average Response',fontsize = 15)
    p.fig.suptitle('Average %s Across Neurons'%(response_type),fontsize = 15)


    p.ax.text(bar_loc[0]-.25, ymax, 'n=%i, f=%.04f' % (num_active_rois, frac_active_rois), fontsize=10)


    fig_fn = 'avg_response_per_cond_scatter_neurons_%s_thresh_%s_%i.png'%(response_type,filter_crit,filter_thresh)

    fig_file_path = os.path.join(fig_out_dir, fig_fn)
    plt.savefig(fig_file_path)

    plt.close()


    #save arrays to file
    # Create outfile:
    data_array_fn = 'filtered_%s_responses_thresh_%s_%i.hdf5'%(response_type,filter_crit, filter_thresh)
    data_array_filepath = os.path.join(data_array_dir, data_array_fn)
    data_grp = h5py.File(data_array_filepath, 'w')

    data_grp.attrs['s2p_cell_rois'] = cell_rois

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

    act_idx_dset = data_grp.create_dataset('/'.join([curr_slice, 'active_cell_idx' ]), active_cell_idx.shape, active_cell_idx.dtype)
    act_idx_dset[...] = active_cell_idx

    passrois_dset = data_grp.create_dataset('/'.join([curr_slice, 'active_rois_per_config' ]), active_rois_per_config.shape, active_rois_per_config.dtype)
    passrois_dset[...] = active_rois_per_config

    passrois_img_dset = data_grp.create_dataset('/'.join([curr_slice, 'active_rois_per_img' ]), active_rois_per_img.shape, active_rois_per_img.dtype)
    passrois_img_dset[...] = active_rois_per_img

    unfilt_cfg_neu_dset = data_grp.create_dataset('/'.join([curr_slice, 'unfiltered_response_per_cfg_per_neuron' ]), mean_response_matrix.shape, mean_response_matrix.dtype)
    unfilt_cfg_neu_dset[...] = mean_response_matrix

    filt_trace_cfg_neu_dset = data_grp.create_dataset('/'.join([curr_slice, 'filtered_response_per_cfg_per_neuron' ]), filt_response_array.shape, filt_response_array.dtype)
    filt_trace_cfg_neu_dset[...] = filt_response_array

    filt_trace_cond_neu_dset = data_grp.create_dataset('/'.join([curr_slice, 'filtered_response_per_cond_per_neuron' ]), filter_response_cond_per_neuron.shape, filter_response_cond_per_neuron.dtype)
    filt_trace_cond_neu_dset[...] = filter_response_cond_per_neuron
    print(filter_response_cond_per_neuron.shape)

    filt_trace_cond_mean_neu_dset = data_grp.create_dataset('/'.join([curr_slice, 'filtered_response_per_cond_mean_across_neurons' ]), filter_response_cond_mean_neuron.shape, filter_response_cond_mean_neuron.dtype)
    filt_trace_cond_mean_neu_dset[...] = filter_response_cond_mean_neuron

    filt_trace_cond_se_neu_dset = data_grp.create_dataset('/'.join([curr_slice, 'filtered_response_per_cond_se_across_neurons' ]), filter_response_cond_se_neuron.shape, filter_response_cond_se_neuron.dtype)
    filt_trace_cond_se_neu_dset[...] = filter_response_cond_se_neuron

    filt_trace_cfg_mean_neu_dset = data_grp.create_dataset('/'.join([curr_slice, 'filtered_response_per_cfg_mean_across_neurons' ]), img_response_mean.shape, img_response_mean.dtype)
    filt_trace_cfg_mean_neu_dset[...] = img_response_mean

    filt_trace_cfg_se_neu_dset = data_grp.create_dataset('/'.join([curr_slice, 'filtered_response_per_cfg_se_across_neurons' ]), img_response_se.shape, img_response_se.dtype)
    filt_trace_cfg_se_neu_dset[...] = img_response_se

    midx_dset = data_grp.create_dataset('/'.join([curr_slice, 'mod_idx_per_neuron' ]), mod_idx.shape, mod_idx.dtype)
    midx_dset[...] = mod_idx

    midx_mean_dset = data_grp.create_dataset('/'.join([curr_slice, 'mod_idx_mean_across_neurons' ]), mean_mod_idx.shape, mean_mod_idx.dtype)
    midx_mean_dset[...] = mean_mod_idx

    mdif_dset = data_grp.create_dataset('/'.join([curr_slice, 'diff_mean_across_neurons' ]), mean_diff.shape, mean_diff.dtype)
    mdif_dset[...] = mean_diff

    midx_se_dset = data_grp.create_dataset('/'.join([curr_slice, 'mod_idx_se_across_neurons' ]), se_mod_idx.shape, se_mod_idx.dtype)
    midx_se_dset[...] = se_mod_idx

    mdif_se_dset = data_grp.create_dataset('/'.join([curr_slice, 'diff_se_across_neurons' ]), se_diff.shape, se_diff.dtype)
    mdif_se_dset[...] = se_diff


    data_grp.close()




def extract_options(options):
    parser = optparse.OptionParser()

    # PATH opts:
    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/n/coxfs01/2p-data', help='source dir (root project dir containing all expts) [default: /n/coxfs01/2p-data]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')
    parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID') 
    parser.add_option('-A', '--acq', action='store', dest='acquisition', default='', help="acquisition folder (ex: 'FOV1_zoom3x')")
    parser.add_option('-T', '--traceid', action='store', dest='traceid', default='traces001', help="(ex: traces001_s2p)")
    parser.add_option('-C', '--combined_run', action='store', dest='combined_run', default='', help='name of combo run') 
    parser.add_option('-f', '--filter_crit', action='store', dest='filter_crit', default='zscore', help='criterion to filter traces e.g.zscore') 
    parser.add_option('-t', '--filter_thresh', action='store', dest='filter_thresh', default='zscore', help='cutoff value of filter criterion') 
    parser.add_option('-d', '--response_type', action='store', dest='response_type', default='norm_df', help='response type') 
    parser.add_option('-m', '--motion_thresh', action='store', dest='motion_thresh', default='5', help='threshold for motion to exclude trials') 
    (options, args) = parser.parse_args() 

    return options




#-----------------------------------------------------
#           MAIN SET OF ACTIONS
#-----------------------------------------------------

def main(options): 
    
    options = extract_options(options)

    print('----- filterting responses-----')
    filter_responses(options)


    
#%%

if __name__ == '__main__':
    main(sys.argv[1:])
