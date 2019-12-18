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
sys.path.append('/n/coxfs01/cechavarria/repos/2p-pipeline/')
from pipeline.python.paradigm import utils as util

def make_mean_psth_scenes(meandfs,psth_cols,fig_title,fig_file_path):

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
                p.axes[ri].text(-0.999, ymax+(ymax*0.2), 'n=%i' % int(nreps), fontsize=6)
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

def make_trials_psth_scenes(alldfs,psth_cols,fig_title,fig_file_path):
    p = sns.relplot(x="tsec", y=ylabel,units = 'trial',estimator = None,hue = 'stim_cond', col="img", col_wrap=psth_cols,
                kind="line", palette=["#4c72b0","#c44e52","#55a868"],size = 5,alpha = 0.6,data = alldfs,legend=False)


    plt.subplots_adjust(wspace=0.1, hspace=0.8, top=0.85, bottom=0.1, left=0.1)

    axes = p.axes
    ymin,ymax = axes[0].get_ylim()
    xmin,xmax = axes[0].get_xlim()
    start_val = 0.0
    end_val = 1.0 #hard-coding
    for ri in range(p.axes.shape[0]):
            #print ri, ci
            p.axes[ri].add_patch(patches.Rectangle((start_val, ymin), end_val, ymax-ymin, linewidth=0, fill=True, color='k', alpha=0.2))
            p.axes[ri].text(-0.999, ymax+(ymax*0.2), 'n=%i' % int(alldfs['nreps'].values[0]), fontsize=6)
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



    # Create the legend patches
    legend_patches = [matplotlib.patches.Patch(color=C, label=L) for
                      C, L in zip(["#4c72b0","#c44e52","#55a868"],
                                  trace_labels)]

    # # Plot the legend
    plt.legend(handles=legend_patches,bbox_to_anchor=(0, -0.0), loc=2, borderaxespad=0.1, fontsize=8)
    
    p.fig.suptitle(fig_title)

    p.savefig(fig_file_path)
    plt.close()

class struct: pass
opts = struct()
opts.rootdir = '/n/coxfs01/2p-data'
opts.animalid = 'JC113'
opts.session = '20191021'
opts.acquisition = 'FOV1_zoom4p0x'
opts.traceid = 'traces102'
opts.run = 'scenes_combined'
opts.motion_thresh = 10

#threhsold opitons
# filter_crit = None
# filter_thresh = 4


if opts.motion_thresh is not None:
    motion_thresh = int(opts.motion_thresh)

#figures options
trace_labels = ['Original','Medium Complex','Low Complex']
data_type = 'cell'
trace_type = ['global_df_f','delta_spks','zscore_df','zscore_spks','df']
type_title = ['df/f','spks_response','z-score_df','zscore_spks','df']
# trace_type = ['delta_spks']
# type_title = ['delta_spks']
# trace_type = ['df_f']
# type_title = ['df/f']
psth_cols = 5
sns.set_style("ticks")
palette = sns.color_palette(["#4c72b0","#c44e52","#55a868"])
sns.set_palette(palette)

curr_slice = 'Slice01'#hard,coding for now

traceid = '%s_s2p'%(opts.traceid)

#% Set up paths:    
acquisition_dir = os.path.join(opts.rootdir, opts.animalid, opts.session, opts.acquisition)
if 'combined' in opts.run:
    traceid_dir = os.path.join(acquisition_dir, opts.run,'traces',traceid)
else:
    traceid_dir = util.get_traceid_from_acquisition(acquisition_dir, opts.run, traceid)
    
run_dir = traceid_dir.split('/traces')[0]
trace_arrays_dir = os.path.join(traceid_dir,'files')
responses_file_dir = paradigm_dir = os.path.join(run_dir, 'responses',traceid,'files')

#open response file to read
resp_array_fn = 'trial_response_array.hdf5'
resp_array_filepath = os.path.join(responses_file_dir, resp_array_fn)
print('Reading: %s'%(resp_array_filepath))
resp_grp = h5py.File(resp_array_filepath, 'r')

#unpack
trial_config = np.array(resp_grp['trial_config'])
trial_motion = np.array(resp_grp['trial_max_motion'])

#apply motion threshold
trial_filter0 = trial_motion<motion_thresh

#group trials by config and match

nconfigs = np.unique(trial_config).size
ntrials = int(trial_filter0.size/nconfigs)

trial_filter_cfgs = np.empty((nconfigs,ntrials))

for cfg in range(nconfigs):
    tidx = np.where(trial_config == cfg)[0]
    trial_filter_cfgs[cfg,:] = trial_filter0[tidx]



#count number of non-excluded trials
good_trial_count = np.sum(trial_filter_cfgs,1)
#minimum is how many trials we will mach
trials_to_keep = int(np.nanmin(good_trial_count))

#match trials
trial_filter = np.zeros(trial_filter_cfgs.shape)
trial_filter[good_trial_count==trials_to_keep,:] = trial_filter_cfgs[good_trial_count==trials_to_keep,:]

for cfg in np.where(good_trial_count>trials_to_keep)[0]:

    trial_filter[cfg,np.where(trial_filter_cfgs[cfg,:]>0)[0][0:trials_to_keep]]=1

#trial_filter[good_trial_count>trials_to_keep,0:trials_to_keep] = 1
trial_filter = trial_filter.astype('bool')

print('**Matching all configs to have %i trials**'%(trials_to_keep))



#final trial filter will be across tirals
trial_filter1=np.empty(trial_filter0.shape)
for cfg in range(nconfigs):
    tidx = np.where(trial_config == cfg)[0]
    trial_filter1[tidx] = trial_filter[cfg,:]


#get  img and condition for each config
trial_cond = np.array(resp_grp['trial_cond'])
trial_img = np.array(resp_grp['trial_img'])

cfg_img = np.zeros((nconfigs,))
cfg_cond = np.zeros((nconfigs,))
for cfg in range(nconfigs):
    tidx = np.where(trial_config == cfg)[0][0]
    cfg_cond[cfg] = trial_cond[tidx]
    cfg_img[cfg] = trial_img[tidx]

#open trace file to read
data_array_fn = 'processed_traces.hdf5'
data_array_filepath = os.path.join(traceid_dir, 'files', data_array_fn)
data_grp = h5py.File(data_array_filepath, 'r')

frames_tsec = data_grp.attrs['frames_tsec']
nrois = data_grp.attrs['nrois']
print('ROIs:%i'%(nrois))

if 's2p_cell_rois' in data_grp.attrs.keys():
    cell_rois = data_grp.attrs['s2p_cell_rois']
else:
    cell_rois = np.arange(nrois)

# #type_idx = 0
# #t_type = trace_type[type_idx]
print('---- TRIAL-AVERAGED RESPONSES ----------')
for type_idx,t_type in enumerate(trace_type):
    print('type: %s'%(t_type))

    
    #unpack traces
    trace = np.array(data_grp[curr_slice]['traces'][t_type][data_type])

    #filter
    trace[np.where(trial_filter1==0)[0],:,:] = np.nan

    #get mean trace and std error
    trace_mean = np.empty((nconfigs,trace.shape[1],nrois)) 
    trace_se = np.empty((nconfigs,trace.shape[1],nrois)) 
    for cfg in range(nconfigs):
        tidx = np.where(trial_config == cfg)[0]
        trace_mean[cfg,:,:] = np.nanmean(trace[tidx,:,:],0)
        trace_se[cfg,:,:] = np.nanstd(trace[tidx,:,:],0)/np.sqrt(trials_to_keep)


    if motion_thresh is not None:
        fig_out_dir = os.path.join(traceid_dir,'figures','mean_trace','motion_%i'%(motion_thresh),data_type,t_type)
    else:
        fig_out_dir = os.path.join(traceid_dir,'figures','mean_trace',data_type,t_type)
    if not os.path.exists(fig_out_dir):
        os.makedirs(fig_out_dir)

    #go through rois
    #ridx = 2
    for ridx in cell_rois:
        print(ridx)

        meandfs = []
        ylabel = type_title[type_idx]
        #cfg_key = 'config001'
        for cfg in range(nconfigs):
            img = cfg_img[cfg]
            stim_cond = cfg_cond[cfg]
            mean_trace = trace_mean[cfg,:,ridx]
            sem_trace = trace_se[cfg,:,ridx]
            nreps = trials_to_keep
            mdf = pd.DataFrame({'%s' % ylabel: mean_trace,
                                'tsec': frames_tsec,
                                'sem': sem_trace,
                                'fill_minus': mean_trace - sem_trace,
                                'fill_plus': mean_trace + sem_trace,
                                'config': [cfg for _ in range(len(mean_trace))],
                                'img' : [img for _ in range(len(mean_trace))],
                                'stim_cond' : [stim_cond for _ in range(len(mean_trace))],
                                'nreps': [nreps for _ in range(len(mean_trace))]
                               })

            meandfs.append(mdf)
        meandfs = pd.concat(meandfs, axis=0)

        fig_title = ('roi %05d' % (int(ridx)))
        fig_fn = 'roi%05d_mean_trace.png'%ridx
        fig_file_path = os.path.join(fig_out_dir,fig_fn)


        #make figure
        make_mean_psth_scenes(meandfs,psth_cols,fig_title,fig_file_path)

# print('---- MOTION PER TRIAL ----------')
# #plot motion
# if motion_thresh is not None:
#     fig_out_dir = os.path.join(traceid_dir,'figures','trials_trace','motion_%i'%(motion_thresh),'parsed_motion')
# else:
#     fig_out_dir = os.path.join(traceid_dir,'figures','trials_trace','parsed_motion')
# if not os.path.exists(fig_out_dir):
#     os.makedirs(fig_out_dir)




# motion_traces = np.array(data_grp['motion'])
# ylabel = 'motion'
# mdfs = []
# for tidx in range(motion_traces.shape[0]):
#     trial_filter1[tidx]
#     if trial_filter1[tidx]:
#         motion_trace = motion_traces[tidx,:]
#         df = pd.DataFrame({'%s'% ylabel: motion_trace,
#                             'tsec': frames_tsec,
#                            'trial': [int(tidx) for _ in range(len(motion_trace))],
#                             'config': [int(trial_config[tidx]) for _ in range(len(motion_trace))],
#                             'img': [int(trial_img[tidx]) for _ in range(len(motion_trace))],
#                             'stim_cond': [int(trial_cond[tidx]) for _ in range(len(motion_trace))],
#                            'nreps': [trials_to_keep for _ in range(len(motion_trace))]
#                            })
#         mdfs.append(df)
# mdfs = pd.concat(mdfs, axis=0)


# p = sns.relplot(x="tsec", y=ylabel,units = 'trial',estimator = None,hue = 'stim_cond', col="img", col_wrap=psth_cols,
#                 kind="line", palette=["#4c72b0","#c44e52","#55a868"],size = 5,alpha = 0.6,data = mdfs,legend=False)


# plt.subplots_adjust(wspace=0.1, hspace=0.8, top=0.85, bottom=0.1, left=0.1)

# axes = p.axes
# ymin,ymax = axes[0].get_ylim()
# xmin,xmax = axes[0].get_xlim()
# start_val = 0.0
# end_val = 1.0 #hard-coding
# for ri in range(p.axes.shape[0]):
#         #print ri, ci
#         p.axes[ri].add_patch(patches.Rectangle((start_val, ymin), end_val, ymax-ymin, linewidth=0, fill=True, color='k', alpha=0.2))
#         p.axes[ri].text(-0.999, ymax+(ymax*0.2), 'n=%i' % int(mdfs['nreps'].values[0]), fontsize=6)
#         p.axes[ri].axhline(y=0, xmin=xmin, xmax= xmax, linewidth=1, color='k',linestyle = '--')

#         if ri == 0:
#           #  p.axes[ri].yaxis.set_major_locator(pl.MaxNLocator(2))
#             p.axes[ri].set_xticks(())
#             sns.despine(trim=True, offset=0, bottom=True, left=False, ax=p.axes[ri])
#             p.axes[ri].set_xlabel('time (s)', fontsize=8)
#             p.axes[ri].set_ylabel('%s' % ylabel, fontsize=8)
#         else:
#             sns.despine(trim=True, offset=0, bottom=True, left=True, ax=p.axes[ri])
#             p.axes[ri].tick_params(
#                                     axis='both',          # changes apply to the x-axis
#                                     which='both',      # both major and minor ticks are affected
#                                     bottom='off',      # ticks along the bottom edge are off
#                                     left='off',
#                                     top='off',         # ticks along the top edge are off
#                                     labelbottom='off',
#                                     labelleft='off') # labels along the bottom edge are off)
#             p.axes[ri].set_xlabel('')
#             p.axes[ri].set_ylabel('')



# # Create the legend patches
# legend_patches = [matplotlib.patches.Patch(color=C, label=L) for
#                   C, L in zip(["#4c72b0","#c44e52","#55a868"],
#                               trace_labels)]

# # # Plot the legend
# plt.legend(handles=legend_patches,bbox_to_anchor=(0, -0.0), loc=2, borderaxespad=0.1, fontsize=8)

# p.fig.suptitle('Parsed Motion')

# p.savefig(os.path.join(fig_out_dir,'parsed_motion_trace.png'))
# plt.close()



# # type_idx = 1
# # t_type = trace_type[type_idx]
# # traces = np.array(data_grp[curr_slice]['traces'][t_type][data_type])

# # ridx = 2
# print('---- REPONSE PER TRIAL ----------')
# for type_idx,t_type in enumerate(trace_type):
#     # t_type = 'df'
#     print('type: %s'%(t_type))
#     traces = np.array(data_grp[curr_slice]['traces'][t_type][data_type])

#     #plot individual trials
#     if motion_thresh is not None:
#         fig_out_dir = os.path.join(traceid_dir,'figures','trials_trace','motion_%i'%(motion_thresh),data_type,t_type)
#     else:
#         fig_out_dir = os.path.join(traceid_dir,'figures','trials_trace',data_type,t_type)
#     if not os.path.exists(fig_out_dir):
#         os.makedirs(fig_out_dir)

#     for ridx in cell_rois:
#         print(ridx)

#         ylabel = t_type
#         alldfs = []
#         for tidx in range(traces.shape[0]):
#             if trial_filter1[tidx]:
#                 trace = traces[tidx,:,ridx]
#                 df = pd.DataFrame({'%s'% ylabel: trace,
#                                     'tsec': frames_tsec,
#                                    'trial': [int(tidx) for _ in range(len(trace))],
#                                     'config': [int(trial_config[tidx]) for _ in range(len(trace))],
#                                     'img': [int(trial_img[tidx]) for _ in range(len(trace))],
#                                     'stim_cond': [int(trial_cond[tidx]) for _ in range(len(trace))],
#                                    'nreps': [trials_to_keep for _ in range(len(trace))]
#                                    })
#                 alldfs.append(df)
#         alldfs = pd.concat(alldfs, axis=0)
                
            

#         fig_title = ('roi %05d' % (int(ridx)))
#         fig_fn = 'roi%05d_trials_trace.png'%ridx
#         fig_file_path = os.path.join(fig_out_dir,fig_fn)


#         #make figure
#         make_trials_psth_scenes(alldfs,psth_cols,fig_title,fig_file_path)

print('ALL DONE!')
print(opts.animalid,opts.session)