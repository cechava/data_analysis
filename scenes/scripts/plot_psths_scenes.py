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
              #  p.axes[ri].text(-0.999, ymax+(ymax*0.2), 'n=%i' % nreps, fontsize=6)
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

class struct: pass

optsE = struct()
optsE.rootdir = '/n/coxfs01/2p-data'
optsE.animalid = 'JC110'
optsE.session = '20190916'
optsE.acquisition = 'FOV1_zoom4p0x'
traceid = 'traces102_s2p'
run = 'scenes_combined'


#% Set up paths:    
acquisition_dir = os.path.join(optsE.rootdir, optsE.animalid, optsE.session, optsE.acquisition)
if 'combined' in run:
    traceid_dir = os.path.join(acquisition_dir, run,'traces',traceid)
else:
    traceid_dir = util.get_traceid_from_acquisition(acquisition_dir, run, traceid)
run_dir = traceid_dir.split('/traces')[0]
trace_arrays_dir = os.path.join(traceid_dir,'files')
paradigm_dir = os.path.join(acquisition_dir, run, 'paradigm')

#open file to read
data_array_fn = 'processed_config_traces.hdf5'
data_array_filepath = os.path.join(traceid_dir, 'data_arrays', data_array_fn)
data_grp = h5py.File(data_array_filepath, 'r')

frames_tsec = data_grp.attrs['frames_tsec']
nrois = data_grp.attrs['nrois']
print('ROIs:%i'%(nrois))

if 's2p_cell_rois' in data_grp.attrs.keys():
    cell_rois = data_grp.attrs['s2p_cell_rois']
else:
    cell_rois = np.arange(nrois)

curr_slice = 'Slice01'#hard,coding for now

#figures options
trace_labels = ['Original','Medium Complex','Low Complex']
data_type = 'np_subtracted'
trace_type = ['df_f','zscore','df']
type_title = ['df/f','z-score','df']
# trace_type = ['zscore']
# type_title = ['zscore']
# trace_type = ['df_f']
# type_title = ['df/f']
psth_cols = 5
sns.set_style("ticks")
palette = sns.color_palette(["#4c72b0","#c44e52","#55a868"])
sns.set_palette(palette)

for type_idx,t_type in enumerate(trace_type):

    fig_out_dir = os.path.join(traceid_dir,'figures','mean_trace',data_type,t_type)
    if not os.path.exists(fig_out_dir):
        os.makedirs(fig_out_dir)

    #go through rois
    for ridx in cell_rois:
        print(ridx)

        meandfs = []
        ylabel = type_title[type_idx]
        #cfg_key = 'config001'
        for cfg_count,cfg_key in enumerate(data_grp[curr_slice].keys()):
            img = np.array(data_grp['/'.join([curr_slice,cfg_key,'img'])])[0]+1
            stim_cond = np.array(data_grp['/'.join([curr_slice,cfg_key,'scene_cond'])])[0]
            mean_trace = np.array(data_grp['/'.join([curr_slice,cfg_key,t_type, 'trace_mean',data_type])])[:,ridx]
            sem_trace = np.array(data_grp['/'.join([curr_slice,cfg_key,t_type, 'trace_se',data_type])])[:,ridx]
            nreps = np.array(data_grp['/'.join([curr_slice,cfg_key,t_type, 'trace',data_type])]).shape[0]
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
        
        fig_title = ('roi %05d' % (int(ridx)))
        fig_fn = 'roi%05d_mean_trace.png'%ridx
        fig_file_path = os.path.join(fig_out_dir,fig_fn)
        
        #make figure
        make_mean_psth_scenes(meandfs,psth_cols,fig_title,fig_file_path)