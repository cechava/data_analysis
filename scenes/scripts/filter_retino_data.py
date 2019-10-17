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


def correct_phase_wrap(phase):
        
    corrected_phase = phase.copy()
    
    corrected_phase[phase<0] =- phase[phase<0]
    corrected_phase[phase>0] = (2*np.pi) - phase[phase>0]
    
    return corrected_phase


#hard-coding these measurements.
screen_resolution = [1920, 1080]
screen_width = 103.0 # in cm
screen_height = 58.0 # in cm
screen_distance = 30.0 # in cm

screen_height_deg = 2*np.rad2deg(np.arctan2(screen_height/2,screen_distance))
screen_width_deg = 2*np.rad2deg(np.arctan2(screen_width/2,screen_distance))


def filter_retino(opts):

    filter_crit = opts.filter_crit
    filter_thresh = opts.filter_thresh

    traceid = '%s_s2p'%(opts.traceid)
    #% Set up paths:    
    acquisition_dir = os.path.join(opts.rootdir, opts.animalid, opts.session, opts.acquisition)
    traceid_dir = os.path.join(acquisition_dir, opts.retino_run, 'traces',traceid)

    file_dir = os.path.join(traceid_dir,'retino_analysis','files')
    run_dir = traceid_dir.split('/traces')[0]
    trace_arrays_dir = os.path.join(traceid_dir,'retino_analysis','files')

    #make figure directory for stimulus type
    fig_out_dir = os.path.join(traceid_dir, 'retino_analysis','combined_figures')
    if not os.path.exists(fig_out_dir):
        os.makedirs(fig_out_dir)

    # Get associated RUN info:
    runmeta_path = os.path.join(run_dir, '%s.json' % opts.retino_run)
    with open(runmeta_path, 'r') as r:
        runinfo = json.load(r)

    nslices = len(runinfo['slices'])
    nchannels = runinfo['nchannels']
    nvolumes = runinfo['nvolumes']
    ntiffs = runinfo['ntiffs']
    frame_rate = runinfo['frame_rate']

    #-----Get info from paradigm file
    para_file_dir = os.path.join(run_dir,'paradigm','files')
    if not os.path.exists(para_file_dir): os.makedirs(para_file_dir)
    para_files =  [f for f in os.listdir(para_file_dir) if f.endswith('.json')]#assuming a single file for all tiffs in run
    if len(para_files) == 0:
        # Paradigm info not extracted yet:
        raw_para_files = [f for f in glob.glob(os.path.join(run_dir, 'raw*', 'paradigm_files', '*.mwk')) if not f.startswith('.')]
        print(run_dir)
        assert len(raw_para_files) == 1, "No raw .mwk file found, and no processed .mwk file found. Aborting!"
        raw_para_file = raw_para_files[0]           
        print("Extracting .mwk trials: %s" % (raw_para_file))
        fn_base = os.path.split(raw_para_file)[1][:-4]
        trials = mw.extract_trials(raw_para_file, retinobar=True, trigger_varname='frame_trigger', verbose=True)
        para_fpath = mw.save_trials(trials, para_file_dir, fn_base)
        para_file = os.path.split(para_fpath)[-1]
    else:
        assert len(para_files) == 1, "Unable to find unique .mwk file..."
        para_file = para_files[0]

    print('Getting paradigm file info from %s'%(os.path.join(para_file_dir, para_file)))

    with open(os.path.join(para_file_dir, para_file), 'r') as r:
        parainfo = json.load(r)


    #get masks
    curr_slice = 'Slice01'#hard-coding planar data for now

    masks_fn = os.path.join(trace_arrays_dir,'masks.hdf5')
    mask_file = h5py.File(masks_fn, 'r')

    iscell = np.array(mask_file[curr_slice]['iscell'])
    total_cells = np.sum(iscell)
    print('Totals cells: %i'% total_cells)

    mean_img = np.array(mask_file[curr_slice]['meanImg'])
    mask_array = np.array(mask_file[curr_slice]['mask_array'])
    mask_file.close()

    RF_center_array = np.ones((ntiffs,total_cells))*np.nan
    mag_ratio_array = np.ones((ntiffs,total_cells))*np.nan
    bar_orientation = np.zeros((ntiffs,))
    bar_cond = np.zeros((ntiffs,))



    #fid = 1
    for fid in range(1,ntiffs+1):

        retino_filepath = os.path.join(file_dir,'File%03d_retino_data.hdf5'%(fid))


        file_grp = h5py.File(retino_filepath,  'r')
        file_mag_ratio= np.array(file_grp['/'.join([curr_slice,'mag_ratio_array'])])
        file_phase = np.array(file_grp['/'.join([curr_slice,'phase_array'])])

        stimulus = parainfo[str(fid)]['stimuli']['stimulus']

        #set phase map range for visualization
        file_phase_disp = correct_phase_wrap(file_phase)

        #by convention: left edge of screen is 0, bottom edge of screen is 0
        if stimulus == 'bottom':
            file_location = screen_height_deg * (file_phase_disp/(2*np.pi))
            bar_orientation[fid-1] = 0
            bar_cond[fid-1] = 0
        elif stimulus == 'top':
            file_location = screen_height_deg - (screen_height_deg * (file_phase_disp/(2*np.pi)))
            #file_location = (screen_height_deg * (file_phase_disp/(2*np.pi)))
            bar_orientation[fid-1] = 0
            bar_cond[fid-1] = 1
        elif stimulus == 'left':
            file_location = screen_width_deg * (file_phase_disp/(2*np.pi))
            bar_orientation[fid-1] = 1
            bar_cond[fid-1] = 2
        elif stimulus == 'right':
            file_location = screen_width_deg - (screen_width_deg * (file_phase_disp/(2*np.pi)))
         #   file_location = (screen_width_deg * (file_phase_disp/(2*np.pi)))
            bar_orientation[fid-1] = 1
            bar_cond[fid-1] = 3
        
        #only pass values for cell ROIs
        mag_ratio_array[fid-1,:] = file_mag_ratio[iscell]
        RF_center_array[fid-1,:] = file_location[iscell]





    #plot distribution of mag ratio across all cells/files
    fig = plt.figure(figsize=(7.5,5))
    p = sns.distplot(mag_ratio_array.flatten(),bins = 30)

    axes = plt.gca()
    xmin, xmax = axes.get_xlim()
    ymin, ymax = axes.get_ylim()

    if filter_crit == 'ratio':
        plt.vlines(x=filter_thresh, ymin=ymin, ymax= ymax, linewidth=1, color='r',linestyle = '-')
        
        thresh_zscore = (filter_thresh - np.nanmean(mag_ratio_array.flatten()))/np.nanstd(mag_ratio_array.flatten())
        n_pass_elements = np.sum(mag_ratio_array.flatten()>filter_thresh)
        n_elements = float(mag_ratio_array.flatten().size)
        frac_pass_elements = n_pass_elements/n_elements
        
        plt.text(xmin, ymax+.1, 'n = %i ; f = %.02f ; z = %.02f' % \
        (n_pass_elements,frac_pass_elements,thresh_zscore), fontsize=10)

    plt.xlabel('Element Count',fontsize = 15)
    plt.ylabel('Mag Ratio',fontsize = 15)


    fig_fn = 'mag_ratio_hist_across_filtes_%s_%.02f.png'%(filter_crit,filter_thresh)
    fig_file_path = os.path.join(fig_out_dir, fig_fn)
    plt.savefig(fig_file_path)
    plt.close()



    #apply threshold and average per condition
    if filter_crit == 'ratio':
        RF_center_array[np.where(mag_ratio_array<filter_thresh)] = np.nan
        mag_ratio_array[np.where(mag_ratio_array<filter_thresh)] = np.nan



    RF_center_mean = np.zeros((4,RF_center_array.shape[1]))
    RF_center_std = np.zeros((4,RF_center_array.shape[1]))
    mag_ratio_mean_cond = np.zeros((4,RF_center_array.shape[1]))
    active_cell_count = np.zeros((4,))
    for cidx in np.unique(bar_cond):
    #cidx = 0 
        cidx = int(cidx)
        RF_center_mean[cidx,:] = np.nanmean(RF_center_array[np.where(bar_cond==cidx)[0],:],0)
        mag_ratio_mean_cond[cidx,:] = np.nanmean(mag_ratio_array[np.where(bar_cond==cidx)[0],:],0)
        RF_center_std[cidx,:] = np.nanstd(RF_center_array[np.where(bar_cond==cidx)[0],:],0)
        if filter_crit == 'ratio':
            active_cell_count[cidx] = np.sum(np.sum((mag_ratio_array>filter_thresh)[bar_cond==cidx],0)>0)

    p = sns.jointplot(RF_center_std.flatten(),mag_ratio_mean_cond.flatten(), kind = 'reg',height = 10)

    p.annotate(stats.pearsonr)
    p.set_axis_labels(xlabel='RF SD', ylabel='Mag Ratio',fontsize = 15)

    fig_fn = 'location_std_vs_mag_ratio_joint_plot_thresh_%s_%.02f.png'%(filter_crit,filter_thresh)
    fig_file_path = os.path.join(fig_out_dir, fig_fn)
    p.savefig(fig_file_path)
    plt.close()




    stimulus_labels = ['bottom','top','left','right']


    cond1 = 0
    cond2 = 1

    num_active_cells = (active_cell_count[cond1],active_cell_count[cond2])
    frac_active_cells = (active_cell_count[cond1]/total_cells,active_cell_count[cond2]/total_cells)


    num_valid_cells = np.sum(np.logical_and(np.logical_not(np.isnan(RF_center_mean))[cond1,:],np.logical_not(np.isnan(RF_center_mean))[cond2,:]))
    frac_valid_cells = num_valid_cells/float(total_cells)

    print('# active cells = %i, %i'%(num_active_cells))
    print('frac active cells = %.04f, %.04f'%(frac_active_cells))

    print('# valid cells = %i'%(num_valid_cells))
    print('frac valid_cells = %.04f'%(frac_valid_cells))


    #fig = plt.figure(figsize=(10.3,5.8))
    sns.set(style="darkgrid")
    #scatter plot considers only cells with valid values for both axes
    p = sns.jointplot(RF_center_mean[cond1,:],RF_center_mean[cond2,:],kind = 'scatter',height = 10,\
                      xlim = (0,screen_width_deg),ylim = (0,screen_width_deg))#


    x0, x1 = p.ax_joint.get_xlim()
    y0, y1 = p.ax_joint.get_ylim()
    lims = [max(x0, y0), min(x1, y1)]
    p.ax_joint.plot(lims, lims, ':k')   

    p.set_axis_labels(xlabel=stimulus_labels[cond1], ylabel=stimulus_labels[cond2],fontsize = 15)

    p.annotate(stats.pearsonr)

    fig_fn = '%s_%s_joint_plot_thresh_%s_%.02f.png'%(stimulus_labels[cond1],stimulus_labels[cond2]\
                                                       ,filter_crit,filter_thresh)
    fig_file_path = os.path.join(fig_out_dir, fig_fn)
    p.savefig(fig_file_path)
    plt.close()

    cond1 = 2
    cond2 = 3

    num_active_cells = (active_cell_count[cond1],active_cell_count[cond2])
    frac_active_cells = (active_cell_count[cond1]/total_cells,active_cell_count[cond2]/total_cells)


    num_valid_cells = np.sum(np.logical_and(np.logical_not(np.isnan(RF_center_mean))[cond1,:],np.logical_not(np.isnan(RF_center_mean))[cond2,:]))
    frac_valid_cells = num_valid_cells/float(total_cells)

    print('# active cells = %i, %i'%(num_active_cells))
    print('frac active cells = %.04f, %.04f'%(frac_active_cells))

    print('# valid cells = %i'%(num_valid_cells))
    print('frac valid_cells = %.04f'%(frac_valid_cells))


    #fig = plt.figure(figsize=(10.3,5.8))
    sns.set(style="darkgrid")
    #scatter plot considers only cells with valid values for both axes
    p = sns.jointplot(RF_center_mean[cond1,:],RF_center_mean[cond2,:],kind = 'scatter',height = 10,\
                      xlim = (0,screen_width_deg),ylim = (0,screen_width_deg))#


    x0, x1 = p.ax_joint.get_xlim()
    y0, y1 = p.ax_joint.get_ylim()
    lims = [max(x0, y0), min(x1, y1)]
    p.ax_joint.plot(lims, lims, ':k')   

    p.set_axis_labels(xlabel=stimulus_labels[cond1], ylabel=stimulus_labels[cond2],fontsize = 15)

    p.annotate(stats.pearsonr)

    fig_fn = '%s_%s_joint_plot_thresh_%s_%.02f.png'%(stimulus_labels[cond1],stimulus_labels[cond2]\
                                                       ,filter_crit,filter_thresh)
    fig_file_path = os.path.join(fig_out_dir, fig_fn)
    p.savefig(fig_file_path)
    plt.close()





    #3,1 for: X - right, Y- top
    #2,0 for: X - left, Y- bottom
    hcond = 3
    vcond = 1

    num_active_cells = (active_cell_count[hcond],active_cell_count[vcond])
    frac_active_cells = (active_cell_count[hcond]/total_cells,active_cell_count[vcond]/total_cells)

    mag_ratio_mean = np.nanmean(np.vstack((mag_ratio_mean_cond[hcond,:],mag_ratio_mean_cond[vcond,:])),0)

    num_valid_cells = np.sum(np.logical_and(np.logical_not(np.isnan(RF_center_mean))[hcond,:],np.logical_not(np.isnan(RF_center_mean))[vcond,:]))
    frac_valid_cells = num_valid_cells/float(total_cells)

    print('# active cells = %i, %i'%(num_active_cells))
    print('frac active cells = %.04f, %.04f'%(frac_active_cells))

    print('# valid cells = %i'%(num_valid_cells))
    print('frac valid_cells = %.04f'%(frac_valid_cells))

    fig = plt.figure(figsize=(20.6,11.6))

    #scatter plot considers only cells with valid values for both axes
    plt.scatter(RF_center_mean[hcond,:],RF_center_mean[vcond,:],100,c = mag_ratio_mean, cmap = 'inferno')#

    plt.xlim([0,screen_width_deg])
    plt.ylim([0,screen_height_deg])

    axes = plt.gca()
    xmin, xmax = axes.get_xlim()
    ymin, ymax = axes.get_ylim()

    plt.text(xmin-.25, ymax, 'n = %s ; f = %s ; n = %i ; f = %.04f' % \
             (num_active_cells,frac_active_cells,num_valid_cells,frac_valid_cells), fontsize=10)

    plt.colorbar()
    plt.xlabel('Azimuth',fontsize = 15)
    plt.ylabel('Elevation',fontsize = 15)
    plt.suptitle('%s / %s'%(stimulus_labels[hcond], stimulus_labels[vcond]),fontsize = 15)

    fig_fn = '%s_%s_joint_plot_thresh_%s_%.02f.png'%(stimulus_labels[hcond],stimulus_labels[vcond]\
                                                       ,filter_crit,filter_thresh)
    fig_file_path = os.path.join(fig_out_dir, fig_fn)
    plt.savefig(fig_file_path)
    plt.close()

    #fig = plt.figure(figsize=(10.3,5.8))
    sns.set(style="darkgrid")
    #scatter plot considers only cells with valid values for both axes
    p = sns.jointplot(RF_center_mean[hcond,:],RF_center_mean[vcond,:],kind = 'kde',height = 20.6,\
                      xlim = (0,screen_width_deg),ylim = (0,screen_height_deg),joint_kws=dict(shade_lowest=False))#

    p.fig.set_figwidth(10.3)
    p.fig.set_figheight(5.8)
    axes = plt.gca()
    xmin, xmax = axes.get_xlim()
    ymin, ymax = axes.get_ylim()

    p.set_axis_labels(xlabel='Azimuth', ylabel='Elevation',fontsize = 15)

    fig_fn = '%s_%s_joint_plot_kde_thresh_%s_%.02f.png'%(stimulus_labels[hcond],stimulus_labels[vcond]\
                                                       ,filter_crit,filter_thresh)
    fig_file_path = os.path.join(fig_out_dir, fig_fn)
    p.savefig(fig_file_path)
    plt.close()

    #3,1 for: X - right, Y- top
    #2,0 for: X - left, Y- bottom
    hcond = 2
    vcond = 0

    num_active_cells = (active_cell_count[hcond],active_cell_count[vcond])
    frac_active_cells = (active_cell_count[hcond]/total_cells,active_cell_count[vcond]/total_cells)

    mag_ratio_mean = np.nanmean(np.vstack((mag_ratio_mean_cond[hcond,:],mag_ratio_mean_cond[vcond,:])),0)

    num_valid_cells = np.sum(np.logical_and(np.logical_not(np.isnan(RF_center_mean))[hcond,:],np.logical_not(np.isnan(RF_center_mean))[vcond,:]))
    frac_valid_cells = num_valid_cells/float(total_cells)

    print('# active cells = %i, %i'%(num_active_cells))
    print('frac active cells = %.04f, %.04f'%(frac_active_cells))

    print('# valid cells = %i'%(num_valid_cells))
    print('frac valid_cells = %.04f'%(frac_valid_cells))

    fig = plt.figure(figsize=(20.6,11.6))

    #scatter plot considers only cells with valid values for both axes
    plt.scatter(RF_center_mean[hcond,:],RF_center_mean[vcond,:],100,c = mag_ratio_mean, cmap = 'inferno')#

    plt.xlim([0,screen_width_deg])
    plt.ylim([0,screen_height_deg])

    axes = plt.gca()
    xmin, xmax = axes.get_xlim()
    ymin, ymax = axes.get_ylim()

    plt.text(xmin-.25, ymax, 'n = %s ; f = %s ; n = %i ; f = %.04f' % \
             (num_active_cells,frac_active_cells,num_valid_cells,frac_valid_cells), fontsize=10)

    plt.colorbar()
    plt.xlabel('Azimuth',fontsize = 15)
    plt.ylabel('Elevation',fontsize = 15)
    plt.suptitle('%s / %s'%(stimulus_labels[hcond], stimulus_labels[vcond]),fontsize = 15)

    fig_fn = '%s_%s_joint_plot_thresh_%s_%.02f.png'%(stimulus_labels[hcond],stimulus_labels[vcond]\
                                                       ,filter_crit,filter_thresh)
    fig_file_path = os.path.join(fig_out_dir, fig_fn)
    plt.savefig(fig_file_path)
    plt.close()

    #fig = plt.figure(figsize=(10.3,5.8))
    sns.set(style="darkgrid")
    #scatter plot considers only cells with valid values for both axes
    p = sns.jointplot(RF_center_mean[hcond,:],RF_center_mean[vcond,:],kind = 'kde',\
                      xlim = (0,screen_width_deg),ylim = (0,screen_height_deg),joint_kws=dict(shade_lowest=False))#

    p.fig.set_figwidth(10.3)
    p.fig.set_figheight(5.8)
    axes = plt.gca()
    xmin, xmax = axes.get_xlim()
    ymin, ymax = axes.get_ylim()


    p.set_axis_labels(xlabel='Azimuth', ylabel='Elevation',fontsize = 15)

    fig_fn = '%s_%s_joint_plot_kde_thresh_%s_%.02f.png'%(stimulus_labels[hcond],stimulus_labels[vcond]\
                                                       ,filter_crit,filter_thresh)
    fig_file_path = os.path.join(fig_out_dir, fig_fn)
    p.savefig(fig_file_path)
    plt.close()



    #average only cells with valid responses for both directions
    RF_x1 = np.copy(RF_center_mean[2,:])
    RF_x2 = np.copy(RF_center_mean[3,:])
    RF_x1[np.isnan(RF_x2)]=np.nan
    RF_x2[np.isnan(RF_x1)]=np.nan

    RF_y1 = np.copy(RF_center_mean[0,:])
    RF_y2 = np.copy(RF_center_mean[1,:])
    RF_y1[np.isnan(RF_y2)]=np.nan
    RF_y2[np.isnan(RF_y1)]=np.nan


    RF_mean_x = np.nanmean(np.vstack((RF_x1,RF_x2)),0)
    RF_mean_y = np.nanmean(np.vstack((RF_y1,RF_y2)),0)
    mag_ratio_mean = np.nanmean(mag_ratio_mean_cond,0)

    #flag values that are out of range!
    if np.sum(RF_mean_x>screen_width_deg) or np.sum(RF_mean_y>screen_height_deg):
        print('Location offscreen!! Check code')

    num_active_cells = np.sum(np.logical_not(np.isnan(mag_ratio_mean)))#reflects number of cells that had above thresh response during at least 1 file
    frac_active_cells = num_active_cells/total_cells

    num_valid_cells = np.sum(np.logical_and(np.logical_not(np.isnan(RF_mean_x)),np.logical_not(np.isnan(RF_mean_y))))
    frac_valid_cells = num_valid_cells/float(total_cells)

    print('# active cells = %i'%(num_active_cells))
    print('frac active cells = %.04f'%(frac_active_cells))

    print('# valid cells = %i'%(num_valid_cells))
    print('frac valid_cells = %.04f'%(frac_valid_cells))


    fig = plt.figure(figsize=(20.6,11.6))

    #scatter plot considers only cells with valid values for both axes
    plt.scatter(RF_mean_x, RF_mean_y,100,c = mag_ratio_mean, cmap = 'inferno')#

    plt.xlim([0,screen_width_deg])
    plt.ylim([0,screen_height_deg])

    axes = plt.gca()
    xmin, xmax = axes.get_xlim()
    ymin, ymax = axes.get_ylim()


    plt.text(xmin-.25, ymax, 'n = %s ; f = %s ; n = %i ; f = %.04f' % \
             (num_active_cells,frac_active_cells,num_valid_cells,frac_valid_cells), fontsize=10)

    plt.colorbar()
    plt.xlabel('Azimuth',fontsize = 15)
    plt.ylabel('Elevation',fontsize = 15)
    plt.suptitle('Mean Azimuth/Mean Elevation')

    fig_fn = 'average_position_joint_plot_thresh_%s_%.02f.png'%(filter_crit,filter_thresh)
    fig_file_path = os.path.join(fig_out_dir, fig_fn)
    plt.savefig(fig_file_path)
    plt.close()

    #fig = plt.figure(figsize=(10.3,5.8))
    sns.set(style="darkgrid")
    #scatter plot considers only cells with valid values for both axes
    p = sns.jointplot(RF_mean_x,RF_mean_y,kind = 'kde',\
                      xlim = (0,screen_width_deg),ylim = (0,screen_height_deg),joint_kws=dict(shade_lowest=False))#

    p.fig.set_figwidth(10.3)
    p.fig.set_figheight(5.8)
    axes = plt.gca()
    xmin, xmax = axes.get_xlim()
    ymin, ymax = axes.get_ylim()


    p.set_axis_labels(xlabel='Azimuth', ylabel='Elevation',fontsize = 15)

    fig_fn = 'average_position_joint_plot_kde_thresh_%s_%.02f.png'%(filter_crit,filter_thresh)
    fig_file_path = os.path.join(fig_out_dir, fig_fn)
    p.savefig(fig_file_path)
    plt.close()

    for cond in np.unique(bar_cond):
        cond = int(cond)
        im0 = mean_img
        szx,szy = im0.shape

        im1 = np.uint8(np.true_divide(im0,np.max(im0))*255)
        im2 = np.dstack((im1,im1,im1))

        #mark rois
        phase_roi = np.ones((szy,szx))*np.nan
        cell_rois = np.where(iscell)[0]
        for midx in range(RF_center_mean.shape[1]):
            cell_idx = cell_rois[midx]
            maskpix = np.where(np.squeeze(mask_array[cell_idx,:,:]))
            phase_roi[maskpix]=RF_center_mean[cond,midx]



        fig_name = 'phase_nice_%s.png' % stimulus_labels[cond] #curr_file #(tiff_fn[:-4])
        dpi = 80
        szY,szX = im1.shape
        # What size does the figure need to be in inches to fit the image?
        figsize = szX / float(dpi), szY / float(dpi)
        # Create a figure of the right size with one axes that takes up the full figure
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])
        # Hide spines, ticks, etc.
        ax.axis('off')
        ax.imshow(im2,'gray')
        if cond<2:
            color_limit = screen_height_deg
        else:
            color_limit = screen_width_deg
        plt.imshow(phase_roi,'nipy_spectral',alpha = 0.5,vmin=0,vmax=color_limit)
        fig.savefig(os.path.join(fig_out_dir,fig_name), dpi=dpi, transparent=True)
        plt.close()





    #save things to file

     #save arrays to file
    # Create outfile:
    data_array_fn = 'retino_data_thresh_%s_%.02f.hdf5'%(filter_crit, filter_thresh)
    data_array_filepath = os.path.join(trace_arrays_dir, data_array_fn)
    data_grp = h5py.File(data_array_filepath, 'w')

    data_grp.attrs['ntiffs'] = ntiffs
    data_grp.attrs['s2p_cell_rois'] = iscell
    data_grp.attrs['stimulus_labels'] = stimulus_labels



    bar_cond_dset = data_grp.create_dataset('bar_cond', bar_cond.shape, bar_cond.dtype)
    bar_cond_dset[...] = bar_cond


    rf_x_dset = data_grp.create_dataset('/'.join([curr_slice, 'RF_mean_x' ]), RF_mean_x.shape, RF_mean_x.dtype)
    rf_x_dset[...] = RF_mean_x

    rf_y_dset = data_grp.create_dataset('/'.join([curr_slice, 'RF_mean_y' ]), RF_mean_y.shape, RF_mean_y.dtype)
    rf_y_dset[...] = RF_mean_y

    rf_all_dset = data_grp.create_dataset('/'.join([curr_slice, 'RF_center_per_file' ]), RF_center_array.shape, RF_center_array.dtype)
    rf_all_dset[...] = RF_center_array

    rf_cond_dset = data_grp.create_dataset('/'.join([curr_slice, 'RF_center_mean_across_files_per_condition' ]), RF_center_mean.shape, RF_center_mean.dtype)
    rf_cond_dset[...] = RF_center_mean

    rf_cond_sd_dset = data_grp.create_dataset('/'.join([curr_slice, 'RF_center_std_across_files_per_condition' ]), RF_center_std.shape, RF_center_std.dtype)
    rf_cond_sd_dset[...] = RF_center_std

    n_cell_dset = data_grp.create_dataset('/'.join([curr_slice, 'active_cell_count_per_cond' ]), active_cell_count.shape, active_cell_count.dtype)
    n_cell_dset[...] = active_cell_count

    ratio_dset = data_grp.create_dataset('/'.join([curr_slice, 'mag_ratio_mean_per_file' ]), mag_ratio_array.shape, mag_ratio_array.dtype)
    ratio_dset[...] = mag_ratio_array

    ratio_cond_dset = data_grp.create_dataset('/'.join([curr_slice, 'mag_ratio_mean_per_cond' ]), mag_ratio_mean_cond.shape, mag_ratio_mean_cond.dtype)
    ratio_cond_dset[...] = mag_ratio_mean_cond



    data_grp.close()


def extract_options(options):
    parser = optparse.OptionParser()

    # PATH opts:
    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/n/coxfs01/2p-data', help='source dir (root project dir containing all expts) [default: /n/coxfs01/2p-data]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')
    parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID') 
    parser.add_option('-A', '--acq', action='store', dest='acquisition', default='', help="acquisition folder (ex: 'FOV1_zoom3x')")
    parser.add_option('-T', '--traceid', action='store', dest='traceid', default='traces001', help="(ex: traces001)")
    parser.add_option('-R', '--run', action='store', dest='retino_run', default='', help='name of combo run') 
    parser.add_option('-f', '--filter_crit', action='store', dest='filter_crit', default='zscore', help='criterion to filter traces e.g.zscore') 
    parser.add_option('-t', '--filter_thresh', action='store', dest='filter_thresh', default='zscore', help='cutoff value of filter criterion') 
    (options, args) = parser.parse_args() 

    return options




#-----------------------------------------------------
#           MAIN SET OF ACTIONS
#-----------------------------------------------------

def main(options): 
    
    options = extract_options(options)

    print('----- filterting responses-----')
    filter_retino(options)


    
#%%

if __name__ == '__main__':
    main(sys.argv[1:])
