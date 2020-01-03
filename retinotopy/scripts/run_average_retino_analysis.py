import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import h5py
import sys
import glob
import os
import json
import pandas as pd
import numpy as np
import pylab as pl
import pprint
pp = pprint.PrettyPrinter(indent=4)


def get_para_info(para_file_dir):
    if not os.path.exists(para_file_dir): os.makedirs(para_file_dir)
    para_files =  [f for f in os.listdir(para_file_dir) if f.endswith('.json')]#assuming a single file for all tiffs in run
    if len(para_files) == 0:
        # Paradigm info not extracted yet:
        raw_para_files = [f for f in glob.glob(os.path.join(run_dir, 'raw*', 'paradigm_files', '*.mwk')) if not f.startswith('.')]
        print run_dir
        assert len(raw_para_files) == 1, "No raw .mwk file found, and no processed .mwk file found. Aborting!"
        raw_para_file = raw_para_files[0]           
        print "Extracting .mwk trials: %s" % raw_para_file 
        fn_base = os.path.split(raw_para_file)[1][:-4]
        trials = mw.extract_trials(raw_para_file, retinobar=True, trigger_varname='frame_trigger', verbose=True)
        para_fpath = mw.save_trials(trials, para_file_dir, fn_base)
        para_file = os.path.split(para_fpath)[-1]
    else:
        assert len(para_files) == 1, "Unable to find unique .mwk file..."
        para_file = para_files[0]

    print 'Getting paradigm file info from %s'%(os.path.join(para_file_dir, para_file))

    with open(os.path.join(para_file_dir, para_file), 'r') as r:
        parainfo = json.load(r)
    return parainfo

def findOccurrences(s, ch):
    return [i for i, letter in enumerate(s) if letter == ch]

class struct: pass

opts = struct()
opts.rootdir = '/n/coxfs01/2p-data'
opts.animalid = 'JC110'
opts.session = '20190913'
opts.acquisition = 'FOV1_zoom4p0x'
opts.traceid = 'traces102'
opts.run_list = ['retino_run1','retino_run2']

exclude_file_dict = {'retino_run1':[1,3,6,10,12],\
                    'retino_run2':[4,9]}

#% Set up paths:    
traceid = '%s_s2p'%(opts. traceid)
#% Set up paths:    
acquisition_dir = os.path.join(opts.rootdir, opts.animalid, opts.session, opts.acquisition)


traceid_dir = os.path.join(acquisition_dir, 'retino_combined', 'traces',traceid)
#Output paths(retino_combined)
fig_base_dir = os.path.join(traceid_dir,'retino_analysis','figures')
if not os.path.exists(fig_base_dir):
    os.makedirs(fig_base_dir)
file_out_dir = os.path.join(traceid_dir,'retino_analysis','files')
if not os.path.exists(file_out_dir):
    os.makedirs(file_out_dir)
file_grp = h5py.File(os.path.join(file_out_dir,'combined_retino_data.hdf5'),  'w')



stim_type = ['right','top','left','bottom']#just hard-code all stim types

#current_stimulus = stim_type[0]
for current_stimulus in stim_type:
    print('Running Condition:%s'%current_stimulus)
    all_trace = np.array([])
    # run_idx = 0
    # run = opts.run_list[run_idx]

    for run_idx, run in enumerate(opts.run_list):
        traceid_dir = os.path.join(acquisition_dir, run, 'traces',traceid)

        file_dir = os.path.join(traceid_dir,'retino_analysis','files')
        run_dir = traceid_dir.split('/traces')[0]
        trace_arrays_dir = os.path.join(traceid_dir,'files')


        # Get associated RUN info:
        runmeta_path = os.path.join(run_dir, '%s.json' % run)
        with open(runmeta_path, 'r') as r:
            runinfo = json.load(r)

        nslices = len(runinfo['slices'])
        nchannels = runinfo['nchannels']
        nvolumes = runinfo['nvolumes']
        ntiffs = runinfo['ntiffs']
        frame_rate = runinfo['frame_rate']

        #-----Get info from paradigm file
        para_file_dir = os.path.join(run_dir,'paradigm','files')

        parainfo = get_para_info(para_file_dir)

        #get masks
        curr_slice = 'Slice01'#hard-coding planar data for now

        #get stimulus type for each run
        file_stim = []
        for fid in range(1,ntiffs+1):
            stimfreq = parainfo[str(fid)]['stimuli']['scale']
            file_stim.append(parainfo[str(fid)]['stimuli']['stimulus'])

        stim_type = list(set(file_stim))


        #for current_stimulus in stim_type:
        relevant_files = [i for i, e in enumerate(file_stim) if e == current_stimulus]
        relevant_files = [x+1 for x in relevant_files]

        #filter out runs
        if len(exclude_file_dict[run])>0:
            exclude_idxs = [np.where(np.array(relevant_files) == file)\
                            for file in exclude_file_dict[run]]
            exclude_idxs = np.hstack(exclude_idxs)[0]
            relevant_files_filt = np.delete(relevant_files, exclude_idxs, axis=0)
        else:
            relevant_files_filt = relevant_files


        for counter,fid in enumerate(relevant_files_filt):

            trace_file = [f for f in os.listdir(trace_arrays_dir) if 'File%03d'%(fid) in f and f.endswith('hdf5')][0]
            trace_fn = os.path.join(trace_arrays_dir,trace_file)
            print(trace_fn)

            rawfile = h5py.File(trace_fn, 'r')
            cell_rois = np.array(rawfile.attrs['s2p_cell_rois'])
            if counter == 0:
                file_grp.attrs['s2p_cell_rois'] = cell_rois
            
            #consider cell rois only
            file_trace = np.transpose(rawfile[curr_slice]['traces']['global_df_f']['cell'])[cell_rois]

            if all_trace.size == 0:
                frametimes = np.array(rawfile[curr_slice]['frames_tsec'])
                all_trace = file_trace
            else:
                all_trace = np.dstack((all_trace,file_trace))
            rawfile.close()

        #keep track of all files used for average
        print('Saving',run)
        relevant_files_filt = np.array(relevant_files_filt)
        src_set = file_grp.create_dataset('/'.join([current_stimulus,'src_files',run]),\
            relevant_files_filt.shape, relevant_files_filt.dtype,relevant_files_filt)

    
    roi_trace = np.squeeze(np.mean(all_trace,2))


    #Get fft  
    print('Getting fft....')
    fourier_data = np.fft.fft(roi_trace)

    nrois,nframes = roi_trace.shape


    #Get magnitude and phase data
    print('Analyzing phase and magnitude....')
    mag_data=abs(fourier_data)
    phase_data=np.angle(fourier_data)

    #label frequency bins
    freqs = np.fft.fftfreq(nframes, float(1/frame_rate))
    idx = np.argsort(freqs)
    freqs=freqs[idx]

    #sort magnitude and phase data
    mag_data=mag_data[:,idx]
    phase_data=phase_data[:,idx]

    #excluding DC offset from data
    freqs=freqs[np.round(nframes/2)+1:]
    mag_data=mag_data[:,np.round(nframes/2)+1:]
    phase_data=phase_data[:,np.round(nframes/2)+1:]

    freq_idx=np.argmin(np.absolute(freqs-stimfreq))#find out index of stimulation freq
    top_freq_idx=np.where(freqs>1)[0][0]#find out index of 1Hz, to cut-off zoomed out plot
    max_mod_idx=np.argmax(mag_data[:,freq_idx],0)#best pixel index

    #unpack values from frequency analysis
    mag_array = mag_data[:,freq_idx]                    
    phase_array = phase_data[:,freq_idx]      

    #get magnitude ratio
    tmp=np.copy(mag_data)
    np.delete(tmp,freq_idx,1)
    nontarget_mag_array=np.sum(tmp,1)
    mag_ratio_array=mag_array/nontarget_mag_array




    #Save values to file


    magset = file_grp.create_dataset('/'.join([current_stimulus,curr_slice,'mag_array']),\
                                     mag_array.shape, mag_array.dtype)
    magset[...] = mag_array

    phaseset = file_grp.create_dataset('/'.join([current_stimulus,curr_slice,'phase_array']),\
                                       phase_array.shape, phase_array.dtype)
    phaseset[...] = phase_array

    ratioset = file_grp.create_dataset('/'.join([current_stimulus,curr_slice,'mag_ratio_array']),\
                                       mag_ratio_array.shape, mag_ratio_array.dtype)
    ratioset[...] = mag_ratio_array

    #VISUALIZE!!!
    print('Visualizing results')
    print('Output folder: %s'%(fig_base_dir))

    std_thresh = 0.01
    active_idxs = np.where(mag_ratio_array>std_thresh)[0]

    print('Number of active cells for %s: %d '%(current_stimulus,active_idxs.size))   

    #visualize pixel-based result

    #make figure directory for stimulus type
    fig_dir = os.path.join(fig_base_dir,current_stimulus,'spectrum')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    for midx in active_idxs:
        fig_name = 'full_spectrum_mask%04d.png' %(cell_rois[midx])
        fig=plt.figure()
        plt.plot(freqs,mag_data[midx,:])
        plt.xlabel('Frequency (Hz)',fontsize=16)
        plt.ylabel('Magnitude',fontsize=16)
        axes = plt.gca()
        ymin, ymax = axes.get_ylim()
        plt.axvline(x=freqs[freq_idx], linewidth=1, color='r')
        plt.savefig(os.path.join(fig_dir,fig_name))
        plt.close()

    for midx in active_idxs:
        fig_name = 'zoom_spectrum_mask%04d.png' %(cell_rois[midx])
        fig=plt.figure()
        plt.plot(freqs[0:top_freq_idx],mag_data[midx,0:top_freq_idx])
        plt.xlabel('Frequency (Hz)',fontsize=16)
        plt.ylabel('Magnitude',fontsize=16)
        axes = plt.gca()
        ymin, ymax = axes.get_ylim()
        plt.axvline(x=freqs[freq_idx], linewidth=1, color='r')
        plt.savefig(os.path.join(fig_dir,fig_name))
        plt.close()

    fig_dir = os.path.join(fig_base_dir,current_stimulus,'timecourse')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    stimperiod_t=np.true_divide(1,stimfreq)
    stimperiod_frames=stimperiod_t*frame_rate
    periodstartframes=np.round(np.arange(0,len(frametimes),stimperiod_frames))[:]
    periodstartframes = periodstartframes.astype('int')

    for midx in active_idxs:
        fig_name = 'timecourse_fit_mask%03d.png' %(cell_rois[midx])
        fig=plt.figure(figsize=(20,5))
        plt.plot(frametimes,roi_trace[midx,:],'b')
        plt.xlabel('Time (s)',fontsize=16)
        plt.ylabel('df/F',fontsize=16)
        plt.title('ROI %04d'%cell_rois[midx],fontsize=16)
        axes = plt.gca()
        ymin, ymax = axes.get_ylim()
        for f in periodstartframes:
                plt.axvline(x=frametimes[f], linewidth=1, color='k')
        axes.set_xlim([frametimes[0],frametimes[-1]])
        plt.savefig(os.path.join(fig_dir,fig_name))
        plt.close()

    # #Read in average image (for viuslization)
    masks_fn = os.path.join(file_dir,'masks.hdf5')
    mask_file = h5py.File(masks_fn, 'r')

    mask_array = np.array(mask_file[curr_slice]['mask_array'])

    im0 = np.array(mask_file[curr_slice]['meanImg'])
    mask_file.close()
    szx,szy = im0.shape

    mask_array.shape



    im1 = np.uint8(np.true_divide(im0,np.max(im0))*255)
    im2 = np.dstack((im1,im1,im1))

    #set phase map range for visualization
    phase_array_disp=np.copy(phase_array)
    phase_array_disp[phase_array<0]=-phase_array[phase_array<0]
    phase_array_disp[phase_array>0]=(2*np.pi)-phase_array[phase_array>0]


    #mark rois
    magratio_roi = np.empty((szy,szx))
    magratio_roi[:] = np.NAN
    mag_roi = np.copy(magratio_roi)
    phase_roi = np.copy(magratio_roi)

    magratio_roi_thresh = np.copy(magratio_roi)
    phase_roi_thresh = np.copy(magratio_roi)

    for midx in range(nrois):
        maskpix = np.where(np.squeeze(mask_array[cell_rois[midx],:,:]))
        #print(len(maskpix))
        magratio_roi[maskpix]=mag_ratio_array[midx]
        mag_roi[maskpix]=mag_array[midx]
        phase_roi[maskpix]=phase_array_disp[midx]
        
    for midx in active_idxs:
        maskpix = np.where(np.squeeze(mask_array[cell_rois[midx],:,:]))
        #print(len(maskpix))
        magratio_roi_thresh[maskpix]=mag_ratio_array[midx]
        phase_roi_thresh[maskpix]=phase_array_disp[midx]

    nrois

    fig_dir = os.path.join(fig_base_dir,current_stimulus)



    sns.set_style("whitegrid", {'axes.grid' : False})

    fig_name = 'phase_info_all.png'
    fig=plt.figure()
    plt.imshow(im2,'gray')
    plt.imshow(phase_roi,'nipy_spectral',alpha = 0.5,vmin=0,vmax=2*np.pi)
    plt.savefig(os.path.join(fig_dir,fig_name))
    plt.close()

    fig_name = 'phase_info_thresh.png'
    fig=plt.figure()
    plt.imshow(im2,'gray')
    plt.imshow(phase_roi_thresh,'nipy_spectral',alpha = 0.5,vmin=0,vmax=2*np.pi)
    plt.savefig(os.path.join(fig_dir,fig_name))

    fig_name = 'mag_ratio_all.png'
    fig=plt.figure()
    plt.imshow(im2,'gray')
    plt.imshow(magratio_roi)
    plt.colorbar()
    plt.savefig(os.path.join(fig_dir,fig_name))

    fig_name = 'mag_ratio_thresh.png'
    fig=plt.figure()
    plt.imshow(im2,'gray')
    plt.imshow(magratio_roi_thresh,vmin = np.nanmin(magratio_roi[:]))
    plt.colorbar()
    plt.savefig(os.path.join(fig_dir,fig_name))

    fig_name = 'phase_nice_all.png'
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
    ax.imshow(phase_roi,'nipy_spectral',alpha = 0.5,vmin=0,vmax=2*np.pi)
    fig.savefig(os.path.join(fig_dir,fig_name), dpi=dpi, transparent=True)
    plt.close()

    fig_name = 'phase_nice_thresh.png'
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
    ax.imshow(phase_roi_thresh,'nipy_spectral',alpha = 0.5,vmin=0,vmax=2*np.pi)
    fig.savefig(os.path.join(fig_dir,fig_name), dpi=dpi, transparent=True)
    plt.close()

     #correct orientation
    phase_roi_corr = np.rot90(phase_roi_thresh)
    phase_roi_corr = np.flip(phase_roi_corr,1)

    im1_corr = np.rot90(im1,0)
    im1_corr = np.flip(im1_corr,1)

    im2_corr = np.dstack((im1_corr,im1_corr,im1_corr))

    fig_name = 'phase_nice_corrected_thresh.png'
    dpi = 80
    szY,szX = im1.shape
    # What size does the figure need to be in inches to fit the image?
    figsize = szX / float(dpi), szY / float(dpi)
    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    # Hide spines, ticks, etc.
    ax.axis('off')
    ax.imshow(im2_corr,'gray')
    ax.imshow(phase_roi_corr,'nipy_spectral',alpha = 0.5,vmin=0,vmax=2*np.pi)
    fig.savefig(os.path.join(fig_dir,fig_name), dpi=dpi, transparent=True)
    plt.close()

file_grp.close()