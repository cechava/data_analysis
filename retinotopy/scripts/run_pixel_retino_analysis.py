import matplotlib
matplotlib.use('Agg')
import os
import glob
import copy
import sys
import h5py
import json
import re
import datetime
import optparse
import pprint
import traceback
import time
import skimage
import shutil
from scipy.ndimage import filters
import matplotlib.pyplot as plt
import matplotlib
import pylab as pl
import numpy as np
from tifffile import imsave

def get_comma_separated_args(option, opt, value, parser):
  setattr(parser.values, option.dest, value.split(','))


def run_analysis(opts):
    traceid = '%s_s2p'%(opts.traceid)
    opts.analysis = 'suite2p_analysis%s'%opts.traceid[opts.traceid.find('s')+1:]

    run_list = opts.run_list
    nruns = len(run_list)#% Set up paths:
    acquisition_dir = os.path.join(opts.rootdir, opts.animalid, opts.session, opts.acquisition)
    s2p_source_dir = os.path.join(acquisition_dir, 'all_combined','processed', opts.analysis, 'suite2p','plane0')#load s2p files
    s2p_ops_fn = os.path.join(s2p_source_dir,'ops.npy')
    s2p_ops = np.load(s2p_ops_fn).item()

    #open binary file
    reg_file = open(s2p_ops['reg_file'], 'rb')

    #go through runs
    for ridx,indie_run in enumerate(run_list):
        #ridx = 0
        #indie_run = run_list[ridx]
        print(indie_run)
        #----Get  relevant run info
        run_dir = os.path.join(acquisition_dir, indie_run)

        with open(os.path.join(run_dir, '%s.json' % indie_run), 'r') as fr:
            scan_info = json.load(fr)

        all_frames_tsecs = np.array(scan_info['frame_tstamps_sec'])
        T = len(all_frames_tsecs)
        framerate = scan_info['frame_rate']
        volumerate = scan_info['volume_rate']
        nvolumes = scan_info['nvolumes']
        nfiles = scan_info['ntiffs']

        print("N tsecs:", T)


        #-----Get info from paradigm file
        para_file_dir = os.path.join(run_dir,'paradigm','files')
        para_files =  [f for f in os.listdir(para_file_dir) if f.endswith('.json')]#assuming a single file for all tiffs in run
        assert len(para_files) == 1, "Unable to find unique .mwk file..."
        para_file = para_files[0]

        with open(os.path.join(para_file_dir, para_file), 'r') as r:
            parainfo = json.load(r)
            


        traceid_dir = os.path.join(run_dir, 'traces',traceid)
        file_out_dir = os.path.join(traceid_dir,'pix_retino_analysis','files')
        if not os.path.exists(file_out_dir):
            os.makedirs(file_out_dir)
            
        filetraces_fn = '%s_data_arrays.hdf5'%(indie_run)
        filetraces_filepath = os.path.join(file_out_dir, filetraces_fn)
        file_grp = h5py.File(filetraces_filepath, 'w')
        file_grp.attrs['source_file'] = s2p_ops['reg_file']

        #figure out bounds of frames for all tiffs in runs
        read_start = (ridx*nfiles*T)
        read_end = ((ridx+1)*nfiles*T)
        file_bounds = np.arange(read_start,read_end,T).astype('int64')

        #---Go though tiffs
        for fid in range(1,nfiles+1):
            #fid = 1
            print('File%03d'%fid)

            Ly = s2p_ops['Ly']
            Lx = s2p_ops['Lx']
            nbytesread =  2 * Ly * Lx

            reg_file.seek(nbytesread * file_bounds[fid-1], 0)
            buff = reg_file.read(nbytesread*T)
            data = np.frombuffer(buff, dtype=np.int16, offset=0)
            buff = []
            frames = np.reshape(data, (T,Ly, Lx))


            #saving frame stack to file

            tiff_out_dir = os.path.join(run_dir,'processed','registered')
            if not os.path.exists(tiff_out_dir):
                    os.makedirs(tiff_out_dir)
            tiff_fn = 'File%03d.tif'%(fid)
            imsave(os.path.join(tiff_out_dir,tiff_fn),frames)

            #swap axes for more familiar format
            frames = np.swapaxes(frames,0,2)
            frames = np.swapaxes(frames,0,1)

            stimulus = parainfo[str(fid)]['stimuli']['stimulus']
            stimfreq = parainfo[str(fid)]['stimuli']['scale']


            roi_trace = np.reshape(frames,(Lx*Ly, T))
            #---Get fft  
            print('Getting fft....')
            fourier_data = np.fft.fft(roi_trace)

            nrois,nframes = roi_trace.shape

            #Get magnitude and phase data
            print('Analyzing phase and magnitude....')
            mag_data=abs(fourier_data)
            phase_data=np.angle(fourier_data)

            stimfreq = 0.13
            #label frequency bins
            freqs = np.fft.fftfreq(nframes, float(1/framerate))
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

            #unpack values from frequency analysis
            mag_array = mag_data[:,freq_idx]                    
            phase_array = phase_data[:,freq_idx]      

            #get magnitude ratio
            tmp=np.copy(mag_data)
            np.delete(tmp,freq_idx,1)
            nontarget_mag_array=np.sum(tmp,1)
            mag_ratio_array=mag_array/nontarget_mag_array

            max_mod_idx=np.argmax(mag_ratio_array)#best pixel index

            #make images
            mag_img = np.reshape(mag_array,(Ly,Lx))
            phase_img = np.reshape(phase_array,(Ly,Lx))
            mag_ratio_img = np.reshape(mag_ratio_array,(Ly,Lx))

            #save images to file

            rimg = file_grp.create_dataset('/'.join([str(fid), 'mag_ratio']), mag_ratio_img.shape, mag_ratio_img.dtype)
            rimg[...] = mag_ratio_img

            mimg = file_grp.create_dataset('/'.join([str(fid), 'mag']), mag_img.shape, mag_img.dtype)
            mimg[...] = mag_img

            phimg = file_grp.create_dataset('/'.join([str(fid), 'phase']), phase_img.shape, phase_img.dtype)
            phimg[...] = phase_img

            fig_base_dir = os.path.join(traceid_dir,'pix_retino_analysis','figures')
            if not os.path.exists(fig_base_dir):
                os.makedirs(fig_base_dir)

            #VISUALIZE!!!
            print('Visualizing results')
            print('Output folder: %s'%(fig_base_dir))
            #visualize pixel-based result

            #make figure directory for stimulus type
            fig_dir = os.path.join(fig_base_dir, stimulus, 'single_pixel')
            if not os.path.exists(fig_dir):
                os.makedirs(fig_dir)


            fig_name = 'full_spectrum_File%03d_top_pix.png' %(fid)
            fig=plt.figure()
            plt.plot(freqs,mag_data[max_mod_idx,:])
            plt.xlabel('Frequency (Hz)',fontsize=16)
            plt.ylabel('Magnitude',fontsize=16)
            axes = plt.gca()
            ymin, ymax = axes.get_ylim()
            plt.axvline(x=freqs[freq_idx], linewidth=1, color='r')
            plt.savefig(os.path.join(fig_dir,fig_name))
            plt.close()


            fig_name = 'zoom_spectrum_File%03d_top_pix.png' %(fid)
            fig=plt.figure()
            plt.plot(freqs[0:top_freq_idx],mag_data[max_mod_idx,0:top_freq_idx])
            plt.xlabel('Frequency (Hz)',fontsize=16)
            plt.ylabel('Magnitude',fontsize=16)
            axes = plt.gca()
            ymin, ymax = axes.get_ylim()
            plt.axvline(x=freqs[freq_idx], linewidth=1, color='r')
            plt.savefig(os.path.join(fig_dir,fig_name))
            plt.close()

            stimperiod_t=np.true_divide(1,stimfreq)
            stimperiod_frames=stimperiod_t*framerate
            periodstartframes=np.round(np.arange(0,T,stimperiod_frames))[:]
            periodstartframes = periodstartframes.astype('int')


            fig_name = 'timecourse_File%03d_top_pix.png' %(fid)
            fig=plt.figure()
            plt.plot(all_frames_tsecs,roi_trace[max_mod_idx,:],'b')
            plt.xlabel('Time (s)',fontsize=16)
            plt.ylabel('Pixel Value',fontsize=16)
            axes = plt.gca()
            ymin, ymax = axes.get_ylim()
            for f in periodstartframes:
                    plt.axvline(x=all_frames_tsecs[f], linewidth=1, color='k')
            axes.set_xlim([all_frames_tsecs[0],all_frames_tsecs[-1]])
            plt.savefig(os.path.join(fig_dir,fig_name))
            plt.close()

            fig_dir = os.path.join(fig_base_dir, stimulus)


            #set phase map range for visualization
            phase_img_disp=np.copy(phase_img)
            phase_img_disp[phase_img<0]=-phase_img[phase_img<0]
            phase_img_disp[phase_img>0]=(2*np.pi)-phase_img[phase_img>0]

            fig_name = 'phase_info_File%03d.png' % fid 
            fig=plt.figure(figsize = (10,10))
            plt.imshow(phase_img_disp,'nipy_spectral',vmin=0,vmax=2*np.pi)
            plt.colorbar()
            plt.savefig(os.path.join(fig_dir,fig_name))
            plt.close()

            fig_name = 'mag_ratio_File%03d.png' % fid 
            fig=plt.figure(figsize = (10,10))
            plt.imshow(mag_ratio_img)
            plt.colorbar()
            plt.savefig(os.path.join(fig_dir,fig_name))
            plt.close()

            fig_name = 'mag_File%03d.png' % fid 
            fig=plt.figure(figsize = (10,10))
            plt.imshow(mag_img)
            plt.colorbar()
            plt.savefig(os.path.join(fig_dir,fig_name))
            plt.close()


            im0 = np.array(s2p_ops['meanImg'])
            im1 = np.uint8(np.true_divide(im0,np.max(im0))*255)
            im2 = np.dstack((im1,im1,im1))

            fig_name = 'phase_nice_File%03d.png' % fid
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
            ax.imshow(phase_img_disp,'nipy_spectral',alpha = 0.5,vmin=0,vmax=2*np.pi)
            fig.savefig(os.path.join(fig_dir,fig_name), dpi=dpi, transparent=True)
            plt.close()

            phase_img_thresh = np.copy(phase_img_disp)
            phase_img_thresh[mag_ratio_img>.003] = np.nan

            fig_name = 'phase_nice_File%03d_thresh.png' % fid
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
            ax.imshow(phase_img_thresh,'nipy_spectral',alpha = 0.5,vmin=0,vmax=2*np.pi)
            fig.savefig(os.path.join(fig_dir,fig_name), dpi=dpi, transparent=True)
            plt.close()

            #correact orientation
            phase_img_disp_corr = np.copy(phase_img_disp)
            phase_img_disp_corr = np.transpose(phase_img_disp_corr)
            phase_img_disp_corr = np.flip(phase_img_disp_corr,0)
            phase_img_disp_corr = np.flip(phase_img_disp_corr,1)


            fig_name = 'phase_nice_File%03d_corrected.png' % fid
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
            ax.imshow(phase_img_disp_corr,'nipy_spectral',alpha = 0.5,vmin=0,vmax=2*np.pi)
            fig.savefig(os.path.join(fig_dir,fig_name), dpi=dpi, transparent=True)
            plt.close()
            
        file_grp.close()





def extract_options(options):
    parser = optparse.OptionParser()

    # PATH opts:
    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/n/coxfs01/2p-data', help='source dir (root project dir containing all expts) [default: /n/coxfs01/2p-data]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')
    parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID') 
    parser.add_option('-A', '--acq', action='store', dest='acquisition', default='', help="acquisition folder (ex: 'FOV1_zoom3x')")
    parser.add_option('-T', '--traceid', action='store', dest='traceid', default='', help="(ex: traces001_s2p)")
    parser.add_option('-r', '--run_list', action='callback', dest='run_list', default='',type='string',callback=get_comma_separated_args, help='comma-separated names of run dirs containing tiffs to be processed (ex: run1, run2, run3)')


    (options, args) = parser.parse_args() 

    return options




#-----------------------------------------------------
#           MAIN SET OF ACTIONS
#-----------------------------------------------------

def main(options): 
    
    options = extract_options(options)

    print('----- Running pixel-based retino analysis -----')
    run_analysis(options)


    
#%%

if __name__ == '__main__':
    main(sys.argv[1:])
