
import os,glob,sys

import numpy as np
from natsort import natsorted

sys.path.insert(0, '/n/coxfs01/cechavarria/repos/suite2p')
from scipy import ndimage
from suite2p.utils import get_tif_list, list_tifs

import scipy.io
from skimage.external.tifffile import imread, TiffFile, imsave
from skimage.measure import block_reduce

def findOccurrences(s, ch):
    return [i for i, letter in enumerate(s) if letter == ch]

def block_mean_stack(stack0, ds_block):
    im0 = block_reduce(stack0[0,:,:],ds_block) 
    print(im0.shape)
    stack1 = np.zeros((stack0.shape[0],im0.shape[0],im0.shape[1]))
    for i in range(0,stack0.shape[0]):
        stack1[i,:,:] = block_reduce(stack0[i,:,:],ds_block) 
    return stack1

#provide some info
rootdir = '/n/coxfs01/2p-data'


animalid = 'JC120'
session = '20191115'
acquisition = 'FOV1_zoom4p0x'

#run_list = ['retino_run1','scenes_run1','scenes_run2','scenes_run3','scenes_run4','scenes_run5','scenes_run6']
#run_list = ['scenes_run8','scenes_run9','scenes_run10']
run_list = ['scenes_run3']
#run_list = ['retino_run1','retino_run2','scenes_run1','scenes_run2','scenes_run3','scenes_run4','scenes_run5','scenes_run6','scenes_run7','scenes_run8','scenes_run9','scenes_run10']
#run_list = ['scenes_run1','scenes_run2','scenes_run3','scenes_run3','scenes_run4','scenes_run5','scenes_run6','scenes_run7','scenes_run8','scenes_run9','scenes_run10']
#figure out directories to search
dst_dir = os.path.join(rootdir,animalid,session,acquisition,'all_combined','block_reduced')
if not os.path.isdir(dst_dir):
    os.makedirs(dst_dir)

for run in run_list:
    #run = run_list[0]
    data_dir = os.path.join(rootdir,animalid,session,acquisition,run)

    raw_dir = glob.glob(os.path.join(data_dir,'raw*'))[0]
    print(raw_dir)
    src_dir = os.path.join(raw_dir,'*.tif')


    for fn in glob.glob(src_dir):

        i0 = findOccurrences(fn,'/')[-1]
        i1 = findOccurrences(fn,'_')[-1]


        new_fn = '%s_%s_%s'%(acquisition,run,fn[i1+1:])
        print(new_fn)


        stack0 = imread(fn)

        stack1 = block_mean_stack(stack0,(2,2))


        print(stack1.shape)

        imsave(os.path.join(dst_dir,new_fn),stack1)