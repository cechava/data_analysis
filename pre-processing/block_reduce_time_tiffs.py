
import os,glob,sys

import numpy as np


sys.path.insert(0, '/n/coxfs01/cechavarria/repos/suite2p')
from scipy import ndimage
from suite2p.utils import get_tif_list, list_tifs

import scipy.io
from skimage.external.tifffile import imread, TiffFile, imsave
from skimage.measure import block_reduce
from scipy.ndimage import filters

def findOccurrences(s, ch):
    return [i for i, letter in enumerate(s) if letter == ch]

def block_reduce_time_stack(stack0, ds_block):
    ntpts0,szy0,szx0 = stack0.shape
    array0 = np.reshape(stack0,(ntpts0,szy0*szx0))
    del stack0 
    
    array1 = filters.uniform_filter1d(array0,ds_block,0)
    
    sub_idx = np.arange(0,array1.shape[0],ds_block)
    array2 = array1[sub_idx,:]
    stack1 = np.reshape(array2,(array2.shape[0],szx0,szy0))
    
    return stack1
#provide some info
rootdir = '/n/coxfs01/2p-data'


animalid = 'JC097'
session = '20190717'
acquisition = 'FOV1_zoom2p0x'

run_list = ['retino_run1','scenes_run1','scenes_run2','scenes_run3','scenes_run4','scenes_run5','scenes_run6']
#run_list = ['scenes_run6']
#figure out directories to search
dst_dir = os.path.join(rootdir,animalid,session,acquisition,'all_combined','block_time_reduced')
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

        stack1 = block_reduce_time_stack(stack0,4)

        print(stack1.shape)

        imsave(os.path.join(dst_dir,new_fn),stack1)