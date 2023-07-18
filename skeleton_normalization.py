import os
import os.path as osp
import numpy as np
import pickle
import logging
# import h5py
# from sklearn.model_selection import train_test_split
# from visualize import Draw3DSkeleton
# from fpca import FPCA_manager
from prenormalization import pre_normalization
from height_remove import HeightRemove

#------------------------------
'''
Shanaka Ramesh Gunasekara
2022.08.12

'''


def skeleton_normalization (data_fp):
    print('*************  skeleton normalization ***************')
    data_fp = np.array(data_fp,dtype=np.float32)
        
    print('*************  start data filling into data fp ***************')
    data_fp = np.transpose(data_fp , [0, 4, 2, 3, 1])
    

    print('start pre normalization')
    #################orientation ######################
    data_fp = pre_normalization(data_fp) #Input data shape N, C, T, V, M
    
    hr =  HeightRemove(data_fp)
    data_fp= hr.convert()
    
    data_fp = np.transpose(data_fp,[0,4,2,3,1])
    
    
    
    
    
    data_fp= list(data_fp)
    print('normalization done ')

    
    return data_fp


