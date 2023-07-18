# -*- coding:utf-8 -*-
# import os
import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# import imageio
# import glob 
# from standard_skeleton import StandardSkeleton
# from prenormalization import pre_normalization

#------------------------------
'''
Shanaka Ramesh Gunasekara
2022.08.12

'''

trunk_joints = [0, 1, 20, 2, 3]
# arm_joints = [23, 24, 11, 10, 9, 8, 20, 4, 5, 6, 7, 22, 21]
arm_joints_left = [20, 4, 5, 6, 7, 22, 21 ]
arm_joints_right = [20 , 8,9,10,11,24,23]
leg_joints_left = [0, 12, 13, 14, 15 ]
leg_joints_right =[0 ,16 ,17, 18, 19]

# leg_joints = [19, 18, 17, 16, 0, 12, 13, 14, 15]

# body = [trunk_joints, arm_joints, leg_joints]
body = [trunk_joints, arm_joints_left,arm_joints_right, leg_joints_left,leg_joints_right]
# print(body) 

joint_data = []
joint_data_y = []
joint_data_z = []
num_joint = 25
max_frame = 300
max_body_true = 2



class HeightRemove:
    '''
    Input data shape N, C, T, V, M
    
    '''
    def __init__(self, fp, save_path=None, init_horizon=-45,
                 init_vertical=20, x_rotation=None,
                 y_rotation=None, pause_step=0.2):
        self.fp = fp

        self.save_path = save_path

        self.init_horizon = init_horizon
        self.init_vertical = init_vertical

        self.x_rotation = x_rotation
        self.y_rotation = y_rotation

        self._pause_step = pause_step


    def convert(self):
        #generate standard skeleton
        # stndrd = StandardSkeleton()
        # self.standard_skeleton = np.load('standard_skeleton.npy')
        # print(self.standard_skeleton)
        self.standard_skeleton= np.array([[-0.00188754 , 0.  ,        0.00229682 , 0.00324438 ,-0.14400026, -0.17766052,
  -0.12252636 ,-0.10376329,  0.14283249 , 0.17924742 , 0.114939 ,   0.09276552,
  -0.06062813 ,-0.08647052 ,-0.08292387, -0.07686081,  0.05790214 , 0.08870036,
   0.0832494,   0.07401707,  0.0016813 , -0.08784567, -0.09881634 , 0.07571724,
   0.08928604],
 [-0.2748321 ,  0.      ,    0.27121764,  0.38812754 , 0.16684501, -0.03288197,
  -0.1248785,  -0.14750649 , 0.16684499, -0.03119159, -0.10616463, -0.12055575,
  -0.2686846 , -0.48404354 ,-0.79652262 ,-0.83894479, -0.26790756, -0.4852781,
  -0.79769647 ,-0.84047663 , 0.20387371 ,-0.17073786 ,-0.14626023, -0.13662636,
  -0.12192167],
 [-0.0031231 ,  0. ,        -0.00687204 ,-0.01427861 ,-0.022189 ,  -0.04613424,
  -0.1435087, -0.16150194, -0.02218899, -0.04492665, -0.14838816, -0.16893849,
  -0.03027354 ,-0.08772957 ,-0.03670571 ,-0.11625542 ,-0.02946937, -0.08622096,
  -0.03488521, -0.11299883 ,-0.00335224, -0.18044581, -0.17360777 ,-0.19038683,
  -0.18068367,]])  # by averaging all the data belongs to training set in NTU120 Xsub 

        for no_samples in range (0,self.fp.shape[0]):
            print('smaple' , no_samples)
            for frame in range (0,self.fp.shape[2]):
                data = self.fp[no_samples,:,frame] # sample : x,y,z: frame 


                for part in body:
                    
                    for i in range (0,len(part)-1):

                        xx_s = self.standard_skeleton[0][part[i+1]]-self.standard_skeleton[0][part[i]]
                        yy_s = self.standard_skeleton[1][part[i+1]]-self.standard_skeleton[1][part[i]]
                        zz_s = self.standard_skeleton[2][part[i+1]]-self.standard_skeleton[2][part[i]]
                        std_vec = np.array([xx_s,yy_s,zz_s])              
                        # calculating Euclidean distance
                       
                        dist = np.linalg.norm(std_vec)

                        #normal skeleton varibles to be transformed to standard 
                        xx = data[0][part[i+1]][0]-data[0][part[i]][0]
                        yy = data[1][part[i+1]][0]-data[1][part[i]][0]
                        zz = data[2][part[i+1]][0]-data[2][part[i]][0]
                        
                        vec = np.array([xx,yy,zz])
                        uni_vec = vec/ np.linalg.norm(vec)
                        modifed_vec = dist*uni_vec

                        data[0][part[i+1]][0] = modifed_vec[0] + data[0][part[i]][0]
                        data[1][part[i+1]][0] = modifed_vec[1] + data[1][part[i]][0]
                        data[2][part[i+1]][0] = modifed_vec[2] + data[2][part[i]][0]


                self.fp[no_samples,0:data.shape[1],frame] = data

            data= self.fp[0,:,0:data.shape[1]]

            xx = data[0,:,2,:]-data[0,:,0,:]
            yy = data[1,:,2,:]-data[1,:,0,:]
            zz = data[2,:,2,:]-data[2,:,0,:]

            del xx,yy,zz,data ,self.standard_skeleton
            return self.fp
        ###################################################################################
