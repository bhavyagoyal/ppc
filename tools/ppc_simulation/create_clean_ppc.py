import pickle
import copy
import numpy as np
import random
import os

dataset='sunrgbd'
#dataset='kitti'

SAMPLED_POINTS=50000 # for sun rgbd


if(dataset=='sunrgbd'):
    BASE = '../../data/sunrgbd/'
    INBASEFOLDER = os.path.join(BASE, 'points')
    OUTBASEFOLDER = os.path.join(BASE, 'points_min2/0.3/argmax-filtering-sbr', 'clean')
    startdataidx, enddataidx = 1, 10335
    num_feats = 6
else:
    BASE = '../../data/kitti/'
    INBASEFOLDER = os.path.join(BASE, 'training/velodyne_reduced')
    OUTBASEFOLDER = os.path.join(BASE, 'training/points2048_r025_dist10/0.3/argmax-filtering-sbr', 'clean')
    startdataidx, enddataidx = 0, 7480
    num_feats = 4
    

# convert original point cloud to probabilistic point cloud
# by adding an extra feature of probability 1 to each point
for i in range(startdataidx, enddataidx+1):
    fname = str(i).zfill(6) + '.bin'
    print(fname)
    points = np.fromfile(os.path.join(INBASEFOLDER, fname), dtype=np.float32)
    points = points.reshape(-1,num_feats)
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    points = np.concatenate([points[:,:3], ones*1000., ones, points[:,3:]], 1)
    choices = np.random.choice(points.shape[0], SAMPLED_POINTS)
    points = points[choices]
    if not os.path.exists(OUTBASEFOLDER):
        os.makedirs(OUTBASEFOLDER)
    points.tofile(os.path.join(OUTBASEFOLDER,fname))

