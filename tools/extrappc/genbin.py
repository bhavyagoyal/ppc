from os import path as osp
from scipy import io as sio
import numpy as np

def random_sampling(points, num_points, replace=None, return_choices=False):
    if replace is None:
        replace = (points.shape[0] < num_points)
    choices = np.random.choice(points.shape[0], num_points, replace=replace)
    if return_choices:
        return points[choices], choices
    else:
        return points[choices]


BASEFOLDER = '~/Documents/mmdetection3d/data/sunrgbd/'
SBR = '5_5'
for imageId in range(1, 2):#5051):
    imageIdst = str(imageId).zfill(6)
    INFOLDER = BASEFOLDER + 'sunrgbd_trainval/processed_full_lowflux/SimSPADDataset_nr-576_nc-704_nt-1024_tres-586ps_dark-0_psf-0/'
    OUTFOLDER = BASEFOLDER + 'points_' + SBR + '_gtdepth/'
    depth = sio.loadmat(INFOLDER + 'spad_' + imageIdst + '_' + SBR + '_gtdepth.mat')['instance']
    pc_upright_depth_subsampled = random_sampling(depth, 50000)
    pc_upright_depth_subsampled.tofile(OUTFOLDER + imageIdst+'.bin')

