import cv2, sys, os
import argparse
import scipy.io
import scipy.signal
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import torch
from mmcv.ops.ball_query import ball_query
import copy
import skimage.filters
from matplotlib.ticker import FuncFormatter


plt.style.use('ggplot')
matplotlib.use('Agg')
matplotlib.rcParams['text.color'] = 'black'
matplotlib.rcParams['axes.labelcolor'] = 'black'
matplotlib.rcParams['xtick.color'] = 'black'
matplotlib.rcParams['ytick.color'] = 'black'
matplotlib.rcParams.update({'font.size': 20})


SUNRGBDBASE = "../../data/sunrgbd/sunrgbd_trainval/"
KITTIBASE = "../../data/kitti/training/"
SUNRGBD_GEN_FOLDER = 'processed_lowfluxlowsbr_min2/SimSPADDataset_nr-576_nc-704_nt-1024_tres-586ps_dark-0_psf-0'
KITTI_GEN_FOLDER = 'processed_velodyne_reduced_lowfluxlowsbr8192_r025_dist10/nr-576_nc-704_nt-8192_tres-73ps_dark-0_psf-0'
SUNRGBDMeta = '../OFFICIAL_SUNRGBD/SUNRGBDMeta3DBB_v2.mat'
OUTFOLDERNAME = 'points8192_r025_dist10' # ../points_min2'
#OUTFOLDERNAME = '../points_testing'

CORRECTNESS_THRESH = 25
SAMPLED_POINTS=50000 # for sun rgbd


metadata = None

C = 3e8
def tof2depth(tof):
    return tof * C / 2.

def random_sampling(points, num_points, p=None):
    replace = (points.shape[0] < num_points)
    choices = np.random.choice(points.shape[0], num_points, replace=replace, p=p)
    return points[choices], choices

pulse = [[[0.0000, 0.0000, 0.0000, 0.0000, 0.0001, 0.0013, 0.0105, 0.0520, 0.1528, 0.2659, 0.2743, 0.1676, 0.0607, 0.0130, 0.0017, 0.0001, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]]


# the arg parser
def parse_args():
    parser = argparse.ArgumentParser(description='.mat simulation file to point cloud')
    parser.add_argument(
        '--method',
        choices=['argmax-filtering-sbr', 'gaussfilter-argmax-filtering-sbr'],
        default='argmax-filtering-sbr',
        help='Method used for converting histograms to point clouds')
    parser.add_argument(
        '--sbr',
        choices=['5_1', '5_50', '5_100', '5_250', '5_500', '1_10', '1_20', '1_50', '1_100'],
        default='1_50',
        help='SBR')
    parser.add_argument(
        '--dataset',
        choices=['sunrgbd', 'kitti'],
        default='sunrgbd',
        help='select a dataset')
    parser.add_argument('--num_peaks', default=None, type=int,
                    help='num peaks for each pixel')
    parser.add_argument('--threshold', default=None, type=float,
                    help='threshold for spad filtering')
    parser.add_argument('--outfolder_prefix', default=None, type=str,
                    help='add prefix to output folder')
    parser.add_argument('--start', default=None, type=int,
                    help='start index for datalist')
    parser.add_argument('--end', default=None, type=int,
                    help='end index for datalist')
    args = parser.parse_args()
    return args


def camera_params(K):
    cx, cy = K[0,2], K[1,2]
    fx, fy = K[0,0], K[1,1]
    return cx, cy, fx, fy

# Convert dist to depth
def finaldepth(nr, nc, K, dist, gtvalid):
    xx = np.linspace(1, nc, nc)
    yy = np.linspace(1, nr, nr)
    x, y = np.meshgrid(xx, yy)
    cx, cy, fx, fy = camera_params(K)
    x = (x - cx)/fx
    y = (y - cy)/fy
    depthmap = dist/(x**2 + y**2 + 1)**0.5
    depthmap = depthmap*1000.
    depthmap = depthmap.astype(np.uint16)
    # Not sure why SUNRGBD code for converting to point cloud (read3dPoints.m) shifts last 3 bits, but I am zeroing it out for now
    # This removes points that are farther than 65.535 because np.uint16 would have wrapped for those numbers
    depthmap = (depthmap>>3)<<3
    depthmap = depthmap*gtvalid
    return depthmap


# Convert depth to point cloud
def depth2points(nr, nc, K, depthmap, Rtilt):
    depthmap = (depthmap>>3 | np.uint16(depthmap<<13))
    depthmap = depthmap.astype('float32')/1000.
    depthmap[depthmap>8]=8

    xx = np.linspace(1, nc, nc)
    yy = np.linspace(1, nr, nr)
    x, y = np.meshgrid(xx, yy)
    cx, cy, fx, fy = camera_params(K)
    x = (x - cx)*depthmap/fx
    y = (y - cy)*depthmap/fy
    z = depthmap

    points3d = np.stack([x, z, -y])
    points3d = points3d.reshape((3,-1))
    points3d = np.matmul(Rtilt, points3d)
    return points3d

# Sphertical to cartesian cordinates
def sph2cart(az, el, r):
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)
    return x, y, z


# Convert spherical cordinates to point cloud
def dist2points(nr, nc, dist, az, el):
    X, Y, Z = sph2cart(az, el, dist)
    points3d = np.stack([X, Y, Z])
    points3d = points3d.reshape((3,-1))
    return points3d



def argmaxfilteringsbr(spad, gaussian_filter_pulse=False):
    spad[:,:,:20] = 0
    #if(decompressed):
    #    # compress and decompress using truncated fourier
    #    spad = decompress(spad)
    if(gaussian_filter_pulse):
        gf_pulse = np.zeros((5,5,22))
        gf_pulse[2,2,:] = pulse[0][0]
        gf_pulse = skimage.filters.gaussian(gf_pulse,sigma=1.0)
        gf_pulse = gf_pulse/gf_pulse.sum()
        spad = scipy.signal.convolve(spad, gf_pulse, mode='same')
    else:
        spad = scipy.signal.convolve(spad, pulse, mode='same')

    # Returns first index in tie break
    spadargmax, spadmax = spad.argmax(-1), spad.max(-1)

    # Returns last index in tie break
#    nt = spad.shape[-1]
#    spadargmax, spadmax = nt - 1 - spad[:,:,::-1].argmax(-1), spad.max(-1)

    return spadargmax, spadmax, spad.sum(-1)


def human_format(num, pos):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return '%d%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


def main(args):

    if(args.dataset=='sunrgbd'):
        global metadata
        basefolder = SUNRGBDBASE
        gen_folder = SUNRGBD_GEN_FOLDER
        metadata = scipy.io.loadmat( os.path.join(basefolder,SUNRGBDMeta) )['SUNRGBDMeta'][0]
    else:
        basefolder = KITTIBASE
        gen_folder = KITTI_GEN_FOLDER

    outfolder = os.path.join(basefolder, OUTFOLDERNAME)
    if(args.outfolder_prefix):
        outfolder = os.path.join(outfolder, args.outfolder_prefix)

    sbrstr = args.sbr.split('_')
    sbrfloat = float(sbrstr[0])/float(sbrstr[1])
    outfolder = os.path.join(outfolder, args.method, args.sbr)
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    # Only for visualization
    all_correct_cf, all_incorrect_cf = [], []
    all_correct_sp, all_incorrect_sp = [], []
    all_correct_neighcount, all_incorrect_neighcount = [], []
    all_correct_neighcf, all_incorrect_neighcf = [], []
    all_correct_neighsp, all_incorrect_neighsp = [], []
    all_correct_neighcfweighted, all_incorrect_neighcfweighted = [], []
    all_correct_neighspweighted, all_incorrect_neighspweighted = [], []
    cfmax = 0

    scenes = open(os.path.join(basefolder, 'all_data_idx.txt')).readlines()
    scenes = [x.strip() for x in scenes]

    start, end = 0, len(scenes)
    if(args.start is not None):
        start = args.start
    if(args.end is not None):
        end = args.end

    #scenes_selected = random.sample(scenes[:99], 10) # for vis 
    scenes_selected = scenes[start:end]
    for scene in scenes_selected:
        print(scene)
        OUTFILE = os.path.join(outfolder, scene.zfill(6) +'.bin')
        if(os.path.exists(OUTFILE)):
            continue
        mat_file = os.path.join(basefolder, gen_folder, 'spad_' + scene.zfill(6) + '_' + args.sbr +'.mat')
        data = scipy.io.loadmat(mat_file)
    
        nr, nc = data['intensity'].shape
        nt = data['num_bins'][0,0]
        if(args.dataset=='sunrgbd'):
            Rtilt = metadata[int(scene)-1][1]
            K = metadata[int(scene)-1][2]
            depthpath = '../OFFICIAL_SUNRGBD' + metadata[int(scene)-1][3][0][16:]
            rgbpath = '../OFFICIAL_SUNRGBD' + metadata[int(scene)-1][4][0][16:]
            # Using Depth map to remove points that are NAN in original depth image, using gtvalid
            # Simulation script for histograms inpaints NAN depths
            # but I am ignoring those points as SUNRGB dataset processing ignores it too.
            gtdepth = cv2.imread(os.path.join(basefolder, depthpath), cv2.IMREAD_UNCHANGED)
            if(gtdepth is None):
                print('could not load depth image')
                exit(0)
            gtvalid = gtdepth>0 
            rgb = cv2.imread( os.path.join(basefolder, rgbpath), cv2.IMREAD_UNCHANGED)/255.
            rgb = rgb[:, :, ::-1]  # BGR -> RGB
            rgb = rgb.transpose(2,0,1) # HWC -> CHW
            rgb = rgb.reshape((3, -1)).T
        else:
            az = data['az']
            el = data['el']
            # refleactance, naming it rgb
            rgb = data['r']
        
        # Subtract 1 from range bins to get the right bin index in python
        # as matlab indexes it from 1
        # for distance calculation, this is fine to use
        #range_bins = data['range_bins']
        #dist = tof2depth(range_bins*data['bin_size'])
    
        spad = data['spad'].toarray()
        spad = spad.reshape((nr, nc, nt), order='F')
        #spadcopy = scipy.signal.convolve(spad, pulse, mode='same')
        #spadcopy = spad.copy()
        if(args.method=='argmax-filtering-sbr'):
            spad, density, densitysum = argmaxfilteringsbr(spad)
        elif(args.method=='gaussfilter-argmax-filtering-sbr'):
            spad, density, densitysum = argmaxfilteringsbr(spad, gaussian_filter_pulse=True)
        else:
            print('Not implemented')
            exit(0)

        if(args.threshold is not None):
            thresh_mask = density>=args.threshold
            spad, density, densitysum = spad*thresh_mask, density*thresh_mask, densitysum*thresh_mask
        
        density, densitysum = density.reshape(-1), densitysum.reshape(-1)

        correct = abs(data['range_bins']-spad)<=CORRECTNESS_THRESH
        correct = correct.reshape(-1)

        dist = tof2depth(spad*data['bin_size'])
        if(args.dataset=='sunrgbd'):
            depthmap = finaldepth(nr, nc, K, dist, gtvalid)
            points3d = depth2points(nr, nc, K, depthmap, Rtilt)
            valid = np.all(points3d, axis=0) # only select points that have non zero locations    
        else:
            points3d = dist2points(nr, nc, dist, az, el)
            valid = density>0 # only select points that have positive photon counts    


        density = density[valid]
        densitysum = densitysum[valid]
        correct = correct[valid]
        points3d = points3d.T
        points3d = points3d[valid]
        rgb = rgb[valid]

        if(args.dataset=='sunrgbd'):
            points3d, choices = random_sampling(points3d, SAMPLED_POINTS)
            density = density[choices]
            densitysum = densitysum[choices]
            correct = correct[choices]
            rgb = rgb[choices]


        points3d_rgb = np.concatenate([points3d, density[:,np.newaxis], density[:,np.newaxis]/densitysum[:,np.newaxis], rgb], axis=1)

        #points_xyz = torch.from_numpy(points3d_rgb[:,:3]).cuda()[None, :, :]
        #points_probs = torch.from_numpy(points3d_rgb[:,3]).cuda()[None, :]
        #points_sp = torch.from_numpy(points3d_rgb[:,4]).cuda()[None, :]
        ##points_probs = torch.ones(points3d_rgb.shape[0]).cuda()[None, :]
        ##points_sp = torch.ones(points3d_rgb.shape[0]).cuda()[None, :]

        #cfmax = max(cfmax, points_probs.max())
        ##points_sp = points_sp/points_sp.max()
    
        #all_correct_cf.extend(points_probs[0, correct].tolist())
        #all_incorrect_cf.extend(points_probs[0, ~correct].tolist())
        #all_correct_sp.extend(points_sp[0, correct].tolist())
        #all_incorrect_sp.extend(points_sp[0, ~correct].tolist())

        #MAX_BALL_NEIGHBORS = 32 #64 for sunrgbd , 32 for kitti 
        ## Ball query returns same index is neighbors are less than queried number of neighbors
        ## output looks like [3,56,74,2,44,3,3,3,3,3,3,3,3,3,3,3,3]
        ## radius 0.2 for sunrgbd, 0.8 for kitti
        #ball_idxs = ball_query(0, 0.8, MAX_BALL_NEIGHBORS, points_xyz, points_xyz).long()
        #
        ## first idx of the ball query is repeated if neighbors are fewer than MAX
        #ball_idxs_first = ball_idxs[:,:,0][:,:,None]
        #nonzero_ball_idxs = ((ball_idxs-ball_idxs_first)!=0)
        #nonzero_count = nonzero_ball_idxs.sum(-1).cpu().numpy()
    
        #all_correct_neighcount.extend(nonzero_count[0,correct].tolist())
        #all_incorrect_neighcount.extend(nonzero_count[0,~correct].tolist())
    
        #points_probs_tiled = points_probs[:,:,None].tile(MAX_BALL_NEIGHBORS)
        #points_sp_tiled = points_sp[:,:,None].tile(MAX_BALL_NEIGHBORS)
        #neighbor_probs = torch.gather(points_probs_tiled, 1, ball_idxs) 
        #neighbor_sp = torch.gather(points_sp_tiled, 1, ball_idxs) 
        #neighbor_probs = neighbor_probs*nonzero_ball_idxs
        #neighbor_sp = neighbor_sp*nonzero_ball_idxs
        ## average neighbor probability, would be less if neighbors are fewer than MAX
        #neighbor_probs = neighbor_probs.mean(-1)
        #neighbor_sp = neighbor_sp.mean(-1)
        #neighbor_probs_weighted = neighbor_probs*points_probs
        #neighbor_sp_weighted = neighbor_sp*points_sp
 
        #all_correct_neighcf.extend(neighbor_probs[0,correct].tolist())
        #all_incorrect_neighcf.extend(neighbor_probs[0,~correct].tolist())

        #all_correct_neighsp.extend(neighbor_sp[0,correct].tolist())
        #all_incorrect_neighsp.extend(neighbor_sp[0,~correct].tolist())
    
        #all_correct_neighcfweighted.extend(neighbor_probs_weighted[0,correct].tolist())
        #all_incorrect_neighcfweighted.extend(neighbor_probs_weighted[0,~correct].tolist())
    
        #all_correct_neighspweighted.extend(neighbor_sp_weighted[0,correct].tolist())
        #all_incorrect_neighspweighted.extend(neighbor_sp_weighted[0,~correct].tolist())
    
    
        # .bin file should be float 32 for mmdet3d
        points3d_rgb.astype(np.float32).tofile(OUTFILE)
        #cv2.imwrite(outfolder + scene.zfill(6) + '.png', depthmap)
    
    #UPPER = int((cfmax+0.5)*100)
    #bins = [x*0.01 for x in range(UPPER)]
    ##UPPER = int((cfmax+0.5))
    ##bins = [x for x in range(UPPER)]
    #IMAGE_DIR = 'figs_sbr_' + args.dataset + str( args.threshold )

    #plt.close()
    #plt.hist(all_correct_cf, bins, color='g', alpha=0.5)
    #plt.hist(all_incorrect_cf, bins, color='r', alpha=0.5)
    #plt.savefig(IMAGE_DIR + '/pointcf' + str(MAX_BALL_NEIGHBORS) + '_peaks_' + args.sbr + '.png', dpi=500)

    #plt.close()
    #plt.hist(all_correct_neighcf, bins, color='g', alpha=0.5)
    #plt.hist(all_incorrect_neighcf, bins, color='r', alpha=0.5)
    #plt.savefig(IMAGE_DIR + '/neighcf' + str(MAX_BALL_NEIGHBORS) + '_peaks_' + args.sbr + '.png', dpi=500)

    #plt.close()
    #bins = range(MAX_BALL_NEIGHBORS+2)
    #plt.hist(all_correct_neighcount, bins, color='g', alpha=0.5)
    #plt.hist(all_incorrect_neighcount, bins, color='r', alpha=0.5)
    #plt.savefig(IMAGE_DIR + '/neighcount' + str(MAX_BALL_NEIGHBORS) + '_peaks_' + args.sbr + '.png', dpi=500)

    #plt.close()
    #bins = [x*0.001 for x in range(51)]
    #plt.hist(all_correct_sp, bins, color='g', alpha=0.5, label='Ground Truth')
    #plt.hist(all_incorrect_sp, bins, color='r', alpha=0.5, label='Noise')
    #plt.xlabel('Probability')
    #if(args.sbr=='5_50'):
    #    plt.ylabel('Number of Points')
    #if(args.sbr=='1_100'):
    #    plt.legend()
    #plt.locator_params(axis='x', nbins=3)
    #plt.locator_params(axis='y', nbins=3)
    #ax = plt.gca()
    #ax.set_ylim([0,100000])
    #formatter = FuncFormatter(human_format)
    #ax.yaxis.set_major_formatter(formatter)
    #plt.tight_layout()
    #plt.savefig(IMAGE_DIR + '/pointsp' + str(MAX_BALL_NEIGHBORS) + '_peaks_' + args.sbr + '.pdf')

    #plt.close()
    #bins = [x*0.0001 for x in range(51)]
    #plt.hist(all_correct_neighsp, bins, color='g', alpha=0.5, label='Ground Truth')
    #plt.hist(all_incorrect_neighsp, bins, color='r', alpha=0.5, label='Noise')
    #plt.xlabel('NPD Score\n (Avg. SBR='+str(sbrfloat)+')')
    #if(args.sbr=='5_50'):
    #    plt.ylabel('Number of Points')
    #if(args.sbr=='1_100'):
    #    plt.legend()
    #plt.locator_params(axis='y', nbins=3)
    #plt.locator_params(axis='x', nbins=3)
    #ax = plt.gca()
    #ax.set_ylim([0,100000])
    #formatter = FuncFormatter(human_format)
    #ax.yaxis.set_major_formatter(formatter)
    #plt.tight_layout()
    #plt.savefig(IMAGE_DIR + '/neighsp' + str(MAX_BALL_NEIGHBORS) + '_peaks_' + args.sbr + '.pdf')



if __name__ == '__main__':
  args = parse_args()
  print(args)
  main(args)
