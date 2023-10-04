import numpy as np
import numpy.linalg
import sys
import pandas
import skimage
import scipy
import matplotlib.pyplot as plt

fname = sys.argv[1]
raw_hist_fname = fname.replace('PointCloud', 'RawDataHistogramMap').replace('xyz', 'txt')
depth_fname = fname.replace('PointCloud', 'Depth').replace('frame', 'frameid').replace('xyz', 'txt')

points = pandas.read_csv(fname, header=None)
points = points.to_numpy()*[1,-1,1]
points = points[:,[2,0,1]]

depth = pandas.read_csv(depth_fname, header=None)
depth = depth.to_numpy()[:,:-1]
print('Depth ', depth.shape, depth.max())

cx, cy, fx, fy = 128.411, 95.596, 122.391, 122.747
k1, k2, k3 = -0.2317, 0.0805, -0.0102
nr, nc = 192, 256

def undistortedgrid(nr, nc):
    xx = np.linspace(1, nc, nc) 
    yy = np.linspace(1, nr, nr) 
    x, y = np.meshgrid(xx, yy)
    x, y = x - cx, y - cy
    x, y = x/cx, y/cy
    r2 = x**2 + y**2
    r4, r6 = r2**2, r2**3
    gamma = 1 + r2*k1 + r4*k2 + r6*k3
    xp, yp = x/gamma, y/gamma
    xp, yp = xp*cx + cx, yp*cy + cy
    return xp, yp

# Convert dist to depth
def finaldepth(dist):
    xx = np.linspace(1, nc, nc) 
    yy = np.linspace(1, nr, nr) 
    x, y = np.meshgrid(xx, yy) 
    #print(x[:5,:5], y[:5,:5])
    x, y = undistortedgrid(nr, nc)
    #print(x[:5,:5], y[:5,:5])
    x = (x - cx)/fx
    y = (y - cy)/fy
    depthmap = dist/(x**2 + y**2 + 1)**0.5
    return depthmap


def depth2points(depthmap):
    xx = np.linspace(1, nc, nc) 
    yy = np.linspace(1, nr, nr) 
    x, y = np.meshgrid(xx, yy) 
    x, y = undistortedgrid(nr, nc)
    x = (x - cx)*depthmap/fx
    y = (y - cy)*depthmap/fy
    z = depthmap

    points3d = np.stack([z, -x, y])
    points3d = points3d.reshape((3,-1))
    return points3d.T

#depth_points = depth2points(depth) 

raw_hist = pandas.read_csv(raw_hist_fname, header=None, delim_whitespace=True).to_numpy()
gf_pulse = np.zeros((1,22))
gf_pulse[0,11] = 1
gf_pulse = skimage.filters.gaussian(gf_pulse,sigma=1.0)
###gf_pulse = gf_pulse/gf_pulse.sum()
raw_hist = scipy.signal.convolve(raw_hist, gf_pulse, mode='same')

# from 19-32 bins have first epoch, not sure why.
raw_hist[:,:35] = 0


def argmaxrandomtie(spad):
    maxval = spad.max(axis=-1, keepdims=True)
    maxmatrix = spad == maxval
    
    spadmax = np.zeros(spad.shape[0])
    for i in range(spad.shape[0]):
        spadmax[i] = np.random.choice(np.flatnonzero(maxmatrix[i,:]))
    return spadmax



dist = np.linalg.norm(points, axis=1)
keep_points = dist>=0.75
keep_points = keep_points.astype(np.float32)
points = points*keep_points[:,None]
#ignore_points = 1 - keep_points
print(fname, keep_points.sum())

#TOTAL = 49152
#density = np.ones(TOTAL)
#conf = np.linspace(0,1,TOTAL)
total = raw_hist.sum(axis=1)
density = raw_hist.max(axis=1)
keep_hist = density>0.4
keep_hist = keep_hist.astype(np.float32)
#raw_hist = raw_hist*keep_hist[:,None]
#densitynz = density[np.nonzero(density)]
total = np.clip(total, 1, None)
conf = density/total 
reflec = density/density.max()

BIN_WIDTH = 60.0/672
#hist_dist = raw_hist.argmax(-1)
hist_dist = argmaxrandomtie(raw_hist)
hist_dist = hist_dist.reshape(nr,nc)
hist_points = depth2points(finaldepth((hist_dist)*BIN_WIDTH)) 
hist_points = hist_points*keep_hist[:,None]

points3d = np.concatenate([points, density[:,None], conf[:,None], np.tile(reflec[:,None], (1,3))], axis=1)
#depth_points3d = np.concatenate([depth_points, density[:,None], conf[:,None], np.tile(reflec[:,None], (1,3))], axis=1)
hist_points3d = np.concatenate([hist_points, density[:,None], conf[:,None], np.tile(reflec[:,None], (1,3))], axis=1)
print('Final Points ', points3d.shape)

outfname = fname + '.bin'
points3d.astype(np.float32).tofile(outfname)
#outfname = fname + 'depth.bin'
#depth_points3d.astype(np.float32).tofile(outfname)
outfname = fname + 'hist.bin'
hist_points3d.astype(np.float32).tofile(outfname)

points3d = np.concatenate([points, density[:,None], np.zeros((49152,1)), np.tile(reflec[:,None], (1,3))], axis=1)
hist_points3d = np.concatenate([hist_points, density[:,None], np.ones((49152,1)), np.tile(reflec[:,None], (1,3))], axis=1)
histall_points3d = np.concatenate([hist_points3d, points3d], axis=1)

outfname = fname + 'histall.bin'
histall_points3d.astype(np.float32).tofile(outfname)



