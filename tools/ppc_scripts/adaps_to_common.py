import numpy as np
import numpy.linalg
import sys
import pandas


fname = sys.argv[1]
raw_hist_fname = fname.replace('PointCloud', 'RawDataHistogramMap').replace('xyz', 'txt')
depth_fname = fname.replace('PointCloud', 'Depth').replace('frame', 'frameid').replace('xyz', 'txt')

points = pandas.read_csv(fname, header=None)
points = points.to_numpy()*[1,-1,1]
points = points[:,[2,0,1]]

depth = pandas.read_csv(depth_fname, header=None)
depth = depth.to_numpy()[:,:-1]
print('Depth ', depth.shape, depth.max())

cx, cy, fx, fy = 128, 96, 122, 122
nr, nc = 192, 256

# Convert dist to depth
def finaldepth(dist):
    xx = np.linspace(1, nc, nc) 
    yy = np.linspace(1, nr, nr) 
    x, y = np.meshgrid(xx, yy) 
    x = (x - cx)/fx
    y = (y - cy)/fy
    depthmap = dist/(x**2 + y**2 + 1)**0.5
    return depthmap


def depth2points(depthmap):
    xx = np.linspace(1, nc, nc) 
    yy = np.linspace(1, nr, nr) 
    x, y = np.meshgrid(xx, yy) 
    x = (x - cx )*depthmap/fx
    y = (y - cy)*depthmap/fy
    z = depthmap

    points3d = np.stack([z, -x, y])
    points3d = points3d.reshape((3,-1))
    return points3d.T

depth_points = depth2points(depth) 

raw_hist = pandas.read_csv(raw_hist_fname, header=None, delim_whitespace=True).to_numpy()
# TODO fix this number based on threshold
raw_hist[:,:35] = 0

hist_dist = raw_hist.argmax(-1).reshape(nr,nc)
hist_points = depth2points(finaldepth(hist_dist*70/672)) # TODO find bin width


dist = np.linalg.norm(points, axis=1)
keep_points = dist>=0.6
keep_points = keep_points.astype(np.float32)
points = points*keep_points[:,None]
ignore_points = 1 - keep_points
ignore_hist = ignore_points[:,None]*raw_hist
#print(keep_points[10235:10240])
#RANGEx,RANGEy=10000,10005
#print(keep_points[RANGEx:RANGEy])

#TOTAL = 49152
#density = np.ones(TOTAL)
#conf = np.linspace(0,1,TOTAL)
density = raw_hist.max(axis=1)
densitynz = density[np.nonzero(density)]
ignore_density = ignore_hist.max(axis=1)
ignore_densitynz = ignore_density[np.nonzero(ignore_density)]
depthbin = raw_hist.argmax(axis=1)
print(fname, densitynz.mean(), densitynz.max(), densitynz.min(), np.count_nonzero(density), keep_points.sum(), ignore_densitynz.mean(), ignore_densitynz.max(), ignore_densitynz.min())
total = raw_hist.sum(axis=1)
total = np.clip(total, 1, None)
conf = density/total 
reflec = density/density.max()

#print(raw_hist[10235:10240,:100])
#print(depthbin[10235:10240])
#print(density[10235:10240])
#print(total[10235:10240])
#print(points[10235:10240])
#print(depth[10235:10240])
#print(raw_hist[RANGEx:RANGEy,:100])
#print(depthbin[RANGEx:RANGEy])
#print(points[RANGEx:RANGEy])
#print(depth[RANGEx:RANGEy])
#exit(0)

points = np.concatenate([points, density[:,None], np.zeros((49152,1)), np.tile(reflec[:,None], (1,3))], axis=1)
depth_points = np.concatenate([depth_points, density[:,None], conf[:,None], np.tile(reflec[:,None], (1,3))], axis=1)
hist_points = np.concatenate([hist_points, density[:,None], np.ones((49152,1)), np.tile(reflec[:,None], (1,3))], axis=1)
histall_points = np.concatenate([hist_points, points], axis=1)
print('Final Points ', points.shape)

outfname = fname + '.bin'
points.astype(np.float32).tofile(outfname)
#outfname = fname + 'depth.bin'
#depth_points.astype(np.float32).tofile(outfname)
outfname = fname + 'hist.bin'
hist_points.astype(np.float32).tofile(outfname)
outfname = fname + 'histall.bin'
histall_points.astype(np.float32).tofile(outfname)



