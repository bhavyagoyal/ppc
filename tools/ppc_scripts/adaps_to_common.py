import numpy as np
import numpy.linalg
import sys
import pandas


fname = sys.argv[1]
raw_hist_fname = fname.replace('PointCloud', 'RawDataHistogramMap').replace('xyz', 'txt')

points = pandas.read_csv(fname, header=None)
points = points.to_numpy()*[1,-1,1]
points = points[:,[2,0,1]]

raw_hist = pandas.read_csv(raw_hist_fname, header=None, delim_whitespace=True)
raw_hist = raw_hist.to_numpy()
print(fname, raw_hist.shape, points.shape)

dist = np.linalg.norm(points, axis=1)
keep_points = dist>=1.0
keep_points = keep_points.astype(np.float32)
points = points*keep_points[:,None]

#TOTAL = 49152
#density = np.ones(TOTAL)
#conf = np.linspace(0,1,TOTAL)
density = raw_hist.max(axis=1)
print(density.mean(), density.max(), np.count_nonzero(density), keep_points.sum())
total = raw_hist.sum(axis=1)
total = np.clip(total, 1, None)
conf = density/total 
reflec = density/density.max()

points = np.concatenate([points, density[:,None], conf[:,None], np.tile(reflec[:,None], (1,3))], axis=1)
print(points.shape)

outfname = fname + '.bin'
points.astype(np.float32).tofile(outfname)



