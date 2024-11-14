import numpy as np

from mmdet3d.visualization import Det3DLocalVisualizer
import matplotlib
import seaborn as sns
import sys
import pandas

fname = sys.argv[1]
num_points = int(sys.argv[2])
color_idx = int(sys.argv[3])
thresh = float(sys.argv[4])


points = np.fromfile(fname, dtype=np.float32)
print(points.shape)
points = points.reshape(-1,num_points)

#points = pandas.read_csv(fname) 
#points = points.to_numpy()

#choices = np.random.choice(points.shape[0], 5000, replace=False)
choices = points[:,3]>thresh
points = points[choices]

points_color = points[:,color_idx]
#points_color = points[:,5:8]
points_color = points_color/points_color.max()
points_color = sns.color_palette('coolwarm', as_cmap=True)(points_color)[:,:3]

points = points[:,:3]
#print('X ', points[:,0].mean(), ' Y ', points[:,1].mean(), ' Z ', points[:,2].mean(), ' P ', points_color.mean())
visualizer = Det3DLocalVisualizer()

#visualizer.set_points(points, pcd_mode=2, vis_mode='add')
visualizer.set_points(points, pcd_mode=2, vis_mode='add', points_color = points_color)
    
visualizer.show()

