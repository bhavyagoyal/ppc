import pickle
import copy
import numpy as np
import random
import os
splits = ['train', 'val']

dataset='sunrgbd'
#dataset='kitti'

if(dataset=='sunrgbd'):
    BASE = '../../data/sunrgbd/'
    INBASEFOLDER = os.path.join(BASE, 'points')
    OUTBASEFOLDER = os.path.join(BASE, 'points_min2/0.3/argmax-filtering-sbr', 'clean')
    SBR = ['1_100', '1_50', '5_100', '5_50', 'clean']
    startdataidx, enddataidx = 1, 10335
    num_feats = 6
else:
    BASE = '../../data/kitti/'
    INBASEFOLDER = os.path.join(BASE, 'training/velodyne_reduced')
    OUTBASEFOLDER = os.path.join(BASE, 'training/points2048_r025_dist10/0.3/argmax-filtering-sbr', 'clean')
    SBR = ['5_1000', '5_500', '5_250', '5_100', 'clean']
    startdataidx, enddataidx = 0, 7480
    num_feats = 4
    

for sp in splits:
    with open(os.path.join(BASE, dataset + '_infos_' + sp + '.pkl'), 'rb') as f:
        data = pickle.load(f)
    data_list = data['data_list']

    #r = np.random.choice(range(len(data_list)), 100, replace=False).tolist()
    #final_list = [data_list[x] for x in r]

    # create data list with all sbr levels
    final_list = []
    for idx, _ in enumerate(data_list):
        for idy,sbr in enumerate(SBR):
            data_elem = copy.deepcopy(data_list[idx])
            data_elem['lidar_points']['lidar_path'] = sbr + '/' + data_elem['lidar_points']['lidar_path']
            data_elem['lidar_points']['num_pts_feats'] = num_feats + 2
            if(dataset=='kitti'):
                data_elem['sample_idx'] = data_elem['sample_idx']*10+idy
            final_list.append(data_elem)
    print(len(final_list))
    
    data['data_list'] = final_list
    SBRstr = '_'.join(SBR)
    with open(os.path.join(BASE, dataset + '_infos_' + sp + '_' + SBRstr + '.pkl'), 'wb') as f:
        pickle.dump(data, f)

    ### Old code for gaussian noise testing
    #    #np.savetxt(save_path, pcl_denoised.numpy(), fmt='%.8f')
    #    #noise_max = [0.01, 0.1, 1.0]
    #        #points = np.fromfile('../../data/sunrgbd/points_min2/0.3/argmax-filtering-sbr/5_50/'+ data_list[idx]['lidar_points']['lidar_path'], dtype=np.float32)
    #        #points = np.fromfile(BASE+data_list[idx]['lidar_points']['lidar_path'], dtype=np.float32)
    ##        points = points.reshape(-1,8)
    ##        points3d = points[:,:3]
    ##        for noise in noise_max:
    ##            noise_std = random.uniform(0, noise)
    ##            points3d = points3d + np.random.normal(size=points3d.shape) * noise_std
    ##            pointsout = np.concatenate([points3d.astype(np.float32), points[:,3:]], axis=1)
    ##            pointsout.tofile(OUTBASE + str(noise) + '/' + data_list[idx]['lidar_points']['lidar_path'])
    ##        #np.savetxt('txt/points_clean8/' + sp + '/' + data_list[idx]['lidar_points']['lidar_path'][:-4]+'.xyz', points, fmt='%.8f')


    # convert original point cloud to probabilistic point cloud
    # by adding an extra feature of probability 1 to each point
    for i in range(startdataidx, enddataidx+1):
        fname = str(i).zfill(6) + '.bin'
        print(fname)
        points = np.fromfile(os.path.join(INBASEFOLDER, fname), dtype=np.float32)
        points = points.reshape(-1,num_feats)
        ones = np.ones((points.shape[0], 1), dtype=np.float32)
        points = np.concatenate([points[:,:3], ones*1000., ones, points[:,3:]], 1)
        if not os.path.exists(OUTBASEFOLDER):
            os.makedirs(OUTBASEFOLDER)
        points.tofile(os.path.join(OUTBASEFOLDER,fname))


# Object Sampling transformation for KITTI uses a DB of object point clouds
# converting them to probabilistic point clouds as well by adding 1 probability to each point
if(dataset=='kitti'):   
    with open(os.path.join(BASE, 'kitti_dbinfos_train.pkl'), 'rb') as f:
        data = pickle.load(f)
    
    for cat in data.keys():
        objs = data[cat]
        for idx, obj in enumerate(objs):
            obj_file = os.path.join(BASE, obj['path'])
            print(obj_file)
            out_obj_file = obj_file.replace('kitti_gt_database', 'kitti_gt_database6')

            points = np.fromfile(obj_file, dtype=np.float32)
            points = points.reshape(-1,4)
            ones = np.ones((points.shape[0], 1), dtype=np.float32)
            points = np.concatenate([points[:,:3], ones*1000., ones, points[:,3:]], 1)
            objs[idx]['path'] = out_obj_file

            outfolder = '/'.join(out_obj_file.split('/')[:-1]) 
            if not os.path.exists(outfolder):
                os.makedirs(outfolder)
            points.tofile(out_obj_file)
        data[cat]=objs 
    
    with open(os.path.join(BASE,'kitti_dbinfos_train6.pkl'), "wb") as f:
        pickle.dump(data, f)
        
    
