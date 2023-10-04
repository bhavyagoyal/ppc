#!/usr/bin/env bash

#GPUS=4
GPUS=1
PORTUSED=$(( $RANDOM + 10000 ))


### PPC Model
EXPERIMENT=work_dir_py/kitti/pvrcnn/3class/joint_5fluxunder2048/0.3/npupdated_ad01
DATAPATH=training/points2048_r025_dist10/0.3/argmax-filtering-sbr/

PORT=${PORTUSED} ./tools/dist_train.sh configs/pv_rcnn/pv_rcnn_8xb2-80e_kitti-3d-3class.py ${GPUS} --auto-scale-lr --resume --cfg-options \
	work_dir=${EXPERIMENT} \
	default_hooks.checkpoint.interval=1 \
	train_dataloader.dataset.dataset.pipeline.0.use_dim=[0,1,2,5,4,3] \
	val_dataloader.dataset.pipeline.0.use_dim=[0,1,2,5,4,3] \
	train_dataloader.dataset.dataset.pipeline.0.load_dim=6 \
	val_dataloader.dataset.pipeline.0.load_dim=6 \
	train_dataloader.dataset.dataset.data_prefix.pts=${DATAPATH} \
	val_dataloader.dataset.data_prefix.pts=${DATAPATH} \
	train_dataloader.dataset.dataset.pipeline.2.db_sampler.points_loader.use_dim=[0,1,2,5,4,3] \
	train_dataloader.dataset.dataset.pipeline.2.db_sampler.points_loader.load_dim=6 \
	train_dataloader.dataset.dataset.pipeline.2.db_sampler.info_path='data/kitti/kitti_dbinfos_train6.pkl' \
	model.data_preprocessor.in_channels=4 \
	model.points_encoder.rawpoints_sa_cfgs.in_channels=1 \
	train_dataloader.dataset.dataset.ann_file='kitti_infos_train_5_1000_5_500_5_250_5_100_clean.pkl' \
	val_dataloader.dataset.ann_file='kitti_infos_val_5_1000_5_500_5_250_5_100_clean.pkl' \
	val_evaluator.ann_file='data/kitti/kitti_infos_val_5_1000_5_500_5_250_5_100_clean.pkl' \
	param_scheduler.0.T_max=4 \
	param_scheduler.0.end=4 \
	param_scheduler.1.T_max=6 \
	param_scheduler.1.begin=4 \
	param_scheduler.1.end=10 \
	param_scheduler.2.T_max=4 \
	param_scheduler.2.end=4 \
	param_scheduler.3.T_max=6 \
	param_scheduler.3.begin=4 \
	param_scheduler.3.end=10 \
	train_cfg.max_epochs=10 \
	model.data_preprocessor.neighbor_score=0.1 \
	model.data_preprocessor.ad_neighbor_score=True \
	model.data_preprocessor.filter_index=4 \
	model.data_preprocessor.max_ball_neighbors=32 \
	model.data_preprocessor.ball_radius=0.8 \





# ### Matched Filtering
# EXPERIMENT=work_dir_py/kitti/pvrcnn/3class/joint_5fluxunder2048/0.3/baseline
# DATAPATH=training/points2048_r025_dist10/0.3/argmax-filtering-sbr/

# PORT=${PORTUSED} ./tools/dist_train.sh configs/pv_rcnn/pv_rcnn_8xb2-80e_kitti-3d-3class.py ${GPUS} --auto-scale-lr --resume --cfg-options \
# 	work_dir=${EXPERIMENT} \
# 	default_hooks.checkpoint.interval=1 \
# 	train_dataloader.dataset.dataset.pipeline.0.use_dim=[0,1,2,5,4,3] \
# 	val_dataloader.dataset.pipeline.0.use_dim=[0,1,2,5,4,3] \
# 	train_dataloader.dataset.dataset.pipeline.0.load_dim=6 \
# 	val_dataloader.dataset.pipeline.0.load_dim=6 \
# 	train_dataloader.dataset.dataset.data_prefix.pts=${DATAPATH} \
# 	val_dataloader.dataset.data_prefix.pts=${DATAPATH} \
# 	train_dataloader.dataset.dataset.pipeline.2.db_sampler.points_loader.use_dim=[0,1,2,5,4,3] \
# 	train_dataloader.dataset.dataset.pipeline.2.db_sampler.points_loader.load_dim=6 \
# 	train_dataloader.dataset.dataset.pipeline.2.db_sampler.info_path='data/kitti/kitti_dbinfos_train6.pkl' \
# 	model.data_preprocessor.in_channels=4 \
# 	model.points_encoder.rawpoints_sa_cfgs.in_channels=1 \
# 	train_dataloader.dataset.dataset.ann_file='kitti_infos_train_5_1000_5_500_5_250_5_100_clean.pkl' \
# 	val_dataloader.dataset.ann_file='kitti_infos_val_5_1000_5_500_5_250_5_100_clean.pkl' \
# 	val_evaluator.ann_file='data/kitti/kitti_infos_val_5_1000_5_500_5_250_5_100_clean.pkl' \
# 	param_scheduler.0.T_max=4 \
# 	param_scheduler.0.end=4 \
# 	param_scheduler.1.T_max=6 \
# 	param_scheduler.1.begin=4 \
# 	param_scheduler.1.end=10 \
# 	param_scheduler.2.T_max=4 \
# 	param_scheduler.2.end=4 \
# 	param_scheduler.3.T_max=6 \
# 	param_scheduler.3.begin=4 \
# 	param_scheduler.3.end=10 \
# 	train_cfg.max_epochs=10 \



# ### Thresholding
# EXPERIMENT=work_dir_py/kitti/pvrcnn/3class/joint_5fluxunder2048/0.3/thresh_ad10
# DATAPATH=training/points2048_r025_dist10/0.3/argmax-filtering-sbr/

# PORT=${PORTUSED} ./tools/dist_train.sh configs/pv_rcnn/pv_rcnn_8xb2-80e_kitti-3d-3class_thresholding.py ${GPUS} --auto-scale-lr --resume --cfg-options \
# 	work_dir=${EXPERIMENT} \
# 	default_hooks.checkpoint.interval=1 \
# 	train_dataloader.dataset.dataset.pipeline.0.use_dim=[0,1,2,5,4,3] \
# 	val_dataloader.dataset.pipeline.0.use_dim=[0,1,2,5,4,3] \
# 	train_dataloader.dataset.dataset.pipeline.0.load_dim=6 \
# 	val_dataloader.dataset.pipeline.0.load_dim=6 \
# 	train_dataloader.dataset.dataset.data_prefix.pts=${DATAPATH} \
# 	val_dataloader.dataset.data_prefix.pts=${DATAPATH} \
# 	train_dataloader.dataset.dataset.pipeline.2.db_sampler.points_loader.use_dim=[0,1,2,5,4,3] \
# 	train_dataloader.dataset.dataset.pipeline.2.db_sampler.points_loader.load_dim=6 \
# 	train_dataloader.dataset.dataset.pipeline.2.db_sampler.info_path='data/kitti/kitti_dbinfos_train6.pkl' \
# 	model.data_preprocessor.in_channels=4 \
# 	model.points_encoder.rawpoints_sa_cfgs.in_channels=1 \
# 	train_dataloader.dataset.dataset.ann_file='kitti_infos_train_5_1000_5_500_5_250_5_100_clean.pkl' \
# 	val_dataloader.dataset.ann_file='kitti_infos_val_5_1000_5_500_5_250_5_100_clean.pkl' \
# 	val_evaluator.ann_file='data/kitti/kitti_infos_val_5_1000_5_500_5_250_5_100_clean.pkl' \
# 	param_scheduler.0.T_max=4 \
# 	param_scheduler.0.end=4 \
# 	param_scheduler.1.T_max=6 \
# 	param_scheduler.1.begin=4 \
# 	param_scheduler.1.end=10 \
# 	param_scheduler.2.T_max=4 \
# 	param_scheduler.2.end=4 \
# 	param_scheduler.3.T_max=6 \
# 	param_scheduler.3.begin=4 \
# 	param_scheduler.3.end=10 \
# 	train_cfg.max_epochs=10 \
# 	train_dataloader.dataset.dataset.pipeline.3.thresh_index=5 \
# 	train_dataloader.dataset.dataset.pipeline.3.threshall_sampling=1.0 \
# 	train_dataloader.dataset.dataset.pipeline.3.ad_threshall_sampling=True \
# 	val_dataloader.dataset.pipeline.1.thresh_index=5 \
# 	val_dataloader.dataset.pipeline.1.threshall_sampling=1.0 \
# 	val_dataloader.dataset.pipeline.1.ad_threshall_sampling=True \





##########
#	train_dataloader.batch_size=1 \






