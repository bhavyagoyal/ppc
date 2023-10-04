#!/usr/bin/env bash

#GPUS=4
GPUS=1
PORTUSED=$(( $RANDOM + 10000 ))


####################### Fusion Models ######################

EXPERIMENT=work_dir_py/imvotenet/0.3/jointDP/first50000_spupdated0003_post
DATAPATH=points_min2/0.3/argmax-filtering-sbr/

# If using resume flag, model ignores "load_from" flag.
# Using load_from flag for starting weights.
# So, use resume flag only in the beginning.
CHECKPOINTFILE=${EXPERIMENT}/epoch_1.pth
if [ -f $CHECKPOINTFILE ]; then
	RESUMEFLAG=" --resume "
else
	RESUMEFLAG=" "
fi




### PPC Model ####
PORT=${PORTUSED} ./tools/dist_train.sh configs/imvotenet/imvotenet_stage2_8xb16_sunrgbd-3d.py ${GPUS} ${RESUMEFLAG} --auto-scale-lr --cfg-options \
	work_dir=${EXPERIMENT} \
	default_hooks.checkpoint.interval=1 \
	load_from="checkpoints/imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class_20210819_225618-62eba6ce.pth" \
	train_dataloader.dataset.dataset.data_prefix.pts=${DATAPATH} \
	val_dataloader.dataset.data_prefix.pts=${DATAPATH} \
	train_dataloader.dataset.dataset.data_prefix.img='sunrgbd_trainval/image' \
	val_dataloader.dataset.data_prefix.img='sunrgbd_trainval/image' \
	train_dataloader.dataset.dataset.pipeline.0.load_dim=8 \
	val_dataloader.dataset.pipeline.0.load_dim=8 \
	train_dataloader.dataset.dataset.pipeline.0.use_dim="[0,1,2,3,4,5,6,7]" \
	val_dataloader.dataset.pipeline.0.use_dim="[0,1,2,3,4,5,6,7]" \
	train_dataloader.dataset.dataset.ann_file='sunrgbd_infos_train_1_100_1_50_5_100_5_50_clean.pkl' \
	val_dataloader.dataset.ann_file='sunrgbd_infos_val_1_100_1_50_5_100_5_50_clean.pkl' \
	param_scheduler.0.end=12 \
	param_scheduler.0.milestones=[8,10] \
	train_cfg.max_epochs=12 \
	train_dataloader.dataset.dataset.pipeline.7.num_points=50000 \
	val_dataloader.dataset.pipeline.3.num_points=50000 \
	train_dataloader.batch_size=8 \
	train_dataloader.dataset.dataset.pipeline.7.firstk_sampling=True \
	val_dataloader.dataset.pipeline.3.firstk_sampling=True \
	model.data_preprocessor.max_ball_neighbors=64 \
	model.data_preprocessor.ball_radius=0.2 \
	model.data_preprocessor.neighbor_score=0.003 \
	model.data_preprocessor.filter_index=5 \
	model.data_preprocessor.post=True \
	model.data_preprocessor.same_sizes=True \


#
#### Matched Filtering ####
#PORT=${PORTUSED} ./tools/dist_train.sh configs/imvotenet/imvotenet_stage2_8xb16_sunrgbd-3d.py ${GPUS} ${RESUMEFLAG} --auto-scale-lr --cfg-options \
#	work_dir=${EXPERIMENT} \
#	default_hooks.checkpoint.interval=1 \
#	load_from="checkpoints/imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class_20210819_225618-62eba6ce.pth" \
#	train_dataloader.dataset.dataset.data_prefix.pts=${DATAPATH} \
#	val_dataloader.dataset.data_prefix.pts=${DATAPATH} \
#	train_dataloader.dataset.dataset.data_prefix.img='sunrgbd_trainval/image' \
#	val_dataloader.dataset.data_prefix.img='sunrgbd_trainval/image' \
#	train_dataloader.dataset.dataset.pipeline.0.load_dim=8 \
#	val_dataloader.dataset.pipeline.0.load_dim=8 \
#	train_dataloader.dataset.dataset.pipeline.0.use_dim="[0,1,2,3,4,5,6,7]" \
#	val_dataloader.dataset.pipeline.0.use_dim="[0,1,2,3,4,5,6,7]" \
#	train_dataloader.dataset.dataset.ann_file='sunrgbd_infos_train_1_100_1_50_5_100_5_50_clean.pkl' \
#	val_dataloader.dataset.ann_file='sunrgbd_infos_val_1_100_1_50_5_100_5_50_clean.pkl' \
#	param_scheduler.0.end=12 \
#	param_scheduler.0.milestones=[8,10] \
#	train_cfg.max_epochs=12 \
#	train_dataloader.dataset.dataset.pipeline.7.num_points=2048 \
#	val_dataloader.dataset.pipeline.3.num_points=2048 \
#	train_dataloader.batch_size=8 \



#### Thresholding ####
#PORT=${PORTUSED} ./tools/dist_train.sh configs/imvotenet/imvotenet_stage2_8xb16_sunrgbd-3d.py ${GPUS} ${RESUMEFLAG} --auto-scale-lr --cfg-options \
#	work_dir=${EXPERIMENT} \
#	default_hooks.checkpoint.interval=1 \
#	load_from="checkpoints/imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class_20210819_225618-62eba6ce.pth" \
#	train_dataloader.dataset.dataset.data_prefix.pts=${DATAPATH} \
#	val_dataloader.dataset.data_prefix.pts=${DATAPATH} \
#	train_dataloader.dataset.dataset.data_prefix.img='sunrgbd_trainval/image' \
#	val_dataloader.dataset.data_prefix.img='sunrgbd_trainval/image' \
#	train_dataloader.dataset.dataset.pipeline.0.load_dim=8 \
#	val_dataloader.dataset.pipeline.0.load_dim=8 \
#	train_dataloader.dataset.dataset.pipeline.0.use_dim="[0,1,2,3,4,5,6,7]" \
#	val_dataloader.dataset.pipeline.0.use_dim="[0,1,2,3,4,5,6,7]" \
#	train_dataloader.dataset.dataset.ann_file='sunrgbd_infos_train_1_100_1_50_5_100_5_50_clean.pkl' \
#	val_dataloader.dataset.ann_file='sunrgbd_infos_val_1_100_1_50_5_100_5_50_clean.pkl' \
#	param_scheduler.0.end=12 \
#	param_scheduler.0.milestones=[8,10] \
#	train_cfg.max_epochs=12 \
#	train_dataloader.dataset.dataset.pipeline.7.num_points=50000 \
#	val_dataloader.dataset.pipeline.3.num_points=50000 \
#	train_dataloader.dataset.dataset.pipeline.7.thresh_sampling=1.1 \
#	train_dataloader.dataset.dataset.pipeline.7.thresh_index=4 \
#	val_dataloader.dataset.pipeline.3.thresh_sampling=1.1 \
#	val_dataloader.dataset.pipeline.3.thresh_index=4 \
#	train_dataloader.batch_size=8 \
