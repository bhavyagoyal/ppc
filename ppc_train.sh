#!/usr/bin/env bash
#SBATCH --partition=research
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=30G
#SBATCH --time=48:0:0
#SBATCH --exclude=euler07,euler05,euler29,euler30,euler21
###SBATCH --exclude=euler14
#SBATCH -o slurm.%j.%N.out # STDOUT
#SBATCH -e slurm.%j.%N.err # STDERR
#SBATCH --job-name=trainmm
###SBATCH --no-requeue


# Environment Setup

#conda 23.3.1, cuda11.8
#conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
#install mmcv without mim, using pip instead
#pip install "mmcv>=2.0.1" -f  https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html
#pip install mmdet==3.1.0
#git clone https://github.com/open-mmlab/mmdetection3d.git -b dev-1.x
## "-b dev-1.x" means checkout to the `dev-1.x` branch.
#cd mmdetection3d
#pip install -v -e .


# Load Environment
#module load anaconda/mini/23.3.1
module load conda/miniforge/23.1.0
module load nvidia/cuda/11.8.0
bootstrap_conda
conda activate openmmlab2
export CUDA_HOME=/opt/apps/cuda/x86_64/11.8.0/default
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

GPUS=2
PORTUSED=$(( $RANDOM + 10000 ))


# PPC Model Training
EXPERIMENT=work_dir_py/sbr/0.3/jointDP/first50000_spupdated0003_post
DATAPATH=points_min2/0.3/argmax-filtering-sbr/
PORT=${PORTUSED} ./tools/dist_train.sh configs/votenet/votenet_8xb16_sunrgbd-3d.py ${GPUS} --auto-scale-lr --resume --cfg-options \
	train_dataloader.dataset.dataset.data_prefix.pts=${DATAPATH} \
	val_dataloader.dataset.data_prefix.pts=${DATAPATH} \
	train_dataloader.batch_size=8 \
	work_dir=${EXPERIMENT} \
	default_hooks.checkpoint.interval=1 \
	train_dataloader.dataset.dataset.pipeline.4.num_points=50000 \
	val_dataloader.dataset.pipeline.1.transforms.2.num_points=50000 \
	train_dataloader.dataset.dataset.pipeline.0.load_dim=8 \
	val_dataloader.dataset.pipeline.0.load_dim=8 \
	train_dataloader.dataset.dataset.pipeline.0.use_dim="[0,1,2,4]" \
	val_dataloader.dataset.pipeline.0.use_dim="[0,1,2,4]" \
	train_dataloader.dataset.dataset.ann_file='sunrgbd_infos_train_1_100_1_50_5_100_5_50_clean.pkl' \
	val_dataloader.dataset.ann_file='sunrgbd_infos_val_1_100_1_50_5_100_5_50_clean.pkl' \
	param_scheduler.0.end=12 \
	param_scheduler.0.milestones=[8,10] \
	train_cfg.max_epochs=12 \
	train_dataloader.dataset.dataset.pipeline.4.firstk_sampling=True \
	val_dataloader.dataset.pipeline.1.transforms.2.firstk_sampling=True \
	model.data_preprocessor.max_ball_neighbors=64 \
	model.data_preprocessor.ball_radius=0.2 \
	model.data_preprocessor.neighbor_score=0.003 \
	model.data_preprocessor.filter_index=4 \
	model.data_preprocessor.post=True \
	model.data_preprocessor.same_sizes=True \




#	model.backbone.in_channels=6 \
#	model.backbone.sa_mask=True \
#	model.backbone.clip=0.9 \

	#model.bbox_head.proposals_conf=1 \
	#model.bbox_head.clip=0.9 \
	#model.post_sort=4 \
	#model.updated_fps=0.005 \
#	model.max_ball_neighbors=256 \

	#train_dataloader.dataset.dataset.pipeline.0.unit_probabilities=3 \
	#val_dataloader.dataset.pipeline.0.unit_probabilities=3 \


## Baselines 
### Matched Filtering
#EXPERIMENT=work_dir_py/sbr/0.3/joint/point2048
#DATAPATH=points_min2/0.3/argmax-filtering-sbr/
#PORT=${PORTUSED} ./tools/dist_train.sh configs/votenet/votenet_8xb16_sunrgbd-3d.py ${GPUS} --auto-scale-lr --resume --cfg-options \
#	train_dataloader.dataset.dataset.data_prefix.pts=${DATAPATH} \
#	val_dataloader.dataset.data_prefix.pts=${DATAPATH} \
#	train_dataloader.batch_size=8 \
#	work_dir=${EXPERIMENT} \
#	default_hooks.checkpoint.interval=1 \
#	train_dataloader.dataset.dataset.pipeline.4.num_points=2048 \
#	val_dataloader.dataset.pipeline.1.transforms.2.num_points=2048 \
#	train_dataloader.dataset.dataset.pipeline.0.load_dim=8 \
#	val_dataloader.dataset.pipeline.0.load_dim=8 \
#	train_dataloader.dataset.dataset.pipeline.0.use_dim="[0,1,2,4]" \
#	val_dataloader.dataset.pipeline.0.use_dim="[0,1,2,4]" \
#	train_dataloader.dataset.dataset.ann_file='sunrgbd_infos_train_1_100_1_50_5_100_5_50_clean.pkl' \
#	val_dataloader.dataset.ann_file='sunrgbd_infos_val_1_100_1_50_5_100_5_50_clean.pkl' \
#	param_scheduler.0.end=12 \
#	param_scheduler.0.milestones=[8,10] \
#	train_cfg.max_epochs=12 \
#


#### Matched Filtering + Thresholding
#EXPERIMENT=work_dir_py/sbr/0.3/joint/thresh/thresh50000_11
#DATAPATH=points_min2/0.3/argmax-filtering-sbr/
#PORT=${PORTUSED} ./tools/dist_train.sh configs/votenet/votenet_8xb16_sunrgbd-3d.py ${GPUS} --auto-scale-lr --resume --cfg-options \
#	train_dataloader.dataset.dataset.data_prefix.pts=${DATAPATH} \
#	val_dataloader.dataset.data_prefix.pts=${DATAPATH} \
#	train_dataloader.batch_size=8 \
#	work_dir=${EXPERIMENT} \
#	default_hooks.checkpoint.interval=1 \
#	train_dataloader.dataset.dataset.pipeline.4.num_points=50000 \
#	val_dataloader.dataset.pipeline.1.transforms.2.num_points=50000 \
#	train_dataloader.dataset.dataset.pipeline.0.load_dim=8 \
#	val_dataloader.dataset.pipeline.0.load_dim=8 \
#	train_dataloader.dataset.dataset.pipeline.0.use_dim="[0,1,2,3]" \
#	val_dataloader.dataset.pipeline.0.use_dim="[0,1,2,3]" \
#	train_dataloader.dataset.dataset.ann_file='sunrgbd_infos_train_1_100_1_50_5_100_5_50_clean.pkl' \
#	val_dataloader.dataset.ann_file='sunrgbd_infos_val_1_100_1_50_5_100_5_50_clean.pkl' \
#	param_scheduler.0.end=12 \
#	param_scheduler.0.milestones=[8,10] \
#	train_cfg.max_epochs=12 \
#	val_dataloader.dataset.pipeline.1.transforms.2.thresh_sampling=1.1 \
#	train_dataloader.dataset.dataset.pipeline.4.thresh_sampling=1.1 \
#	val_dataloader.dataset.pipeline.1.transforms.2.thresh_index=4 \
#	train_dataloader.dataset.dataset.pipeline.4.thresh_index=4 \




####################### Fusion Models ######################
#
#EXPERIMENT=work_dir_py/imvotenet/0.3/jointDP/first50000_spupdated0003_post
#DATAPATH=points_min2/0.3/argmax-filtering-sbr/
#
## with resume flag, model does not load from "load_from" flag
#CHECKPOINTFILE=${EXPERIMENT}/epoch_1.pth
#if [ -f $CHECKPOINTFILE ]; then
#	RESUMEFLAG=" --resume "
#else
#	RESUMEFLAG=" "
#fi
#
#
#### PPC ####
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
#	train_dataloader.batch_size=8 \
#	train_dataloader.dataset.dataset.pipeline.7.firstk_sampling=True \
#	val_dataloader.dataset.pipeline.3.firstk_sampling=True \
#	model.data_preprocessor.max_ball_neighbors=64 \
#	model.data_preprocessor.ball_radius=0.2 \
#	model.data_preprocessor.neighbor_score=0.003 \
#	model.data_preprocessor.filter_index=5 \
#	model.data_preprocessor.post=True \
#	model.data_preprocessor.same_sizes=True \


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



################ KITTI Dataset ########
#
#EXPERIMENT=work_dir_py/kitti/pvrcnn/3class/joint_5fluxunder2048/0.3/npupdated_ad01
#DATAPATH=training/points2048_r025_dist10/0.3/argmax-filtering-sbr/
#
#PORT=${PORTUSED} ./tools/dist_train.sh configs/pv_rcnn/pv_rcnn_8xb2-80e_kitti-3d-3class.py ${GPUS} --auto-scale-lr --resume --cfg-options \
#	work_dir=${EXPERIMENT} \
#	default_hooks.checkpoint.interval=1 \
#	train_dataloader.dataset.dataset.pipeline.0.use_dim=[0,1,2,5,4,3] \
#	val_dataloader.dataset.pipeline.0.use_dim=[0,1,2,5,4,3] \
#	train_dataloader.dataset.dataset.pipeline.0.load_dim=6 \
#	val_dataloader.dataset.pipeline.0.load_dim=6 \
#	train_dataloader.dataset.dataset.data_prefix.pts=${DATAPATH} \
#	val_dataloader.dataset.data_prefix.pts=${DATAPATH} \
#	train_dataloader.dataset.dataset.pipeline.2.db_sampler.points_loader.use_dim=[0,1,2,5,4,3] \
#	train_dataloader.dataset.dataset.pipeline.2.db_sampler.points_loader.load_dim=6 \
#	train_dataloader.dataset.dataset.pipeline.2.db_sampler.info_path='data/kitti/kitti_dbinfos_train6.pkl' \
#	model.data_preprocessor.in_channels=4 \
#	model.points_encoder.rawpoints_sa_cfgs.in_channels=1 \
#	train_dataloader.dataset.dataset.ann_file='kitti_infos_train_5_1000_5_500_5_250_5_100_clean.pkl' \
#	val_dataloader.dataset.ann_file='kitti_infos_val_5_1000_5_500_5_250_5_100_clean.pkl' \
#	val_evaluator.ann_file='data/kitti/kitti_infos_val_5_1000_5_500_5_250_5_100_clean.pkl' \
#	param_scheduler.0.T_max=4 \
#	param_scheduler.0.end=4 \
#	param_scheduler.1.T_max=6 \
#	param_scheduler.1.begin=4 \
#	param_scheduler.1.end=10 \
#	param_scheduler.2.T_max=4 \
#	param_scheduler.2.end=4 \
#	param_scheduler.3.T_max=6 \
#	param_scheduler.3.begin=4 \
#	param_scheduler.3.end=10 \
#	train_cfg.max_epochs=10 \
#	model.data_preprocessor.neighbor_score=0.1 \
#	model.data_preprocessor.ad_neighbor_score=True \
#	model.data_preprocessor.filter_index=4 \
#	model.data_preprocessor.max_ball_neighbors=32 \
#	model.data_preprocessor.ball_radius=0.8 \


#	train_dataloader.batch_size=1 \


# adding sampling transform in config file
#	train_dataloader.dataset.dataset.pipeline.3.thresh_index=5 \
#	train_dataloader.dataset.dataset.pipeline.3.threshall_sampling=1.0 \
#	train_dataloader.dataset.dataset.pipeline.3.ad_threshall_sampling=True \
#	val_dataloader.dataset.pipeline.1.thresh_index=5 \
#	val_dataloader.dataset.pipeline.1.threshall_sampling=1.0 \
#	val_dataloader.dataset.pipeline.1.ad_threshall_sampling=True \


