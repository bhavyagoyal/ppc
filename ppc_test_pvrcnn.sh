#!/usr/bin/env bash

SBR=(${1//,/ }) # comma separated argmument: clean,5_1000
if [ -z "$1" ]; then
    SBR=("clean" "5_100" "5_250" "5_500" "5_1000")
fi
CHECKPOINT=$2
if [ -z "$2" ]; then
	CHECKPOINT=checkpoints/pvrcnn/ppc.pth
fi

## PPC Model Testing 
EXPERIMENT=pvrcnn/ppc/

for i in "${!SBR[@]}"
do
DATAPATH=kitti_points_ppc/${SBR[$i]}/
CUDA_VISIBLE_DEVICES=0 python -u tools/test.py configs/pv_rcnn/pv_rcnn_8xb2-80e_kitti-3d-3class.py ${CHECKPOINT} --cfg-options \
	work_dir=mapresults/${EXPERIMENT} \
	test_dataloader.dataset.data_prefix.pts=${DATAPATH} \
	test_dataloader.dataset.pipeline.0.load_dim=6 \
	test_dataloader.dataset.pipeline.0.use_dim="[0,1,2,5,4,3]" \
	model.data_preprocessor.in_channels=4 \
	model.data_preprocessor.neighbor_score=0.1 \
	model.data_preprocessor.filter_index=4 \
	model.data_preprocessor.ad_neighbor_score=True \
	model.data_preprocessor.max_ball_neighbors=32 \
	model.data_preprocessor.ball_radius=0.8 \

done


#### Baselines
#### Matched Filtering
#EXPERIMENT=pvrcnn/matchedfiltering/
#
#for i in "${!SBR[@]}"
#do
#DATAPATH=kitti_points_ppc/${SBR[$i]}/
#CUDA_VISIBLE_DEVICES=0 python -u tools/test.py configs/pv_rcnn/pv_rcnn_8xb2-80e_kitti-3d-3class.py ${CHECKPOINT} --cfg-options \
#	work_dir=mapresults/${EXPERIMENT} \
#	test_dataloader.dataset.data_prefix.pts=${DATAPATH} \
#	test_dataloader.dataset.pipeline.0.load_dim=6 \
#	test_dataloader.dataset.pipeline.0.use_dim="[0,1,2,5]" \
#
#done


### Thresholding
#EXPERIMENT=pvrcnn/thresholding/
#
#for i in "${!SBR[@]}"
#do
#DATAPATH=kitti_points_ppc/${SBR[$i]}/
#CUDA_VISIBLE_DEVICES=0 python -u tools/test.py configs/pv_rcnn/pv_rcnn_8xb2-80e_kitti-3d-3class_thresholding.py ${CHECKPOINT} --cfg-options \
#	work_dir=mapresults/${EXPERIMENT} \
#	test_dataloader.dataset.data_prefix.pts=${DATAPATH} \
#	test_dataloader.dataset.pipeline.0.load_dim=6 \
#	test_dataloader.dataset.pipeline.0.use_dim="[0,1,2,5,4,3]" \
#	test_dataloader.dataset.pipeline.1.thresh_index=5 \
#	test_dataloader.dataset.pipeline.1.threshall_sampling=1.0 \
#	test_dataloader.dataset.pipeline.1.ad_threshall_sampling=True \
#	model.data_preprocessor.in_channels=4 \
#
#
#done