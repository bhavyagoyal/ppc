#!/usr/bin/env bash

SBR=(${1//,/ }) # comma separated argmument: clean,5_50,5_100,1_50,1_100
if [ -z "$1" ]; then
    SBR=("clean" "5_50" "5_100" "1_50" "1_100")
fi
CHECKPOINT=$2
if [ -z "$2" ]; then
    CHECKPOINT=checkpoints/imvotenet/ppc.pth
fi

### PPC Model Testing
EXPERIMENT=imvotenet/ppc/

for i in "${!SBR[@]}"
do
DATAPATH=sunrgbd_points_ppc/${SBR[$i]}/
CUDA_VISIBLE_DEVICES=0 python -u tools/test.py configs/imvotenet/imvotenet_stage2_8xb16_sunrgbd-3d.py ${CHECKPOINT} --cfg-options \
	work_dir=mapresults/${EXPERIMENT} \
	test_dataloader.dataset.data_prefix.pts=${DATAPATH} \
	test_dataloader.dataset.data_prefix.img='sunrgbd_trainval/image' \
	test_dataloader.dataset.pipeline.0.load_dim=8 \
	test_dataloader.dataset.pipeline.3.num_points=50000 \
	test_dataloader.dataset.pipeline.3.firstk_sampling=True \
	test_dataloader.dataset.pipeline.0.use_dim="[0,1,2,3,4]" \
	model.data_preprocessor.max_ball_neighbors=64 \
	model.data_preprocessor.ball_radius=0.2 \
	model.data_preprocessor.neighbor_score=0.003 \
	model.data_preprocessor.filter_index=5 \
	model.data_preprocessor.post=True \

done


#### Baselines
####
##### Matched Filtering
#EXPERIMENT=imvotenet/matchedfiltering/
#
#for i in "${!SBR[@]}"
#do
#DATAPATH=sunrgbd_points_ppc/${SBR[$i]}/
#CUDA_VISIBLE_DEVICES=0 python -u tools/test.py configs/imvotenet/imvotenet_stage2_8xb16_sunrgbd-3d.py ${CHECKPOINT} --cfg-options \
#	work_dir=mapresults/${EXPERIMENT} \
#	test_dataloader.dataset.data_prefix.pts=${DATAPATH} \
#	test_dataloader.dataset.data_prefix.img='sunrgbd_trainval/image' \
#	test_dataloader.dataset.pipeline.0.load_dim=8 \
#	test_dataloader.dataset.pipeline.3.num_points=2048 \
#	test_dataloader.dataset.pipeline.3.firstk_sampling=True \
#	test_dataloader.dataset.pipeline.0.use_dim="[0,1,2]" \
#
#done
#
#
##### Matched Filtering + Thresholding
#EXPERIMENT=imvotenet/thresholding/
#
#for i in "${!SBR[@]}"
#do
#DATAPATH=sunrgbd_points_ppc/${SBR[$i]}/
#CUDA_VISIBLE_DEVICES=0 python -u tools/test.py configs/imvotenet/imvotenet_stage2_8xb16_sunrgbd-3d.py ${CHECKPOINT} --cfg-options \
#	work_dir=mapresults/${EXPERIMENT} \
#	test_dataloader.dataset.data_prefix.pts=${DATAPATH} \
#	test_dataloader.dataset.data_prefix.img='sunrgbd_trainval/image' \
#	test_dataloader.dataset.pipeline.0.load_dim=8 \
#	test_dataloader.dataset.pipeline.3.num_points=50000 \
#	test_dataloader.dataset.pipeline.0.use_dim="[0,1,2,3,4]" \
#	test_dataloader.dataset.pipeline.3.thresh_sampling=1.1 \
#	test_dataloader.dataset.pipeline.3.thresh_index=4 \
#
#done
