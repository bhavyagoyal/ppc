#!/usr/bin/env bash

SBR=(${1//,/ }) # comma separated argmument: clean,5_50,5_100,1_50,1_100
if [ -z "$1" ]; then
    SBR=("clean" "5_50" "5_100" "1_50" "1_100")
fi
CHECKPOINT=$2
if [ -z "$2" ]; then
    CHECKPOINT=checkpoints/votenet/ppc.pth
fi

## PPC Model Testing
EXPERIMENT=votenet/ppc/

for i in "${!SBR[@]}"
do
DATAPATH=sunrgbd_points_ppc/${SBR[$i]}/
CUDA_VISIBLE_DEVICES=0 python -u tools/test.py configs/votenet/votenet_8xb16_sunrgbd-3d.py ${CHECKPOINT} --cfg-options \
	work_dir=mapresults/${EXPERIMENT} \
	test_dataloader.dataset.data_prefix.pts=${DATAPATH} \
	test_dataloader.dataset.pipeline.0.load_dim=8 \
	test_dataloader.dataset.pipeline.1.transforms.2.num_points=50000 \
	test_dataloader.dataset.pipeline.1.transforms.2.firstk_sampling=True \
	test_dataloader.dataset.pipeline.0.use_dim="[0,1,2,4]" \
	model.data_preprocessor.max_ball_neighbors=64 \
	model.data_preprocessor.ball_radius=0.2 \
	model.data_preprocessor.neighbor_score=0.003 \
	model.data_preprocessor.filter_index=4 \
	model.data_preprocessor.post=True \
	model.data_preprocessor.same_sizes=True \

done


### Baselines
### Matched Filtering
#EXPERIMENT=votenet/matchedfiltering/
#
#for i in "${!SBR[@]}"
#do
#DATAPATH=sunrgbd_points_ppc/${SBR[$i]}/
#CUDA_VISIBLE_DEVICES=0 python -u tools/test.py configs/votenet/votenet_8xb16_sunrgbd-3d.py ${CHECKPOINT} --cfg-options \
#	work_dir=mapresults/${EXPERIMENT} \
#	test_dataloader.dataset.data_prefix.pts=${DATAPATH} \
#	test_dataloader.dataset.pipeline.0.load_dim=8 \
#	test_dataloader.dataset.pipeline.1.transforms.2.num_points=2048 \
#	test_dataloader.dataset.pipeline.0.use_dim="[0,1,2]" \
#
#done
#
#
#### Matched Filtering + Thresholding
#EXPERIMENT=votenet/thresholding/
#
#for i in "${!SBR[@]}"
#do
#DATAPATH=sunrgbd_points_ppc/${SBR[$i]}/
#CUDA_VISIBLE_DEVICES=0 python -u tools/test.py configs/votenet/votenet_8xb16_sunrgbd-3d.py ${CHECKPOINT} --cfg-options \
#	work_dir=mapresults/${EXPERIMENT} \
#	test_dataloader.dataset.data_prefix.pts=${DATAPATH} \
#	test_dataloader.dataset.pipeline.0.load_dim=8 \
#	test_dataloader.dataset.pipeline.1.transforms.2.num_points=50000 \
#	test_dataloader.dataset.pipeline.1.transforms.2.thresh_sampling=1.1 \
#	test_dataloader.dataset.pipeline.1.transforms.2.thresh_index=4 \
#	test_dataloader.dataset.pipeline.0.use_dim="[0,1,2,3]" \
#
#done
