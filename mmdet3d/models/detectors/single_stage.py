# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple, Union

import torch
from torch import Tensor

from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from ...structures.det3d_data_sample import OptSampleList, SampleList
from .base import Base3DDetector
from mmcv.ops.ball_query import ball_query


@MODELS.register_module()
class SingleStage3DDetector(Base3DDetector):
    """SingleStage3DDetector.

    This class serves as a base class for single-stage 3D detectors which
    directly and densely predict 3D bounding boxes on the output features
    of the backbone+neck.


    Args:
        backbone (dict): Config dict of detector's backbone.
        neck (dict, optional): Config dict of neck. Defaults to None.
        bbox_head (dict, optional): Config dict of box head. Defaults to None.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 #neighbor_score: float = None,
                 updated_fps: float = None,
                 #weighted_filtering_score: bool = False,
                 post_sort: int = None,
                 #filter_index: int = 5,
                 #max_ball_neighbors: int = 64,
                 #max_ball_radius: float = 0.2,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        #self.neighbor_score = neighbor_score
        self.updated_fps = updated_fps
        #self.weighted_filtering_score = weighted_filtering_score
        self.post_sort = post_sort
        #self.filter_index = filter_index
        #self.max_ball_neighbors = max_ball_neighbors
        #self.max_ball_radius = max_ball_radius

    def loss(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
             **kwargs) -> Union[dict, list]:
        """Calculate losses from a batch of inputs dict and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points', 'img' keys.

                    - points (list[torch.Tensor]): Point cloud of each sample.
                    - imgs (torch.Tensor, optional): Image of each sample.

            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

        Returns:
            dict: A dictionary of loss components.
        """
        x = self.extract_feat(batch_inputs_dict)
        losses = self.bbox_head.loss(x, batch_data_samples, **kwargs)
        return losses

    def predict(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                **kwargs) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points', 'img' keys.

                    - points (list[torch.Tensor]): Point cloud of each sample.
                    - imgs (torch.Tensor, optional): Image of each sample.

            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input samples. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

                - scores_3d (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels_3d (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes_3d (Tensor): Contains a tensor with shape
                    (num_instances, C) where C >=7.
        """
        x = self.extract_feat(batch_inputs_dict)
        results_list = self.bbox_head.predict(x, batch_data_samples, **kwargs)
        predictions = self.add_pred_to_datasample(batch_data_samples,
                                                  results_list)
        return predictions

    def _forward(self,
                 batch_inputs_dict: dict,
                 data_samples: OptSampleList = None,
                 **kwargs) -> Tuple[List[torch.Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs_dict (dict): The model input dict which include
                'points', 'img' keys.

                    - points (list[torch.Tensor]): Point cloud of each sample.
                    - imgs (torch.Tensor, optional): Image of each sample.

            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

        Returns:
            tuple[list]: A tuple of features from ``bbox_head`` forward.
        """
        x = self.extract_feat(batch_inputs_dict)
        results = self.bbox_head.forward(x)
        return results

    def extract_feat(
        self, batch_inputs_dict: Dict[str, Tensor]
    ) -> Union[Tuple[torch.Tensor], Dict[str, Tensor]]:
        """Directly extract features from the backbone+neck.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points', 'img' keys.

                    - points (list[torch.Tensor]): Point cloud of each sample.
                    - imgs (torch.Tensor, optional): Image of each sample.

        Returns:
            tuple[Tensor] | dict:  For outside 3D object detection, we
                typically obtain a tuple of features from the backbone + neck,
                and for inside 3D object detection, usually a dict containing
                features will be obtained.
        """
        points = batch_inputs_dict['points']
        stack_points = torch.stack(points)

        #### This is old code for NPD filtering. I'm just using data_preprocessing.py now
        #K = 1000 # using a large number for xyz cordinates of filtered points
        #B = stack_points.shape[0]
 
        #if(self.neighbor_score):
        #    points_xyz = stack_points[:,:,:3].detach().contiguous()
        #    points_probs = stack_points[:,:,self.filter_index].detach().contiguous()
        #    #self.backbone.SA_modules[0].groupers[0].sample_num
        #    #self.backbone.SA_modules[0].groupers[0].max_radius
        #    ball_idxs = ball_query(0, self.max_ball_radius, self.max_ball_neighbors, points_xyz, points_xyz).long()

        #    # ball query returns repeats first neighbor if neighbors are fewer than requested
        #    # so, ignore the first neighbor in ball query output
        #    ball_idxs_first = ball_idxs[:,:,0][:,:,None]
        #    nonzero_ball_idxs = ((ball_idxs-ball_idxs_first)!=0)
        #    nonzero_count = nonzero_ball_idxs.sum(-1)

        #    points_probs_tiled = points_probs[:,:,None].tile(self.max_ball_neighbors)
        #    neighbor_probs = torch.gather(points_probs_tiled, 1, ball_idxs) 
        #    neighbor_probs = neighbor_probs*nonzero_ball_idxs
        #    neighbor_probs = neighbor_probs.mean(-1)
        #    #neighbor_probs_weighted = neighbor_probs*points_probs
        #   
        #    #if(self.weighted_filtering_score):
        #    #    ignore_points = neighbor_probs_weighted<self.neighbor_score
        #    #    stack_points = torch.concatenate([stack_points, neighbor_probs_weighted[...,None]], axis=-1)
        #    #    #ignore_points = nonzero_count<self.neighbor_score
        #    #else: 
        #    ignore_points = neighbor_probs<self.neighbor_score
        #    stack_points = torch.concatenate([stack_points, neighbor_probs[...,None]], axis=-1)

        #    # This updates stack_points
        #    stack_points_xyzfh = stack_points[...,:4]
        #    stack_points_feat = stack_points[...,4:]
        #    stack_points_xyzfh[ignore_points]=-K
        #    stack_points_feat[ignore_points]=0

        #    neighbor_probs[ignore_points]=0
        #    #neighbor_probs_weighted[ignore_points]=0

        #    #if(self.post_sort is not None):
        #    choices = torch.argsort(stack_points[:,:,self.filter_index], descending=True)
        #    stack_points = torch.gather(stack_points, 1, choices[:,:,None].tile(stack_points.shape[2]))
        #    
        #    # Remove points that are zero probability from filtering step
        #    # Since mini batch will have different number of points now
        #    # only remove from maximum index of zero probability in the batch
        #    values = torch.zeros(stack_points.shape[0],1).cuda()
        #    crop_index = torch.searchsorted(stack_points[...,self.filter_index]*-1, values)[:,0]
        #    stack_points = stack_points[:,:crop_index.max(),:]

        #    # Set the rest of the points to the first point
        #    # replace points with -K value to K+firstpoint
        #    mask = (stack_points == -K)
        #    first_point = stack_points[:,:1,:].tile(1,stack_points.shape[1],1)+K
        #    first_point = first_point*(mask.int())
        #    stack_points += first_point


        stack_points_nonfps = None
        if(self.updated_fps):
            stack_points_nonfps = stack_points.clone().detach()
            choices = torch.argsort(stack_points[:,:,self.post_sort], descending=True)
            stack_points = torch.gather(stack_points, 1, choices[:,:,None].tile(stack_points.shape[2]))

            values = torch.ones(stack_points.shape[0],1).cuda()*-1*self.updated_fps
            new_fps = torch.searchsorted(stack_points[...,self.post_sort]*-1, values)[:,0]
            new_fps = new_fps.median()
            stack_points = stack_points[:,:new_fps,:]

            ordering = torch.rand(B, stack_points.shape[1], device=stack_points.device).argsort(-1)
            stack_points = torch.gather(stack_points, 1, ordering[:,:,None].tile(stack_points.shape[2]))

            #self.backbone.SA_modules[0].fps_sample_range_list[0]=new_fps if new_fps<stack_points.shape[1] else -1
            batch_inputs_dict['points'] = torch.unbind(stack_points)


        #if(self.neighbor_score or self.updated_fps):
        #    batch_inputs_dict['points'] = torch.unbind(stack_points)
        x = self.backbone(stack_points, stack_points_nonfps)
        #x = self.backbone(stack_points)
        if self.with_neck:
            x = self.neck(x)
        return x
