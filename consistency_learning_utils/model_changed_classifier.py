from .nets.detection.faster_rcnn import fasterrcnn_resnet50_fpn
from torch import nn
import torch

class SkipConnection(nn.Module):

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        z = self.module(x)
        return torch.cat((x, z), dim=-1)

def model_changed_classifier(reuse_classifier=False, initialize=False, class_num=20, gamma=1, box_score_thresh=0.05):
    """
    Args:
        reuse_classifier:
            - False: a regular Faster RCNN is used.
            - Otherwise: take a Faster RCNN and add a new layer on top of:
                - 'add': the original logits
                - 'concatenate': the concatenation of original logits and the feature vector
        initialize: initialization of Faster RCNN
            - False: random
            - 'backbone': ImageNet-pretrained ResNet backbone only
            - 'full': COCO pretrained Faster RCNN
        class_num: number of foreground classes (the code will add one for background)
    Returns:
        Initialized model
    """
    pretrained = initialize == 'full'
    backbone = initialize == 'backbone'

    if reuse_classifier is False:
        model = fasterrcnn_resnet50_fpn(pretrained=pretrained, progress=True,
                                        num_classes=class_num+1,
                                        pretrained_backbone=backbone, gamma=gamma,
                                        box_score_thresh=box_score_thresh)
        return model

    model = fasterrcnn_resnet50_fpn(pretrained=pretrained, progress=True,
                                    num_classes=91, pretrained_backbone=backbone,
                                    gamma=gamma, box_score_thresh=box_score_thresh)

    if reuse_classifier == 'add':
        old_cls_score = model.roi_heads.box_predictor.cls_score
        old_bbox_pred = model.roi_heads.box_predictor.bbox_pred

        new_cls_score = nn.Sequential(
            old_cls_score,  # 1024 -> 91
            nn.Linear(in_features=91, out_features=class_num+1, bias=True)
        )

        new_bbox_pred = nn.Sequential(
            old_bbox_pred,
            nn.Linear(in_features=364, out_features=4*(class_num+1), bias=True)
        )
    elif reuse_classifier == 'concatenate':
        old_cls_score = SkipConnection(model.roi_heads.box_predictor.cls_score)
        old_bbox_pred = SkipConnection(model.roi_heads.box_predictor.bbox_pred)

        new_cls_score = nn.Sequential(
            old_cls_score,  # 1024 -> 91
            nn.Linear(in_features=91+1024, out_features=class_num+1, bias=True)
        )

        new_bbox_pred = nn.Sequential(
            old_bbox_pred,
            nn.Linear(in_features=364+1024, out_features=4*(class_num+1), bias=True)
        )
    else:
        raise NotImplementedError

    model.roi_heads.box_predictor.cls_score = new_cls_score
    model.roi_heads.box_predictor.bbox_pred = new_bbox_pred

    return model
