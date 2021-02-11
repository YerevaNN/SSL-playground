# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""
from numpy.core.numeric import True_
from torchvision.utils import save_image

from collections import OrderedDict
import torch
import json
from torch import nn

def frcnn_loss(res):
        final_loss = res['loss_classifier'] + 10 * res['loss_box_reg'] + \
                     res['loss_objectness'] + res['loss_rpn_box_reg']
        return final_loss


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN.

    Arguments:
        backbone (nn.Module):
        rpn (nn.Module):
        heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, backbone, rpn, roi_heads, transform):
        super(GeneralizedRCNN, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        self.is_supervised = True
        self.save_logits = False
    
    def set_is_supervised(self, is_supervised):
        self.is_supervised = is_supervised
    
    def set_save_logits(self, save_logits):
        self.save_logits = save_logits

    def forward(self, images, targets=None, image_paths=None):
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)

        
        if isinstance(features, torch.Tensor):
            features = OrderedDict([(0, features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        # save_image(images.tensors[0], 'image0.png')

        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)


        if not self.training and self.save_logits:
            # curJson = []
            # with open("./logits.json", 'a') as jsonFile:
            #     curJson = json.load(jsonFile)
            preds = []
            for i in range(len(detections)):
                box = detections[i]['boxes']
                score = detections[i]['scores']
                label = detections[i]['labels']
                logit = detections[i]['logits']
                img_boxes = {
                    'img_id': i,
                    'img_path': image_paths[i],
                    'img_predictions': [],
                    'img_features': features['pool'][i].cpu().numpy().tolist()
                }
                for box_id in range(box.shape[0]):
                    cur_box = {}
                    cur_box['bbox'] = box[box_id].cpu().numpy().tolist()
                    cur_box['score'] = score[box_id].cpu().numpy().tolist()
                    cur_box['label'] = label[box_id].cpu().numpy().tolist()
                    cur_box['logits'] = logit[box_id].cpu().numpy().tolist()
                    img_boxes['img_predictions'].append(cur_box)
                preds.append(img_boxes)
            # curJson.append(preds)
            with open("./logits.json", "a") as jsonFile:
                line = json.dumps(preds)
                jsonFile.write(line + '\n')

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if self.training:
            return losses

        return detections
