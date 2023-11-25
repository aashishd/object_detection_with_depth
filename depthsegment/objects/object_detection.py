# from config import OBJECT_DET_PRED_THERSHOLD
import math
from typing import List

import torch
from torchvision.models.detection import (FasterRCNN_ResNet50_FPN_V2_Weights,
                                          fasterrcnn_resnet50_fpn_v2)

from depthsegment.depth.domain import DetectedObject

weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
model.eval()
preprocess = weights.transforms()


def batch_object_detect(imgb: torch.Tensor) -> List[List[DetectedObject]]:
    imgbp = preprocess(imgb)
    batchpreds = model(imgbp)
    detobjects = []
    for i in range(len(batchpreds)):
        detobjects.append(convert_to_objects(batchpreds[i]))
    return detobjects
    
    
def convert_to_objects(imgpred):
    labels = [weights.meta["categories"][i] for i in imgpred["labels"]]
    return [DetectedObject(label, list(map(lambda x: max(0, math.floor(x)), coords))) for label, coords in zip(labels, imgpred["boxes"])]