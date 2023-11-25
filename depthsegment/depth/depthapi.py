import logging
from typing import List

import torch

from depthsegment.depth.depth_detection import batch_depth
from depthsegment.depth.domain import ImageWithDepth, InputImage
from depthsegment.depth.transform import build_out_transform, input_transform

LOG = logging.getLogger(__name__)

def run_depth_est_batch(images: List[InputImage]) -> List[ImageWithDepth]:
    # transfrom all the input images
    tinput = torch.cat([input_transform(img.image()).unsqueeze(0) for img in images], 0)

    # run the depth estimation model
    outdepths = batch_depth(tinput)
    
    # run the object detection model
    
    

    # reshape output depths
    depthsresh = [
        build_out_transform(img.imgdims())(outdepths[i].unsqueeze(0))
        for i, img in enumerate(images)
    ]

    return [
        ImageWithDepth(img, depthsresh[i].squeeze(0)) for i, img in enumerate(images)
    ]