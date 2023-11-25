from typing import List

import torch
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_bounding_boxes


def detections_to_pilimg(img: torch.Tensor, labels: List[str], coordinates: List[List[str]]):
    box = draw_bounding_boxes(img, boxes=coordinates,
                          labels=labels,
                          colors="red",
                          width=4, font_size=30)
    return to_pil_image(box.detach())