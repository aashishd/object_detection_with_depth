# %%
# import os
# import sys

# sys.path.append(os.path.dirname(os.path.abspath('')))
# !export PYTHONPATH="${PYTHONPATH}:/workspaces/cyclomedia_project"
# %% imports
from torchvision.models.detection import (FasterRCNN_ResNet50_FPN_V2_Weights,
                                          fasterrcnn_resnet50_fpn_v2)
from torchvision.utils import draw_bounding_boxes

import utils
from depth.depth_detection import batch_depth
from depth.domain import Image
from depth.transform import input_transform
from utils.ioutils import read_image

# %% Step 1: Initialize model with the best available weights & Initialize the inference transforms
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
model.eval()
preprocess = weights.transforms()


# %% Step 2: Load Image and Apply inference preprocessing transforms
imgb = preprocess(input_transform(read_image('resources/test/cats/chill-cat.jpg')).unsqueeze(0))
# batch = [preprocess(img)]

# %% Step 3: predict object in image
# prediction = model(batch)[0]
prediction = model(imgb)[0]
labels = [weights.meta["categories"][i] for i in prediction["labels"]]

# %% Step 4: visualize the prediction
box = draw_bounding_boxes(imgb, boxes=prediction["boxes"],
                          labels=labels,
                          colors="red",
                          width=4, font_size=30)
im = to_pil_image(box.detach())
im.show()

# %%
