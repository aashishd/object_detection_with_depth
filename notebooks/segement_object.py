# %% : imports
import numpy as np
from PIL import Image

from depth.depth_detection import batch_depth
from depth.domain import Image
from depth.transform import input_transform
from objects.segment import segment_object
from utils.ioutils import read_image

# %%
img = input_transform(read_image('resources/test/cats/chill-cat.jpg')).unsqueeze(0)
# %%
depthmask = batch_depth(img)
# %%
out, outc = segment_object(depthmask.squeeze().numpy())

# %%
