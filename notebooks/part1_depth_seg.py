# %%
import numpy as np
import requests
import torch
from PIL import Image
from torchvision import io
from transformers import GLPNForDepthEstimation, GLPNImageProcessor

# %%
# load the models and preprocessors
processor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")
# %%
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
img2 = Image.open('resources/test/cats/chill-cat.jpg')
# %%
# prepare image for the model
inputs = processor(images=[image, img2], return_tensors="pt")

# %%

with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth
    
# %%

# interpolate to original size
prediction = torch.nn.functional.interpolate(
    predicted_depth.unsqueeze(1),
    size=image.size[::-1],
    mode="bicubic",
    align_corners=False,
)

# visualize the prediction
output = prediction.squeeze().cpu().numpy()
formatted = (output * 255 / np.max(output)).astype("uint8")
depths = Image.fromarray(formatted)

# %%

