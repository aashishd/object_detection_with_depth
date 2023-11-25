import torch
from transformers import GLPNForDepthEstimation, GLPNImageProcessor

img_processor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")


def batch_depth(images, interpolate=False):
    # prepare image for the model
    inputs = img_processor(images=images, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # interpolate to original size
    if interpolate is True:
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=images.size[::-1],
            mode="bicubic",
            align_corners=False,
        )
    else:
        prediction = predicted_depth

    return prediction  # .squeeze().cpu().numpy()
