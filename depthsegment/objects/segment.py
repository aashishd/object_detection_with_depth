import cv2
import numpy as np


def segment_object(depth_map, bounding_box=None):
    """Segment the object from the depth map

    Args:
        depth_map (_type_): _description_
        bounding_box (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: 
    """
    # Cut the region out from the depth map
    if _has_bounds(bounding_box):
        left, top, right, bottom = bounding_box
        region = depth_map[top:bottom, left:right]
    else:
        region = depth_map
    
    # Normalize the pixel values to the range 0-255
    max_val = np.max(region)
    min_val = np.min(region)
    region = np.uint8((region - min_val) / (max_val - min_val) * 255)
    
    # Automatically calculate the threshold using Otsu's Binarization
    # _, thresholded = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, thresholded = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
        # Post-processing - Morphological operations - erode and dilate to remove small noises
    cleaned = _apply_morph_ops(thresholded, 1)
    
    # repad the image to match the original size
    if _has_bounds(bounding_box):
        cleaned = _repad_mask(cleaned, depth_map.shape, bounding_box)
        thresholded = _repad_mask(thresholded, depth_map.shape, bounding_box)
    
    return thresholded.astype(bool), cleaned.astype(bool)


def _repad_mask(mask, orig_shape, bounding_box, maskfill=0):
    """Repads the mask to match the original image size

    Args:
        mask (_type_): _description_
        bounding_box (_type_): _description_

    Returns:
        _type_: _description_
    """
    height, width = orig_shape
    left, top, right, bottom = bounding_box
    return cv2.copyMakeBorder(mask, top, height-bottom, left, width-right, cv2.BORDER_CONSTANT, value=maskfill)

def _has_bounds(bounding_box):
    return bounding_box is not None and len(bounding_box) == 4


def _apply_morph_ops(input, iterations=1, kernel=3):
    """Erosion followed by dilation

    Args:
        input (_type_): _description_
        iterations (int, optional): _description_. Defaults to 1.
        kernel (int, optional): _description_. Defaults to 3.

    Returns:
        _type_: np.nnarray of the same shape as input
    """
    kernel = np.ones((kernel, kernel), np.uint8)
    cleaned = cv2.erode(input, kernel, iterations=iterations)
    return cv2.dilate(cleaned, kernel, iterations=iterations)
    

