from pathlib import Path

import torchvision
from PIL import Image


def read_image(filepath: Path):
    """Read image as tensor

    Args:
        filepath (_type_): _description_

    Returns:
        _type_: _description_
    """
    return torchvision.io.read_image(filepath)


def write_tiff(filepath: Path, img):
    """Write Tiff File

    Args:
        filepath (_type_): output path
        img (_type_): tensor

    Returns:
        _type_: _description_
    """
    # return torchvision.io.write_png(img, filepath)
    tifimg = Image.fromarray(img.squeeze().int().numpy())
    return tifimg.save(filepath.with_suffix(".tiff"))

def write_image(filepath: Path, img, format='PNG'):
    """Write the image

    Args:
        filepath (Path): output path
        img (_type_): numpy array
        format (str, optional): _description_. Defaults to 'PNG'.
"""
    pimg = Image.fromarray(img)  
    pimg.save(str(filepath.with_suffix(f".{format.lower()}")), format) 
    
    
def write_grayscale(filepath: Path, img):
    """Write the image

    Args:
        filepath (Path): output path
        img (_type_): numpy array
        format (str, optional): _description_. Defaults to 'PNG'.
"""
    pimg = Image.fromarray(img)  
    pimg.save(str(filepath.with_suffix(f".png")), 'PNG')