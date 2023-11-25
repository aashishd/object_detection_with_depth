# %% imports
import logging
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from objects.object_detection import batch_object_detect
from objects.segment import segment_object
from utils.ioutils import write_image, write_tiff

from depthsegment.depth.depth_detection import batch_depth
from depthsegment.depth.depthapi import run_depth_est_batch
from depthsegment.depth.domain import (ImageWithDepth,
                                        ImageWithDepthAndObjects, InputImage)
from depthsegment.depth.transform import build_out_transform, input_transform

LOG = logging.getLogger(__name__)

# %% common functions
def run_depth_est_batch(images: List[InputImage]) -> List[ImageWithDepth]:
    # transfrom all the input images
    tinput = torch.cat([input_transform(img.image()).unsqueeze(0) for img in images], 0)

    # run the depth estimation model
    outdepths = batch_depth(tinput)

    # convert to ImageWithDetph
    return [ImageWithDepth(img, outdepths[i].squeeze(0)) for i, img in enumerate(images)]
    
    


def run_object_detect_batch(depthimgs: List[ImageWithDepth]) -> List[ImageWithDepthAndObjects]:
    tinput = torch.cat([input_transform(img.inimg.image()).unsqueeze(0) for img in depthimgs], 0)
    
    # run the object detection model
    outobjects = batch_object_detect(tinput)
    
    # add depth values to the objects
    depobjimgs = [depimg.withobjects(objdets) for depimg, objdets in zip(depthimgs, outobjects)]
    
    # update the object masks
    for doimg in depobjimgs:
        objswithmasks = []
        for obj in doimg.objects:
            # pass a numpy image + corrdinates => to detect mask of detected object
            rawmask, cleanmask = segment_object(doimg.depthimg.numpy(), obj.coordinates)
            objswithmasks.append(obj.withmask(cleanmask))
        # update the objects with masks
        doimg.updateobjects(objswithmasks)
        
    return depobjimgs
 
    
    
def prepare_input_batches(input: str, output: str, batch: int):
     # prepare batches
    inpaths = Path(input).glob("*")
    outp = Path(output)

    inputs = [
        InputImage(inpath, outp / f"{inpath.stem}")
        for inpath in inpaths
    ]
    # replace the following with yield
    for i in range(0, len(inputs), batch):
        yield inputs[i : i + batch]


def save_depth(imgd: ImageWithDepth):
    # build outpath
    outpath = imgd.inimg.outpath
    outpath.parent.mkdir(parents=True, exist_ok=True)
    
    # transfrom img & write
    transfomed_img = build_out_transform(imgd.inimg.imgdims())(imgd.depthimg.unsqueeze(0))
    outpath = outpath.parent / f'{outpath.stem}-depth'
    write_tiff(outpath, transfomed_img)

    # trasform by converting output to numpy uint8 array by interpolating between 0 and 255
    # write
    outpath = outpath.parent / f'{outpath.stem}-viz-depth'
    transfomed_viz = ((transfomed_img - transfomed_img.min()) / (transfomed_img.max() - transfomed_img.min())) * 255.
    write_image(outpath, transfomed_viz.squeeze().numpy().astype(np.uint8), format='PNG')
        
def save_object_segmasks(imgd: ImageWithDepthAndObjects):
    for i, obj in enumerate(imgd.objects):
        # resize object mask to the original image size
        objmask = cv2.resize(obj.mask, imgd.inimg.imgdims()[::-1], interpolation=cv2.INTER_NEAREST)
        outimg = imgd.inimg._imgt.permute(-2, -1, 0).numpy() * objmask.astype(bool)[..., None]
        
        # build outpath
        outpath = imgd.inimg.outpath
        outpath.parent.mkdir(parents=True, exist_ok=True)
        outpath = outpath.parent / f'{outpath.stem}-{i}-{obj.objclass}'
        
        # write img
        write_image(outpath, outimg)
        

def run_pipeline(input: str, output: str, batch: int, detect_objects=True):
    for inbatch in prepare_input_batches(input, output, batch):
        imgwithdepth = run_depth_est_batch(inbatch)
        for imgd in imgwithdepth:
            save_depth(imgd)
        # detect objects if true
        if detect_objects:
            outwithobjects = run_object_detect_batch(imgwithdepth)
            for imgdo in outwithobjects:
                save_object_segmasks(imgdo)

# %%
# run_depth_est_batch('')
# run_pipeline(input='resources/test/cats', output='resources/test/outtest', batch=4, detect_objects=False)
# %%
