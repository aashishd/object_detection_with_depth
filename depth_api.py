# %%
from depth.transform import input_transform, build_out_transform
import torch
from depth.depth_detection import batch_depth
from depth.data import ImageWithDepth, InputImage
import argparse
import logging
from pathlib import Path
from typing import List

LOG = logging.getLogger(__name__)


# %%
# common functions
def cliargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=False)
    parser.add_argument("-b", "--batch", type=int, required=False, default=4)
    return parser


def run_depth_est_batch(images: List[InputImage]) -> List[ImageWithDepth]:
    # transfrom all the input images
    tinput = torch.cat([input_transform(img.image()).unsqueeze(0) for img in images], 0)

    # run the model
    outdepths = batch_depth(tinput)

    # reshape output depths
    depthsresh = [
        build_out_transform(img.imgdims())(outdepths[i].unsqueeze(0))
        for i, img in enumerate(images)
    ]

    return [
        ImageWithDepth(img, depthsresh[i].squeeze(0)) for i, img in enumerate(images)
    ]


def run_depth_estimation(input: str, output: str, batch: int):
    # read all the input images
    inpaths = Path(input).glob("*")

    outp = Path(output)
    if not outp.is_dir():
        outp.mkdir()

    inputs = [
        InputImage(str(inpath), str(outp / f"{inpath.stem}-depth.tiff"))
        for inpath in inpaths
    ]
    inbatches = [inputs[i : i + batch] for i in range(0, len(inputs), batch)]
    for batch in inbatches:
        outputs = run_depth_est_batch(inputs)
        [out.savedepth() for out in outputs]


def main():
    args = cliargs().parse_args()
    try:
        run_depth_estimation(args.input, args.output, args.batch)
    except Exception as e:
        LOG.exception("Exception occurred while running depth estimation")


# %%
if __name__ == "__main__":
    # run_depth_estimation(input='resources/test/cats', output='resources/test/outtest', batch=4)
    main()
