# %%
import argparse
import logging

from depthsegment.depth_segment_api import run_pipeline

LOG = logging.getLogger(__name__)


# %%
# common functions
def cliargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("-b", "--batch", type=int, required=False, default=4)
    #  add a boolean flag to indicate whether to run object detection or not
    parser.add_argument("--obj", action="store_true", required=False, default=False)
    return parser


def main():
    args = cliargs().parse_args()
    try:
        run_pipeline(args.input, args.output, args.batch, args.obj)
    except Exception as e:
        LOG.exception("Exception occurred while running depth estimation")


# %%
if __name__ == "__main__":
    main()
