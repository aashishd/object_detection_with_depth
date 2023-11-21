import numpy as np
import torchvision
from PIL import Image


class InputImage:
    def __init__(self, imgpath: str, outpath: str) -> None:
        self.imgpath = imgpath
        self.outpath = outpath
        self._imgt = None
        self.image()

    def image(self):
        if self._imgt is None:
            self._imgt = read_image(self.imgpath)
            if len(self._imgt.size()) == 2:
                self._imgt = self._imgt.unsqueeze(0)
        return self._imgt

    def imgdims(self):
        imshp = self.image().size()
        return imshp[-2], imshp[-1]

    def withdepth(self, depth):
        return ImageWithDepth(self, depth)


class ImageWithDepth:
    def __init__(self, inimg: InputImage, depthimg: np.ndarray) -> None:
        self.inimg = inimg
        self.depthimg = depthimg

    def savedepth(self):
        write_tiff(self.inimg.outpath, self.depthimg)


def read_image(filepath):
    return torchvision.io.read_image(filepath)


def write_tiff(filepath, img):
    # return torchvision.io.write_png(img, filepath)
    tifimg = Image.fromarray(img.squeeze().int().numpy())
    return tifimg.save(str(filepath))
