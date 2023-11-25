from pathlib import Path
from typing import List

from depthsegment.utils.ioutils import read_image


class DetectedObject:
    def __init__(self, objclass: str, coordinates: List[int]):
        self.objclass = objclass
        self.coordinates = coordinates
        
    def withmask(self, mask):
        return DetectedObjectWithMask(self.objclass, self.coordinates, mask)
        
        
class DetectedObjectWithMask(DetectedObject):
    def __init__(self, objclass: str, coordinates: List[int], mask):
        super().__init__(objclass, coordinates)
        self.mask = mask # numpy array mask


class InputImage:
    def __init__(self, imgpath: Path, outpath: Path) -> None:
        self.imgpath = imgpath
        self.outpath = outpath
        self._imgt = None
        self.image()

    def image(self):
        if self._imgt is None:
            self._imgt = read_image(str(self.imgpath))
            if len(self._imgt.size()) == 2:
                self._imgt = self._imgt.unsqueeze(0)
        return self._imgt

    def imgdims(self):
        imshp = self.image().size()
        return imshp[-2], imshp[-1]

    def withdepth(self, depth):
        return ImageWithDepth(self, depth)


class ImageWithDepth:
    def __init__(self, inimg: InputImage, depthimg) -> None:
        self.inimg = inimg
        self.depthimg = depthimg
        
    def withobjects(self, objects: List[DetectedObject]):
        return ImageWithDepthAndObjects(self.inimg, self.depthimg, objects)
        
        
class ImageWithDepthAndObjects(ImageWithDepth):
    def __init__(self, inimg: InputImage, depthimg, objects: List[DetectedObject]):
        super().__init__(inimg, depthimg)
        self.objects = objects
        
    def updateobjects(self, objects: List[DetectedObject]):
        self.objects = objects
        return self


