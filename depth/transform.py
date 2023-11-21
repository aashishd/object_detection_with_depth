from torchvision.transforms import Compose, InterpolationMode, Resize

# TODO : check for the input shape
# TODO : maybe maintain aspect ratio as well
input_transform = Compose(
    [Resize(size=(640, 480), interpolation=InterpolationMode.BILINEAR)]
)


build_out_transform = lambda outsize: Compose(
    [Resize(size=outsize, interpolation=InterpolationMode.BILINEAR)]
)
