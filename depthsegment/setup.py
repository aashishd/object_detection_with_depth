from setuptools import find_packages, setup

setup(
    name="depthsegment",
    python_requires=">=3.8",
    version=0.1,
    description="Depth Segmentation Project",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False
)