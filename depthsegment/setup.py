import os
from typing import List

from setuptools import find_packages, setup


def parse_requirements(file: str) -> List[str]:
    required_packages = []
    with open(os.path.join(os.path.dirname(__file__), file)) as req_file:
        for line in req_file:
            if "/" not in line:
                required_packages.append(line.strip())
    return required_packages
setup(
    name="depthsegment",
    python_requires=">=3.8",
    version=0.1,
    description="Depth Segmentation Project",
    packages=find_packages(),
    include_package_data=True,
    # install_requires=parse_requirements("requirements.txt"),
    zip_safe=False
)