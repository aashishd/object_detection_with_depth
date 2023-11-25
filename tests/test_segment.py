import cv2
import numpy as np
import pytest
from PIL import Image

from depthsegment.objects.segment import segment_object


def test_segment_object():
    # Create a sample depth map
    # given a depth map of size 100 X 100 of a donut shaped object
    depth_map = np.random.randint(4, 10, (100, 100)).astype(float)
    # create a donut shaped object
    depth_map[10:90, 20:80] = 2.0
    
    # Call the function
    result = segment_object(depth_map)
    
    # Assert the expected output
    expected_result = np.zeros((100, 100)).astype(bool)
    expected_result[10:90, 20:80] = True
    assert result[0].all() == expected_result.all()
    assert result[1].all() == expected_result.all()
    
    
def test_round_circle():
    # Create a sample depth map
    depth_map = np.random.randint(4, 10, (100, 100)).astype(float)
    # create circle using cv2.circle
    depth_map = cv2.circle(depth_map, (50, 50), 20, 2.0, -1)
    
    # Call the function
    result = segment_object(depth_map)
    
    # Assert the expected output
    # print(result)
    assert result[0][40:60, 40:60].all() == True
    assert result[1][40:60, 40:60].all() == True
    