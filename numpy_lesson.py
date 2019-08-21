import unittest
import numpy as np

class TestNumpy(unittest.TestCase):

    def test_random(self):
        print("hello")
        arr1 = np.random.rand(2,2,1)
        print(arr1)
        arr2 = np.float32(arr1)
        print(arr2)
