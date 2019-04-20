import unittest
import os
from os.path import dirname
from basilisk import Node, BN
import numpy as np
import pandas as pd

from basilisk.structure import calc_mi

class test_structure(unittest.TestCase):
    def setUp(self):
        self.dir = os.path.join(dirname( dirname(dirname(__file__) ) ), 'data')
        self.data = pd.read_csv(os.path.join(self.dir, 'observations.csv'), index_col = 0)
        
    def test_mi(self):
        testdata = self.data[['cloudy', 'rain']].values.astype(int)
        testbins = [2, 2]
        
        self.assertEqual(np.around(calc_mi(testdata, testbins ), 3), 0.1800)