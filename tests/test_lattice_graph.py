import unittest
from models.spin_chain import LatticeGraph
import numpy as np


class LatticeGraphTestCase(unittest.TestCase):
    @staticmethod
    def DM_z_period4(t, i):
        phase = np.pi / 2 * (i % 4)
        if t == "+DM":
            return phase
        elif t == "-DM":
            return -phase
        else:
            return 0
        
    @staticmethod
    def XY_z_period4(t, i):
        phase = np.pi - 3. * np.pi / 2 * (i % 4)
        if t == "+XY":
            return phase
        elif t == "-XY":
            return -phase
        else:
            return 0

    @staticmethod
    def native(t, i, j):
        if t in ["+DM", "-DM", "+XY", "-XY"]:
            return 0
        else:
            return 0.5
        
    def test_from_interactions(self):
        minus_DM_dict = {'xx': [[0, 0, 1], [0, 1, 2], [0, 2, 3]],
                         'yy': [[0, 0, 1], [0, 1, 2], [0, 2, 3]],
                         'z': [[-0.0, 0], [-1.5707963267948966, 1],
                               [-3.141592653589793, 2], [-4.71238898038469, 3],
                               [0, 0], [0, 1], [0, 2], [0, 3]]}
        minus_DM_pbc_dict = {'xx': [[0, 0, 1], [0, 1, 2], [0, 2, 3], [0, 3, 0]],
                             'yy': [[0, 0, 1], [0, 1, 2], [0, 2, 3], [0, 3, 0]],
                             'z': [[-0.0, 0], [-1.5707963267948966, 1],
                                   [-3.141592653589793, 2],
                                   [-4.71238898038469, 3], [0, 0], [0, 1],
                                   [0, 2], [0, 3]]}

        
        terms = [['XX', self.native, 'nn'], ['yy', self.native, 'nn'],
                 ['z', self.DM_z_period4, np.inf], ['z', self.XY_z_period4,
                                                    np.inf]]
        graph = LatticeGraph.from_interactions(4, terms, pbc=False)
        graph_pbc = LatticeGraph.from_interactions(4, terms, pbc=True)

        self.assertEqual(minus_DM_dict, graph("-DM"))
        self.assertEqual(minus_DM_pbc_dict, graph_pbc("-DM"))

if __name__ == '__main__':
    unittest.main()
