import unittest
from basilisk import Node, BN
import numpy as np
import pandas as pd

class Test_Basilisk(unittest.TestCase):

    def setUp(self):
        """construct a simple bayesian network.
        """
        B = Node("B")
        A = Node("A", [B])
        C = Node("C", [A])
        T = Node("T")
        self.R = Node("R", [C, T])
        S = Node("S", [C])
        self.W = Node("W", [self.R, S])  # leaf node
        ls_nodes = [B, A, C, T, self.R, self.W, S]
        self.model = BN(ls_nodes)

    def test_scheduler(self):
        """a topological sort returns a list of nodes, which represents the 
        order of execution.
        """

        correct_sequence = ['B', 'A', 'T', 'C', 'S', 'R', 'W']
        
        # returns a list of nodes
        computed_sequence = self.model.scheduler(self.W)  
        
        # convert to list of node names
        computed_sequence = list(map(lambda x: x.name, computed_sequence))  

        self.assertEqual(correct_sequence, computed_sequence)

    def test_sample(self):
        """assert distribution from directly sampling nodes matches distribution
        from joint observations.
        """

        # construct a new graph
        C = Node("cloudy")
        R = Node("rain", [C])
        S = Node("sprinkler", [C])
        W = Node("wet", [R, S])
        ls_n = [C, R, S, W]
        model = BN(ls_n)
        obs = pd.read_csv("data/observations.csv").drop("Unnamed: 0", axis=1)
        model.fit(obs)  

        leaf_node = R  # arbitrarily selected node
        samples = model.generate_samples(leaf_node, n_samples=1000)

        # construct conditional probablity table from joint observations
        joint_obs = pd.crosstab(samples["cloudy"], samples["rain"], normalize = 'index').reset_index()
        a = np.array(joint_obs[False])
        b = np.array(leaf_node.cpt[False])
        self.assertTrue(np.allclose(a, b, atol=.05))  # atol = absolute difference

if __name__ == "__main__":
    unittest.main()