import unittest
from basilisk import Node, BN

class Test_Basilisk(unittest.TestCase):

    def setUp(self):
        """construct a simple bayesian network.
        """
        B = Node("B")
        A = Node("A", [B])
        C = Node("C", [A])
        T = Node("T")
        R = Node("R", [C, T])
        S = Node("S", [C])
        self.W = Node("W", [R, S])  # leaf node
        ls_nodes = [B, A, C, T, R, self.W, S]
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

if __name__ == "__main__":
    unittest.main()