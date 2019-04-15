class Node(object):
    """
    each node object represents a discrete random variable.
    
    each node knows its parent(s) but does not know its children.
    """
    
    def __init__(self, name, ls_parents=[]):
        self.name = name
        self.ls_parents = ls_parents
        
    @property
    def is_marginal(self):
        """a marginal node does not have any parents."""
        return not self.ls_parents