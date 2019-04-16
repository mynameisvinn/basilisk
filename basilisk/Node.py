class Node(object):
    """Nodes represents discrete random variables and are used to construct 
    bayesian networks.
    
    parameters
    ----------
    ls_parents : list of parent nodes


    attributes
    ----------
    name : string

    cpt : pandas dataframe
        represents conditional probability table, as computed from joint 
        observations.
        
    status : string
        can be white, gray, black. used for breadth first search.
    """
    
    def __init__(self, name, ls_parents=[]):
        self.name = name
        self.ls_parents = ls_parents
        self.cpt = None  # generated after we fit BN model
        self.status = "white"
        
    @property
    def is_marginal(self):
        """a marginal node does not have any parents.
        """
        return not self.ls_parents
    
    def get_parents(self):
        return self.ls_parents