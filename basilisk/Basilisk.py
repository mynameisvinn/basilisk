"""
BN constructs a bayesian network from Nodes.
"""

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
from graphviz import dot
from collections import deque

class BN(object):
    """
    parameters
    ----------
    ls_nodes : list of nodes

    observations: pandas dataframe
        dataframe, where each column represents a discrete random variable.


    attributes
    ----------
    dict_nodes : dictionary
        key represents node name, value is the corresponding node.

    dict_children : dictionary
        key represents node name, value is a list of its children names.
    """
    
    def __init__(self, ls_nodes):
        self.ls_nodes = ls_nodes  
        self.dict_nodes = self._generate_dict_nodes()  # dict for fast lookup
        self.dict_children = self._generate_dict_children()
        

    def fit(self, observations):
        self.observations = observations
        self._generate_cpt()  # compute cpt for each node - no lazy loading 


    def _generate_cpt(self):
        """iterate through all nodes and compute their respective conditional 
        probability tables.
        """

        for node in self.ls_nodes:
            node.cpt = self._calculate_cpt(node)
        
    def _generate_dict_nodes(self):
        """return a dictionary, where key is node name and value is the 
        corresponding node object."""
        d = {}
        for node in self.ls_nodes:
            d[node.name] = node
        return d
    
    def _generate_dict_children(self):
        """return a dictionary, where key is name of node and value is a
        list of its children."""
        d = {}
        for parent in self.ls_nodes:
            children = []
            
            for child in self.ls_nodes:
                if parent in child.ls_parents:
                    children.append(child.name)
            d[parent.name] = children
        return d
    
    def show(self, **kwargs):
        graph = nx.DiGraph(self.dict_children)
        layout = graphviz_layout(graph, 'dot')
        nx.draw_networkx(graph, layout = layout, **kwargs)
        plt.axis('off')
        plt.show()
        
    def _calculate_cpt(self, node):
        """compute conditional probability table for the specified node.
        """

        # find its parents
        parent = node.ls_parents
        
        # subset its corresponding marginals
        ps = [self.observations[x.name] for x in parent]
        cs = self.observations[node.name]
        
        # finally, crosstab
        # # https://stackoverflow.com/questions/53510319/python-pandas-merging-with-more-than-one-level-overlap-on-a-multi-index-is-not
        cpt = pd.crosstab(ps, cs, normalize = 'index').reset_index()
        
        # if node is a marginal, it wont have parents, so drop unnecessary columns
        if node.is_marginal:
            cpt.drop("index", axis=1, inplace=True)
        return cpt

    def scheduler(self, node):
        """given a node, return its topological graph, which refers to the 
        precise sequence of parent nodes to be executed. this allows proper
        execution of nodes.

        for example, if A->B->C is the causal model, then model.sample(C)
        returns [A, B, C].
        """
        lifo = deque([node])  # final list for topological sort
        fifo = deque([node])  # temporary list, for breadth first search.

        while len(fifo) > 0:
            
            # grab a node from fifo for examination
            curr = fifo.pop()
            # print('evaluating', curr.name)

            # fetch its parents
            ls_parents = curr.parents_nodes

            # evaluate each parent node
            for p in ls_parents:

                # ignore if parent has already been added to topological graph
                if p in lifo:
                    pass
                
                # otherwise, add parent to topological graph
                else:
                    lifo.append(p)
                    fifo.appendleft(p)

            # print("fifo: ", list(map(lambda p: p.name, fifo)))
            # print("lifo", list(map(lambda p: p.name, lifo)))
            # print("-"*40)

        lifo.reverse()  # reverse mutates list in place
        return lifo
    
    def execute(self, node):
        """
        first, execute() fetchs an ordered list of operations from the 
        scheduler. then, execute() samples from each node and records each 
        result. finally, it returns a dictionary of results, where each key 
        represents node name and each value represents its state.
        """
        
        # track results of each sample - to be used for querying cpt
        temp_dict = {}
        
        # ask scheduler for order of operations
        execution_order = self.scheduler(node)
        
        # sample each node
        for curr in execution_order:
            
            # if node is marginal, we dont need parents' states
            if curr.is_marginal:
                res = self._sample(curr)
            
            # otherwise, we query its cpt using its parents' states
            else:
                parent_states = []  # query
                for p in curr.parents_names:
                    parent_states.append(temp_dict[p])
                res = self._sample(curr, parent_states)
                
            # finally, update temp_dict with results
            temp_dict[curr.name] = res
                
        # clean up strings before returning
        for k in temp_dict.keys():
            temp_dict[k] = temp_dict[k].split("==")[1]

        return temp_dict
    
    def _sample(self, node, parent_states=None):
        """a wrapper function around Node's sample() method, which returns a 
        string combining the node's name and its state.

        for example, calling _sample("cloud") returns "cloud==True".
        """        
        res = node.sample(parent_states)[0]
        return node.name + "==" + str(res)