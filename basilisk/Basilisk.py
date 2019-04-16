"""
BN module constructs a bayesian network from Node objects.
"""

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
from graphviz import dot

class BN(object):
    """Bayesian Network

    parameters
    ----------
    ls_nodes : list of Node objects

    observations: pandas dataframe
        dataframe, where each column represents a discrete random variable.

    attributes
    ----------
    dict_nodes : dictionary
        key represents name of Node, value is a list of its parents

    dict_adj : dictionary
        key represents name of Node, value is a list of its children
    """
    
    def __init__(self, ls_nodes, observations):
        self.ls_nodes = ls_nodes
        self.observations = observations
        self.dict_nodes = self._generate_dict_nodes()  # create dict of nodes for fast lookup
        self.dict_children = self._generate_dict_children()
        
    def _generate_dict_nodes(self):
        """return a dictionary, where key is name of node and value is 
        the corresponding node object."""
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
    
    def draw_graph(self, **kwargs):
        graph = nx.DiGraph(self.dict_children)
        layout = graphviz_layout(graph, 'dot')
        nx.draw_networkx(graph, layout = layout, **kwargs)
        plt.axis('off')
        plt.show()
        
    def generate_cpt(self, name):
        # first, fetch node object
        node = self.dict_nodes[name]
        
        # then find its parents
        parent = node.ls_parents
        
        # subset its corresponding marginals
        ps = [self.observations[x.name] for x in parent]
        cs = self.observations[node.name]
        
        # finally, crosstab
        # # https://stackoverflow.com/questions/53510319/python-pandas-merging-with-more-than-one-level-overlap-on-a-multi-index-is-not
        return pd.crosstab(ps, cs, normalize = 'index').reset_index()