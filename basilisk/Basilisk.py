"""
BN constructs a bayesian network from Nodes.
"""

import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm

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
        import networkx as nx
        from networkx.drawing.nx_agraph import write_dot, graphviz_layout
        from graphviz import dot
        
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
    
    def _execute(self, node):
        """sample from a node and its ancestors.

        first, _execute() fetchs an ordered list of operations from the 
        scheduler. then, _execute() samples from each node and records results. 
        finally, it returns a dict of results, where each k = node name and v = 
        state.

        _execute() is important when order of execution matters.
        """
        
        # track state of each random variable (to be used for querying cpt)
        temp_dict = {}
        
        # ask scheduler for order of operations (nodes sampled in order)
        execution_order = self.scheduler(node)
        
        # sample each node according to order
        for curr in execution_order:
            
            # if node is a marginal, we dont need parents' states
            if curr.is_marginal:
                res = self._sample(curr)
            
            # otherwise, we query its cpt using its parents' states
            else:
                parent_states = []  # query
                for p in curr.parents_names:
                    parent_states.append(temp_dict[p])

                # sample node (conditioned on its parents' states)
                res = self._sample(curr, parent_states)
                
            # finally, update temp_dict with results
            temp_dict[curr.name] = res
                
        # clean up strings before returning
        for k in temp_dict.keys():
            temp_dict[k] = temp_dict[k].split("==")[1]

        return temp_dict
    
    def _sample(self, node, parent_states=None):
        """a wrapper function around Node's sample() method

        _sample() calls a Node's sample() method, and subsequently wraps the
        result (the state of the random variable) with the Node's name. 

        for example, calling _sample("cloud") returns "cloudy==True".
        """        
        res = node.sample(parent_states)[0]
        return node.name + "==" + str(res)
    
    def generate_samples(self, node, n_samples=1):
        """generate a batch of joint observations.
        
        generate_samples() generates samples for the specified node and its 
        corresponding ancestors. 

        for each trial, it calls _execute(), which returns a dict, where k = 
        node name and v = state of random variable. these results are tallied
        in a single dataframe and returned to the user.
        """

        # track sample's states with a dictionary across all trials
        df = {}

        # ask scheduler for list of nodes that will be sampled
        random_vars = list(map(lambda x: x.name, self.scheduler(node)))
        for var in random_vars:
            df[var] = []

        # sample from nodes and accordingly update tracker
        for _ in tqdm(range(n_samples)):
            sample = self._execute(node)
            for (var, state) in sample.items():
                df[var].append(state)
        
        # convert dictionary into a dataframe
        df = pd.DataFrame(df)
        
        # recast each row value from str ("True") to boolean (True)
        for col in df.columns:
            df[col] = df[col] == "True"
        return df