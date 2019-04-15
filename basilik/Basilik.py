import pandas as pd

class BN(object):
    
    def __init__(self, ls_nodes, observations):
        self.ls_nodes = ls_nodes
        self.observations = observations
        self.dict_nodes = self._generate_dict_nodes(ls_nodes)  # create dict of nodes for fast lookup
        
    def _generate_dict_nodes(self, ls_nodes):
        d = {}
        for node in ls_nodes:
            d[node.name] = node
        return d
    
    def generate_cpt(self, name):
    	# TODO: handle CPT for marginal probabilities
        
        # first, fetch node
        node = self.dict_nodes[name]
        
        # then find its parents
        parent = node.ls_parents
        
        # subset its corresponding marginals
        ps = [self.observations[x.name] for x in parent]
        cs = self.observations[node.name]
        
        # finally, crosstab
        # # https://stackoverflow.com/questions/53510319/python-pandas-merging-with-more-than-one-level-overlap-on-a-multi-index-is-not
        return pd.crosstab(ps, cs, normalize = 'index').reset_index()