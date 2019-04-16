import itertools
import copy
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
from graphviz import dot

def graph(dict_children, **kwargs):
    graph = nx.DiGraph(dict_children)
    layout = graphviz_layout(graph, 'dot')
    nx.draw_networkx(graph, layout = layout, **kwargs)
    plt.axis('off')
    plt.show()

def calc_mi(depdata, depbins):
    """
    inputs:
        depdata - data for two 'dependent' variables as 2-column numpy array
        depbins - # of bins for each variable - determines resolution for pdf
    outputs:
        mutual information between two 'dependent' variables
    """
    eps = 1e-10

    hist, edges = np.histogramdd(depdata, bins=depbins)

    Pxy = hist / depdata.shape[0] # P(X,Y)
    Px = np.sum(Pxy, axis = 1) # P(X)
    Py = np.sum(Pxy, axis = 0) # P(Y)    
    PxPy = np.outer(Px,Py)

    mi = np.sum(Pxy * np.log((Pxy + eps) / (PxPy + eps)))
            
    return mi

def calc_cmi(depdata, depbins, inddata, indbins):
    """
    inputs:
        depdata - data for two dependent variables as 2-column numpy array
        depbins - # of bins for each variable - determines resolution for pdf
        
        inddata - data for one independent variable as 1-column numpy array
        indbins - # of bins for the independent variable - determines pdf resolution
    outputs:
        conditional mutual information between two dependent variables conditional
        on a third independent variable
    """
    eps = 1e-10
    bins = np.concatenate((depbins, indbins))
    data = np.concatenate((depdata, inddata), axis = 1)

    hist, edges = np.histogramdd(data, bins=bins)

    Pxyz = hist / data.shape[0] #P(X,Y,Z)
    Pz = np.sum(Pxyz, axis = (0,1)) # P(Z)
    Pxz = np.sum(Pxyz, axis = 1) # P(X,Z)
    Pyz = np.sum(Pxyz, axis = 0) # P(Y,Z)    

    lognum = np.empty((Pxyz.shape)) # P(Z)P(X,Y,Z)
    logden = np.empty((Pxyz.shape)) # P(X,Z)P(Y,Z)
    for i in range(bins[0]):
        for j in range(bins[1]):
            for k in range(bins[2]):
                lognum[i][j][k] = Pz[k]*Pxyz[i][j][k]
                logden[i][j][k] = Pxz[i][k]*Pyz[j][k]

    cmi = np.sum(Pxyz * np.log((lognum + eps) / (logden + eps) ) )
    
    return cmi

def dsep(depvars, indvars, test = 'mi', alpha = 0.05):
    if test == 'mi':
        """
        d-separation test using mutual information
        inputs:
            depvars - data for two dependent variables as 2-column dataframe
            indvars - data for N independent variables as N-column dataframe
            test - choice of d-separation test - defaults to mutual information test
            alpha - pvalue threshold for dependence test
            
        output:
            if p-value(test statistic) > alpha, return true, else return false
            
        issues:
            can only handle integer valued data
        """
        depbins = depvars.apply(pd.Series.nunique).values
        indbins = indvars.apply(pd.Series.nunique).values
        depdata = depvars.values.astype(int)
        inddata = indvars.values.astype(int)

        assert len(depbins) == 2
        
        if len(indvars.columns) == 0: #if no independent variables, compute marginal MI
            mi = calc_mi(depdata, depbins)

            chi2 = 2*len(depvars)*mi
            df = (depbins[0] - 1) * (depbins[1] - 1)
            p_val = 2*stats.chi2.pdf(chi2, df)
            
            if p_val > alpha:
                return True
            else:
                return False
            
        else: #if more than 0 independent variables, compute conditional MI
            if len(indvars.columns) > 1: 
                #if more than 1 independent variable, concatenate them to a product codeword
                inddata = inddata.astype(str)

                for i in range(len(inddata)):
                    inddata[i, 0] = ''.join(inddata[i,:len(indbins)])
                    
                inddata = inddata.astype(int)[:,0].reshape((len(depvars), 1))
                indbins = np.array([ len(np.unique(inddata) ) ])
                
            cmi = calc_cmi(depdata, depbins, inddata, indbins)
            
            chi2 = 2*len(depvars)*cmi
            df = (depbins[0] - 1) * (depbins[1] - 1) * indbins[0]
            p_val = 2*stats.chi2.pdf(chi2, df)
            
            if p_val > alpha:
                return True
            else:
                return False

def pc_basic(data):
    """
    path condition algorithm (spirtes 2nd ed 5.4.2)
    
    pseudo-code:
    
    1) initialize complete undirected "graph"
    2) perform d-sep test 
    
        n = 0
        repeat:
            for each x in graph:
                for each y in adj(x) s.t. |adj(x) \ y| >= n:
                    for each subset z in adj(x) \ y s.t. |z| = n:
                        if x and y are d-separated given z:
                            remove edge x-y from graph
                            record z in sepset(x,y) and sepset(y,x)
            n = n + 1
        until: |adj(x) \ y| < n for every pair x,y
        
    3) find collider nodes
    
        for each triplet x, y, z in graph:
            if x,y and y,z are adjacent but x,z is not adjacent:
                iff y is not in sepset(x,z):
                    orient x - y - z as x -> y <- z
    
    4) orient remaining edges
    
        repeat:
            if x -> y:
                for all z s.t. z is adjacent to y and z is not adjacent to x:
                    if there is no arrowhead at y:
                        orient y - z as y -> z
            if a directed path from x to y exists:
                if an edge exists between x and y:
                    orient x - y as x -> y
        until: no more edges to orient
    """
    
    labels = data.apply(pd.Series.nunique).index.values
    graph = dict([(x, [y for y in labels if x!=y]) for x in labels])
    sepset = dict([(x, []) for x in labels])
    
    n = 0
    stop = False
    while not stop:
        for x in labels: #loop through vertices x
            if len(graph[x]) - 1 >= n: #if |adj(x)| - 1 >= n
                for y in graph[x]: #loop through adj(x)
                    adj = copy.deepcopy(graph[x])
                    adj.remove(y) #define adj(x) \ y
                    for z in itertools.combinations(adj, n): #loop through adj(x) \ y
                        if dsep(data[[x, y]], data[list(z)]):
                            sepset[x] = {y:z} #add z to sepset(x,y)
                            sepset[y] = {x:z} #add z to sepset(y,x)
                            
                            if y in graph[x]: #remove x-y edge
                                graph[x].remove(y)
                                graph[y].remove(x)
                    del adj #probably unecessary
        n += 1
        
        for x in labels:
            if len(graph[x]) > n - 1:
                stop = False
                break
            else:
                stop = True
    
    dgraph = dict([(x,[]) for x in graph.keys()]) #initialize empty directed graph

    for x in graph.keys(): #orient edges
        for z in graph[x]:
            for y in graph[z]:
                if y!=x and x not in graph[y] and y not in graph[x]:
                    if y in sepset[x]:
                        if z not in sepset[x][y]:
                            if z not in dgraph[x]:
                                dgraph[x].append(z)
                            if z not in dgraph[y]:
                                dgraph[y].append(z)
                        else:
                            if x not in dgraph[z]:
                                dgraph[z].append(x)
                            if y not in dgraph[z]:
                                dgraph[z].append(y)
    return dgraph
