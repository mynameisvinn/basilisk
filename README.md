# basilisk
quickly construct bayesian networks with basilisk. define the graph and provide joint observations, and basilik will handle everything else.

## an example
lets use [murphy's classic sprinkler example](https://www.cs.ubc.ca/~murphyk/Bayes/bayes_tutorial.pdf).

first, define nodes and their corresponding parents. in bayesian networks, a parent has a *casual* relationship with its children.
```
C = Node("cloudy")
R = Node("rain", [C])  # rain's parent is cloudy
S = Node("sprinkler", [C])
W = Node("wet", [R, S])

ls_nodes = [C, R, S, W]
```

then, instantiate a model.
```
obs = pd.read_csv("observations.csv")  # joint observations

model = BN(ls_nodes)
model.fit(obs)  # fit a model, just like in scikit
model.draw_graph(node_size=1000)
```

we can inspect conditional probability tables.
```
W.cpt  # basilik automatically computes conditional probabilities
```

## TODO
* create sample method: something like model.sample("rain") will return joint observations for rain and its parents. first, each node will need to know its parents, direct and indirect (do this through BFS).
* assert observations correspond to nodes in graph