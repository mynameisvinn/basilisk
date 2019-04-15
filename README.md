# basilik for bayesian networks
define the graph and provide joint observations, and basilik will handle everything else.

## example
lets use [murphy's classic sprinkler example](https://www.cs.ubc.ca/~murphyk/Bayes/bayes_tutorial.pdf).

first, define nodes and corresponding parents.
```
C = Node("cloudy", None)
R = Node("rain", [C])
S = Node("sprinkler", [C])
W = Node("wet", [R, S])

ls_nodes = [C, R, S, W]
```

then, instantiate basilik with joint observations.
```
obs = pd.read_csv("observations.csv").drop("Unnamed: 0", axis=1)
model = BN(ls_nodes, obs)
```

we can easily inspect conditional probability tables for a given node.
```
model.generate_cpt("wet")  # basilik automatically generates CPTs from observations
```