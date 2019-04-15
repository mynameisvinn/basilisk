# basilisk
define the graph and provide joint observations, and basilik will handle everything else.

## example
lets use [murphy's classic sprinkler example](https://www.cs.ubc.ca/~murphyk/Bayes/bayes_tutorial.pdf).

first, define nodes and corresponding parents.
```
C = Node("cloudy")
R = Node("rain", [C])  # rain's parent is cloudy
S = Node("sprinkler", [C])
W = Node("wet", [R, S])

ls_nodes = [C, R, S, W]
```

then, instantiate a model with joint observations.
```
obs = pd.read_csv("observations.csv").drop("Unnamed: 0", axis=1)
model = BN(ls_nodes, obs)
```

we can inspect conditional probability tables.
```
model.generate_cpt("wet")  # basilik automatically generates CPTs from joint observations
```

## to-do
* sample api: something like model.sample("rain") will return joint observations for rain and its parents. first, each node will need to know its parents, direct and indirect.