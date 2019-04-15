# basilik
basilik makes it easy to construct bayesian networks. 

all you need to is define the graph (ie specifying edge relationships) and provide joint observations. basilik will handle the rest (eg constructing a bayesian network, computing conditional probabilities).

## example
we'll use [murphy's classic sprinkler example](https://www.cs.ubc.ca/~murphyk/Bayes/bayes_tutorial.pdf).

first, define nodes and corresponding parents.
```
C = Node("cloudy", None)
R = Node("rain", [C])
S = Node("sprinkler", [C])
W = Node("wet", [R, S])

ls_n = [C, R, S, W]  # a list of nodes
```

then, instantiate basilik with joint observations.
```
joint_observations = pd.read_csv("observations.csv").drop("Unnamed: 0", axis=1)
bayes_network = BN(ls_n, joint_observations)
```

we can easily inspect conditional probability tables for a given node.
```
bayes_network.generate_cpt("wet")  # returns dataframe
```