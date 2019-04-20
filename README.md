# basilisk
quickly construct bayesian networks with basilisk. define the graph and provide joint observations, and basilik will handle everything else.

## a working example
lets use [murphy's classic sprinkler example](https://www.cs.ubc.ca/~murphyk/Bayes/bayes_tutorial.pdf).

first, define nodes and their corresponding parents. in bayesian networks, a parent has a *casual* relationship with its children.
```
C = Node("cloudy")
R = Node("rain", [C])  # rain's parent is cloudy
S = Node("sprinkler", [C])
W = Node("wet", [R, S])

ls_nodes = [C, R, S, W]
```
then, instantiate a model and fit with joint observations.
```
model = BN(ls_nodes)

obs = pd.read_csv("observations.csv")  # joint observations
model.fit(obs)  # fit a model, scikit-style

model.draw_graph(node_size=1000)
```
we can inspect conditional probability tables.
```
W.cpt  # basilik automatically computes conditional probabilities
```
we can sample from any node in the graph.
```
model.execute(W)

# {'cloudy': 'True', 'sprinkler': 'False', 'rain': 'True', 'wet': 'False'}
```