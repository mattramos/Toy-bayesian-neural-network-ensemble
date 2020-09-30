# Toy-bayesian-neural-network-ensemble

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mattramos/Toy-bayesian-neural-network-ensemble/master?filepath=toy_dataset_example.ipynb)

A basic walkthrough of a runnable example (in binder - link above) of how Bayesian neural networks can be used to ensemble geophysical models.

### Why use a Bayesian neural network ensemble to ensemble physical models
- Ensembles of physical models are often not indepdent (some share code) nor do they perform equally. Some are better than others and there is spatio-temporal variation in their performance. A BayNNE will learn how to combine these models spatio temporally, to produce the most accurate predictions.
- Many methods to ensemble physical models do not consider the uncertainty in observations, or the uncertainty in the models. A BayNNE learns observational uncertainty and is constructed such that predictions are less certain in areas of sparse data.
- Aswell as generating better predictions, the BayNNE can look back in time and be used to infill gaps in observational records. In this sense we can fuse togetehr the best physical models and best observations to gain a more complete and continuous understanding of past states.
- More info in the notebook on binder!
