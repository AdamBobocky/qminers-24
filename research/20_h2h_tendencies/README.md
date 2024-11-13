# H2H tendencies

This folder serves to answer the question "Are some teams inherently better when playing against other teams?"

The first method tested was: h2hmatrix.py where we keep log of every time a team won or lost - odds expectation,
and if this cumulative over / underperformance is above a threshold we make a bet. This is not profitable.

The second method was embeddings.py where I made a simple neural network with team embeddings that fit some value
modifying odds expectations, and the mse improvements were minuscle and soon overfit. Seems that neither method
has worked and h2h is a dead-end.
