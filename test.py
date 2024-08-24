lets think of a smart way to actually implement this optimizer.

lets say i have a (3, 3) tensor of parameters:
[[1, 2, 3], [4, 5, 6]]
 now the optimizer receives the following gradients:

[[0.1, -0.1, 0.2], [0.3, -0.3, 0.4]],
[[-0.5, 0.5, -0.6], [-0.7, 0.7, -0.8]],
[[-0.9, 0.9, 0.0], [-1.1, 1.1, -1.2]]
[[0.0, 0.0, 0.0], [0.2, -0.2, 0.3]]