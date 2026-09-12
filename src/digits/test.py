import numpy as np

from network import Network

network = Network([2, 3, 2], seed=42)

x = np.array([[0.4],[0.9],
])

output = network.forward(x)

print(output)
print(output.shape)