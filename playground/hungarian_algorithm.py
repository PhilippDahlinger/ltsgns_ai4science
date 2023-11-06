import numpy as np
from matplotlib import pyplot as plt

from playground.pc2mesh import get_assignment

pc1 = np.random.randn(100, 2)
# permuate rows
pc2 = pc1.copy()
np.random.shuffle(pc2)

fixed_pc2 = get_assignment(pc1, pc2)

plt.plot(fixed_pc2[:, 0] - pc1[:, 0], fixed_pc2[:, 1] - pc1[:, 1], "o")
plt.plot(pc2[:, 0] - pc1[:, 0], pc2[:, 1] - pc1[:, 1], "x")
plt.ylim([-2, 2])
plt.xlim([-2, 2])
plt.show()