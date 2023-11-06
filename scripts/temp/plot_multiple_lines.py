import matplotlib.pyplot as plt
import numpy as np

x = np.random.randint(low=0, high=10, size=(2, 3))
y = np.random.randint(low=0, high=10, size=(2, 3))

fig, ax = plt.subplots()
ax.plot(x, y, c="black")
plt.show()