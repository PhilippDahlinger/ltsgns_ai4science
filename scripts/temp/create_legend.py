import matplotlib.pyplot as plt

# create a legend with red dots as prediction and blue dots as ground truth
fig, ax = plt.subplots()
ax.scatter([1, 2, 3], [1, 2, 3], label='prediction', color='red', marker='o')
ax.scatter([3, 2, 1], [1, 2, 3], label='ground truth', color='blue', marker='o')
# scale legend to fontsize 24
ax.legend(fontsize=24)
plt.show()