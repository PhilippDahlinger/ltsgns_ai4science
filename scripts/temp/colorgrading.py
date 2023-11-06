import matplotlib.pyplot as plt

# Create a figure and axis
fig, ax = plt.subplots()

# Draw two rectangles with different colors and alpha values
# Overlay 'blue' and 'yellow' with alpha=0.3 to get green
ax.add_patch(plt.Rectangle((0.2, 0.3), 0.6, 0.6, color='turquoise', alpha=0.4))
ax.add_patch(plt.Rectangle((0.3, 0.2), 0.6, 0.6, color='orange', alpha=0.4))

# Set the axis limits
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# Show the plot
plt.show()