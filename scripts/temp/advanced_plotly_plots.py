import time

import numpy as np
import plotly.express as px
import pandas as pd

# Create an empty DataFrame
df = pd.DataFrame(columns=['x', 'y'])

# Create the figure
fig = px.scatter(df, x='x', y='y')

# Display the initial plot
fig.show()

# Create a function to update the plot
def update_plot():
    # Generate random data point
    x = np.random.rand()
    y = np.random.rand()

    # Append new data point to the DataFrame
    df.loc[len(df)] = [x, y]

    # Update the figure
    fig = px.scatter(df, x='x', y='y')
    fig.update_layout(overwrite=True)
    fig.update_traces(overwrite=True)
    fig.show()

# Call the update_plot() function repeatedly
for i in range(10):
    update_plot()
    time.sleep(1)