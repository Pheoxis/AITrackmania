import pickle
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import config.config_constants as cfg

PATH_REWARD = cfg.REWARD_PATH
with open(PATH_REWARD, 'rb') as f:
    data = pickle.load(f)

plt.axis('equal')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='b', marker='o')
plt.axis('equal')
# Create an interactive 3D scatter plot with Plotly
plotly_fig = go.Figure(data=[go.Scatter3d(
    x=data[:, 0],
    y=data[:, 1],
    z=data[:, 2],
    mode='markers',
    marker=dict(
        size=5,
        color='blue',
        opacity=0.8
    )
)])
plt.axis('equal')
# Customize the layout if needed
plotly_fig.update_layout(
    scene=dict(
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
        zaxis_title='Z Axis',
        aspectratio=dict(x=1, y=0.01, z=1)
    ),
    title='Interactive 3D Scatter Plot'
)
plt.axis('equal')
# Display the interactive plot
plotly_fig.show()

