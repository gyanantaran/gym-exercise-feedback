import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Create a figure with a custom grid layout
Y = 3
X = 5
S = 3
N = 5
fig = plt.figure(figsize=(X * S, Y * S))
grid = fig.add_gridspec(N, N)

# Create subplots within the custom grid
ax0 = fig.add_subplot(grid[0, 0])
ax1 = fig.add_subplot(grid[1:N, 0])
ax2 = fig.add_subplot(grid[0, 1:N])
ax3 = fig.add_subplot(grid[1:N, 1:N])

# Set the limits and labels for the subplots
ax1.set_xlim(0, 2 * np.pi)
ax1.set_ylim(-Y, Y)
ax1.set_xlabel("Theta (radians)")
ax1.set_ylabel("Y Coordinate")

ax2.set_xlim(-X, X)
ax2.set_ylim(0, 2 * np.pi)
ax2.set_ylabel("Theta (radians)")
ax2.set_xticks([])

ax3.set_xlim(-X, X)
ax3.set_ylim(-Y, Y)
ax3.set_xlabel("X Coordinate")
ax3.set_yticks([])

# Remove ticks and box from the top-left plot
ax0.set_xticks([])
ax0.set_yticks([])
ax0.spines["top"].set_visible(False)
ax0.spines["right"].set_visible(False)
ax0.spines["bottom"].set_visible(False)
ax0.spines["left"].set_visible(False)

# Initialize the point position
theta = np.linspace(0, 2 * np.pi, 100)
x = np.cos(theta)
y = 2 * np.sin(theta)

(point,) = ax3.plot(x[0], y[0], "ro")  # Plot the moving point
(pointx,) = ax2.plot(x[0], theta[0], "ro")  # Plot the moving point
(pointy,) = ax1.plot(theta[0], y[0], "ro")  # Plot the moving point


start_frame = 10


def update(frame):
    point.set_data(x[frame], y[frame])
    pointx.set_data(x[frame], theta[frame])
    pointy.set_data(theta[frame], y[frame])

    # Plot Y vs. Theta with a moving window
    if frame > start_frame:
        ax1.set_xlim(theta[frame - start_frame], theta[frame])
        ax2.set_ylim(theta[frame - start_frame], theta[frame])

    ax2.plot(x[:frame], theta[:frame], "b-")  # Plot x coordinate vs. theta
    ax1.plot(theta[:frame], y[:frame], "g-")  # Plot y coordinate vs. theta

    ax1.figure.canvas.draw()
    ax2.figure.canvas.draw()
    ax3.figure.canvas.draw()


ani = FuncAnimation(fig, update, frames=len(theta), repeat=False)
plt.tight_layout()

plt.show()
