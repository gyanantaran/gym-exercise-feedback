import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import cv2

image = cv2.imread("../docs/slide-deck-mid-sem/mediapipe-landmarks.png")

# Set grid dimensions and subplot size
Y, X, S = 3, 5, 2
start_theta = np.pi
fig = plt.figure(figsize=(X * S, Y * S))
grid = fig.add_gridspec(2, 3)

# Create subplots within the custom grid
ax1 = fig.add_subplot(grid[0, 1:2])
ax2 = fig.add_subplot(grid[1, 1:2])
ax3 = fig.add_subplot(grid[0:2, 0])

# Set limits and labels for subplots
# time vs X graph
ax1.set_xlim(0, 2 * np.pi)
ax1.set_ylim(-X, X)
ax1.set_xticks([])
ax1.set_ylabel("Y Coordinate")

# time vs Y graph
ax2.set_xlim(0, 2 * np.pi)
ax2.set_ylim(-Y, Y)
# ax2.set_xticks([])
ax2.set_xlabel("Theta (radians)")
ax2.set_ylabel("Y Coordinate")

ax3.set_xlim(-X, X)
ax3.set_ylim(-Y, Y)
ax3.set_ylabel("Y Coordinate")
ax3.set_xlabel("X Coordinate")
ax3.set_yticks([])

# Remove ticks and box from the top-left plot
ax1.tick_params(left=False, top=False, right=False, bottom=True)
ax2.tick_params(left=False, top=False, right=False, bottom=True)
ax3.tick_params(left=False, top=False, right=False, bottom=False)

# Initialize the point positions
theta = np.linspace(0, 6 * np.pi, 100)
x = np.cos(theta)
y = 2 * np.sin(theta)

# Plot the moving points
(point,) = ax3.plot(x[0], y[0], "ro")
(pointx,) = ax2.plot(theta[0], x[0], "ro")
(pointy,) = ax1.plot(theta[0], y[0], "ro")


start_frame = 50


def update(frame):
    point.set_data(x[frame], y[frame])
    pointx.set_data(theta[frame], x[frame])
    pointy.set_data(theta[frame], y[frame])

    # Plot Y vs. Theta with a moving window
    if theta[frame] > start_theta:
        ax1.set_xlim(theta[frame] - np.pi, theta[frame] + np.pi)
        ax2.set_xlim(theta[frame] - np.pi, theta[frame] + np.pi)

    ax1.plot(theta[:frame], y[:frame], "g-")  # Plot y coordinate vs. theta
    ax2.plot(theta[:frame], x[:frame], "b-")  # Plot x coordinate vs. theta
    ax3.clear()
    ax3.imshow(image)

    # Redraw the canvas for the updated plot
    ax1.figure.canvas.draw()
    ax2.figure.canvas.draw()


# Create the animation
ani = FuncAnimation(fig, update, frames=len(theta), repeat=False, interval=1)
plt.tight_layout()

# Display the animation
plt.show()
