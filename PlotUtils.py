import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import torch

def plot_speaker_layout(points, radius=1.0, hemisphere=False, symmetry_axis=None, title="Speaker Layout"):
    """
    Plot speaker positions on a 3D spherical diagram.

    Args:
        points (torch.Tensor or np.ndarray): shape (1, N, 3) or (N, 3)
        radius (float): sphere radius for visual reference
        hemisphere (bool): if True, hides lower half of sphere
        symmetry_axis (str or None): if provided ('x', 'y', 'z'), draw symmetry plane
        title (str): plot title
    """
    if isinstance(points, torch.Tensor):
        points = points.squeeze(0).cpu().numpy()

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)

    # --- Plot the unit sphere surface ---
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi / (2 if hemisphere else 1), 50)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='lightgray', alpha=0.15, linewidth=0)

    # --- Plot speaker points ---
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               color='crimson', s=80, edgecolors='black', depthshade=True, label='Speakers')

    # --- Optionally plot symmetry plane ---
    if symmetry_axis:
        plane_color = (0.4, 0.6, 0.9, 0.15)
        res = np.linspace(-radius, radius, 30)
        if symmetry_axis == 'x':
            X, Y = np.zeros((30, 30)), np.outer(res, np.ones(30))
            Z = np.outer(np.ones(30), res)
            ax.plot_surface(X, Y, Z, color=plane_color)
        elif symmetry_axis == 'y':
            Y = np.zeros((30, 30))
            X, Z = np.meshgrid(res, res)
            ax.plot_surface(X, Y, Z, color=plane_color)
        elif symmetry_axis == 'z':
            Z = np.zeros((30, 30))
            X, Y = np.meshgrid(res, res)
            ax.plot_surface(X, Y, Z, color=plane_color)

    # --- Aesthetics ---
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim([-radius, radius])
    ax.set_ylim([-radius, radius])
    ax.set_zlim([0 if hemisphere else -radius, radius])
    ax.view_init(elev=20, azim=40)
    ax.legend()
    plt.tight_layout()
    plt.show()
