import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D

def load_data():
    X = np.load("data/ex7_X.npy")
    return X

def draw_line(p1, p2, style="-k", linewidth=1):
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], style, linewidth=linewidth)

def plot_data_points(X, idx):
    # Define colormap to match Figure 1 in the notebook
    cmap = ListedColormap(["red", "green", "blue"])
    c = cmap(idx)
    
    # plots data points in X, coloring them so that those with the same
    # index assignments in idx have the same color
    plt.scatter(X[:, 0], X[:, 1], facecolors='none', edgecolors=c, linewidth=0.1, alpha=0.7)

def plot_progress_kMeans(X, centroids, previous_centroids, idx, K, i):
    # Plot the examples
    plot_data_points(X, idx)
    
    # Plot the centroids as black 'x's
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', c='k', linewidths=3)
    
    # Plot history of the centroids with lines
    for j in range(centroids.shape[0]):
        draw_line(centroids[j, :], previous_centroids[j, :])
    
    plt.title("Iteration number %d" %i)


# def plot_kMeans_RGB(X, centroids, idx, K):
#     # Plot the colors and centroids in a 3D space
#     fig = plt.figure(figsize=(16, 16))
#     ax = fig.add_subplot(221, projection='3d')
#     ax.scatter(*X.T*255, zdir='z', depthshade=False, s=.3, c=X)
#     ax.scatter(*centroids.T*255, zdir='z', depthshade=False, s=500, c='red', marker='x', lw=3)
#     ax.set_xlabel('R value - Redness')
#     ax.set_ylabel('G value - Greenness')
#     ax.set_zlabel('B value - Blueness')
#     ax.w_yaxis.set_pane_color((0., 0., 0., .2))
#     ax.set_title("Original colors and their color clusters' centroids")
#     plt.show()


def plot_kMeans_RGB(X, centroids, idx, K): #This function plots your image’s pixel colors and K-Means centroids in RGB space as a 3D scatter plot.
    '''
    X → array of shape (m, 3) — all pixel RGB values (scaled 0–1 here).
    centroids → array of shape (K, 3) — the RGB values of the cluster centers.
    idx → array of shape (m,) — cluster assignment for each pixel (values from 0 to K-1).
    K → number of clusters.
    '''
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(221, projection='3d')

    # Assign each pixel the color of its centroid
    pixel_colors = centroids[idx] #Example: if pixel 0 belongs to cluster 3, it takes centroids[3] as its display color.

    # Plot pixels
    ax.scatter(*(X.T * 255), zdir='z', depthshade=False, s=0.5, c=pixel_colors)
    '''
    X.T → transposes X to (3, m) so we can unpack into x, y, z.
    *255 → converts RGB values from [0,1] to [0,255] range for plotting.
    zdir='z' and depthshade=False → avoid depth-based shading so colors stay accurate.
    s=0.5 → very small points (each pixel is a tiny dot in RGB space).
    c=pixel_colors → colors each dot according to its cluster’s color.
    '''
    # Plot centroids in red
    ax.scatter(*(centroids.T * 255), zdir='z', depthshade=False,
               s=500, c='red', marker='x', lw=3)

    # Labels
    ax.set_xlabel('R value - Redness')
    ax.set_ylabel('G value - Greenness')
    ax.set_zlabel('B value - Blueness')

    # White background for panes
    ax.xaxis.pane.set_facecolor((1., 1., 1., 1.))
    ax.yaxis.pane.set_facecolor((1., 1., 1., 1.))
    ax.zaxis.pane.set_facecolor((1., 1., 1., 1.))

    ax.set_title("Clustered colors and their centroids")
    plt.show()



def show_centroid_colors(centroids): #centroids → shape (K, 3) — the RGB colors found by K-Means.
    # This function creates a horizontal color palette showing each centroid’s RGB color.
    palette = np.expand_dims(centroids, axis=0)
    '''
    Changes shape from (K, 3) to (1, K, 3).
    This is so imshow treats it as a single row of K color swatches.
    '''
    num = np.arange(0,len(centroids)) #Creates an array [0, 1, 2, ..., K-1] — for x-axis tick labels.
    plt.figure(figsize=(16, 16))
    '''
    Big square figure for visibility.
    X-axis ticks are 0..K-1 (cluster numbers).
    Y-axis ticks removed (not needed).
    '''
    plt.xticks(num)
    plt.yticks([])
    plt.imshow(palette)
    '''
    Displays the palette array as an image.
    Since shape is (1, K, 3), you see one row of K color rectangles — each one is a centroid’s RGB value.
    '''
